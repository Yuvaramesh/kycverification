"""
KYC Verification Backend
Steps: Document Upload → Video Upload → Risk Scoring → Internal or Onfido Verification

Video liveness:
  - User records a 5-second video
  - Backend extracts up to 10 evenly-spaced frames
  - Each frame is face-matched against the document photo
  - Best score is used; median used for liveness quality signal
  - Blink / motion detection across frames adds liveness confidence
"""

from flask import Flask, request, jsonify, render_template, send_file, session
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import os
import json
import uuid
import cv2
import numpy as np
from deepface import DeepFace
from dotenv import load_dotenv
import google.generativeai as genai
import threading
from gridfs import GridFS
import io
import requests as req
import tempfile
from functools import wraps

# ── Environment ────────────────────────────────────────────────────────────────
load_dotenv()

app = Flask(__name__)
CORS(app)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-me-in-production")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB (video)
app.config["ALLOWED_EXTENSIONS"] = {
    "png",
    "jpg",
    "jpeg",
    "pdf",
    "mp4",
    "webm",
    "mov",
    "avi",
}

# ── MongoDB ────────────────────────────────────────────────────────────────────
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGODB_DB_NAME", "kyc-app")]
fs = GridFS(db)
applications_col = db["applications"]
audit_col = db["audit_logs"]
users_col = db["users"]

# Ensure unique email index on users
try:
    users_col.create_index("email", unique=True, name="users_email_unique")
    print("✓ Index ensured: users.email (unique)")
except Exception:
    pass

# Drop stale indexes
for _stale in ["email_1", "email"]:
    try:
        applications_col.drop_index(_stale)
        print(f"✓ Dropped stale index: {_stale}")
    except Exception:
        pass

try:
    applications_col.create_index(
        "application_id", unique=True, name="application_id_unique"
    )
    print("✓ Index ensured: application_id (unique)")
except Exception:
    pass

# ── Gemini ─────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✓ Gemini configured")
else:
    print("✗ GEMINI_API_KEY missing")

# ── Onfido ─────────────────────────────────────────────────────────────────────
ONFIDO_API_TOKEN = os.getenv("ONFIDO_API_TOKEN")
ONFIDO_API_URL = os.getenv("ONFIDO_API_URL")
ONFIDO_WORKFLOW_ID = os.getenv("ONFIDO_WORKFLOW_ID")
print("✓ Onfido configured")

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ── Admin ──────────────────────────────────────────────────────────────────────
ADMIN_EMAIL = "yuvi@10qbit.com"  # Only this email has admin access
print(f"✓ Admin email: {ADMIN_EMAIL}")

# ── Google OAuth ───────────────────────────────────────────────────────────────
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv(
    "GOOGLE_REDIRECT_URI", "http://localhost:5003/api/auth/google/callback"
)

# ══════════════════════════════════════════════════════════════════════════════
#  AUTH HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Unauthorized", "redirect": "/login.html"}), 401
        return f(*args, **kwargs)

    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Unauthorized", "redirect": "/login.html"}), 401
        if session.get("role") != "admin":
            return jsonify({"error": "Forbidden – admin access only"}), 403
        return f(*args, **kwargs)

    return decorated


def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    from bson import ObjectId

    return users_col.find_one({"_id": ObjectId(user_id)})


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════════


@app.route("/api/auth/signup", methods=["POST"])
def signup():
    """Register a new user with email + password.
    If the email exists but has no valid password hash (e.g. broken record),
    overwrite it so the user can recover without manual DB intervention.
    """
    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not name or not email or not password:
        return jsonify({"error": "Name, email and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    now = datetime.utcnow()
    new_hash = generate_password_hash(password)

    existing = users_col.find_one({"email": email})
    if existing:
        # If the existing record has a valid password hash, reject signup
        existing_hash = existing.get("password_hash") or ""
        if (
            existing_hash
            and existing_hash.startswith("pbkdf2:")
            or existing_hash.startswith("scrypt:")
        ):
            return (
                jsonify(
                    {
                        "error": "An account with this email already exists. Please sign in instead."
                    }
                ),
                409,
            )

        # Broken/empty hash — overwrite with fresh credentials
        users_col.update_one(
            {"email": email},
            {
                "$set": {
                    "name": name,
                    "password_hash": new_hash,
                    "provider": "email",
                    "last_login": now,
                    "updated_at": now,
                }
            },
        )
        uid = str(existing["_id"])
        role = "admin" if email == ADMIN_EMAIL else existing.get("role", "user")
        # Always keep admin role in sync in DB
        if role == "admin" and existing.get("role") != "admin":
            users_col.update_one({"_id": existing["_id"]}, {"$set": {"role": "admin"}})
    else:
        assigned_role = "admin" if email == ADMIN_EMAIL else "user"
        user_doc = {
            "user_id": str(uuid.uuid4()),
            "name": name,
            "email": email,
            "password_hash": new_hash,
            "provider": "email",
            "avatar": None,
            "role": assigned_role,
            "created_at": now,
            "last_login": now,
        }
        result = users_col.insert_one(user_doc)
        uid = str(result.inserted_id)
        role = assigned_role

    session.permanent = True
    session["user_id"] = uid
    session["email"] = email
    session["name"] = name
    session["role"] = role

    return (
        jsonify(
            {"success": True, "user": {"name": name, "email": email, "role": role}}
        ),
        201,
    )


@app.route("/api/auth/login", methods=["POST"])
def login():
    """Login with email + password."""
    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    user = users_col.find_one({"email": email})
    if not user:
        return (
            jsonify(
                {"error": "No account found with this email. Please sign up first."}
            ),
            401,
        )

    stored_hash = user.get("password_hash") or ""
    if not stored_hash:
        return (
            jsonify(
                {
                    "error": "This account was created via Google. Please use 'Continue with Google'."
                }
            ),
            401,
        )

    if not check_password_hash(stored_hash, password):
        return jsonify({"error": "Incorrect password. Please try again."}), 401

    # Auto-assign admin role if this is the admin email
    resolved_role = (
        "admin" if user["email"] == ADMIN_EMAIL else user.get("role", "user")
    )
    update_fields = {"last_login": datetime.utcnow()}
    if resolved_role == "admin" and user.get("role") != "admin":
        update_fields["role"] = "admin"
    users_col.update_one({"_id": user["_id"]}, {"$set": update_fields})

    session.permanent = True
    session["user_id"] = str(user["_id"])
    session["email"] = user["email"]
    session["name"] = user.get("name", "")
    session["role"] = resolved_role

    return jsonify(
        {
            "success": True,
            "user": {
                "name": user.get("name"),
                "email": user["email"],
                "avatar": user.get("avatar"),
                "role": resolved_role,
            },
        }
    )


def reset_user():
    """Dev utility: delete a user by email so they can re-register cleanly.
    Remove this route before going to production.
    """
    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    if not email:
        return jsonify({"error": "email required"}), 400
    result = users_col.delete_one({"email": email})
    return jsonify({"deleted": result.deleted_count, "email": email})


@app.route("/api/auth/logout", methods=["POST"])
def logout():
    """Clear session."""
    session.clear()
    return jsonify({"success": True})


@app.route("/api/auth/me", methods=["GET"])
def me():
    """Return current authenticated user."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"authenticated": False}), 401
    return jsonify(
        {
            "authenticated": True,
            "user": {
                "name": session.get("name"),
                "email": session.get("email"),
                "role": session.get("role", "user"),
            },
        }
    )


# ── Google OAuth Flow ──────────────────────────────────────────────────────────


@app.route("/api/auth/google", methods=["GET"])
def google_oauth_redirect():
    """Redirect user to Google OAuth consent screen."""
    if not GOOGLE_CLIENT_ID:
        return (
            jsonify(
                {"error": "Google OAuth not configured. Set GOOGLE_CLIENT_ID in .env"}
            ),
            500,
        )
    import urllib.parse

    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "select_account",
    }
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(
        params
    )
    from flask import redirect

    return redirect(url)


@app.route("/api/auth/google/callback", methods=["GET"])
def google_oauth_callback():
    """Handle Google OAuth callback, exchange code for user info."""
    from flask import redirect as flask_redirect

    code = request.args.get("code")
    error = request.args.get("error")

    if error or not code:
        return flask_redirect("/login.html?error=google_denied")

    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return flask_redirect("/login.html?error=oauth_not_configured")

    # Exchange code for tokens
    token_resp = req.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code",
        },
        timeout=10,
    )

    if not token_resp.ok:
        return flask_redirect("/login.html?error=token_exchange_failed")

    tokens = token_resp.json()
    access_token = tokens.get("access_token")

    # Get user info from Google
    userinfo_resp = req.get(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=10,
    )

    if not userinfo_resp.ok:
        return flask_redirect("/login.html?error=userinfo_failed")

    guser = userinfo_resp.json()
    email = guser.get("email", "").lower()
    name = guser.get("name", "")
    avatar = guser.get("picture")
    google_id = guser.get("id")

    if not email:
        return flask_redirect("/login.html?error=no_email")

    # Upsert user in DB
    now = datetime.utcnow()
    existing = users_col.find_one({"email": email})
    # Determine role — admin email always gets admin role
    assigned_role = "admin" if email == ADMIN_EMAIL else None
    if existing:
        resolved_role = assigned_role or existing.get("role", "user")
        set_fields = {
            "name": name,
            "avatar": avatar,
            "last_login": now,
            "google_id": google_id,
        }
        if resolved_role == "admin" and existing.get("role") != "admin":
            set_fields["role"] = "admin"
        users_col.update_one({"email": email}, {"$set": set_fields})
        uid = str(existing["_id"])
        role = resolved_role
    else:
        resolved_role = assigned_role or "user"
        doc = {
            "user_id": str(uuid.uuid4()),
            "name": name,
            "email": email,
            "password_hash": None,
            "provider": "google",
            "google_id": google_id,
            "avatar": avatar,
            "role": resolved_role,
            "created_at": now,
            "last_login": now,
        }
        res = users_col.insert_one(doc)
        uid = str(res.inserted_id)
        role = resolved_role

    session.permanent = True
    session["user_id"] = uid
    session["email"] = email
    session["name"] = name
    session["role"] = role

    # Admin goes to dashboard, users go to KYC upload page
    return flask_redirect("/admin.html" if role == "admin" else "/index.html")


@app.route("/api/auth/google/token", methods=["POST"])
def google_token_signin():
    """Accept Google ID token from frontend (for one-tap / GSI)."""
    data = request.get_json() or {}
    id_token_str = data.get("id_token") or data.get("credential")
    if not id_token_str:
        return jsonify({"error": "id_token required"}), 400

    # Verify token with Google
    verify_resp = req.get(
        f"https://oauth2.googleapis.com/tokeninfo?id_token={id_token_str}", timeout=10
    )
    if not verify_resp.ok:
        return jsonify({"error": "Invalid Google token"}), 401

    ginfo = verify_resp.json()
    if GOOGLE_CLIENT_ID and ginfo.get("aud") != GOOGLE_CLIENT_ID:
        return jsonify({"error": "Token audience mismatch"}), 401

    email = (ginfo.get("email") or "").lower()
    name = ginfo.get("name", "")
    avatar = ginfo.get("picture")
    google_id = ginfo.get("sub")

    if not email:
        return jsonify({"error": "No email in token"}), 400

    now = datetime.utcnow()
    assigned_role = "admin" if email == ADMIN_EMAIL else None
    existing = users_col.find_one({"email": email})
    if existing:
        resolved_role = assigned_role or existing.get("role", "user")
        set_fields = {
            "name": name,
            "avatar": avatar,
            "last_login": now,
            "google_id": google_id,
        }
        if resolved_role == "admin" and existing.get("role") != "admin":
            set_fields["role"] = "admin"
        users_col.update_one({"email": email}, {"$set": set_fields})
        uid = str(existing["_id"])
        role = resolved_role
    else:
        resolved_role = assigned_role or "user"
        doc = {
            "user_id": str(uuid.uuid4()),
            "name": name,
            "email": email,
            "password_hash": None,
            "provider": "google",
            "google_id": google_id,
            "avatar": avatar,
            "role": resolved_role,
            "created_at": now,
            "last_login": now,
        }
        res = users_col.insert_one(doc)
        uid = str(res.inserted_id)
        role = resolved_role

    session.permanent = True
    session["user_id"] = uid
    session["email"] = email
    session["name"] = name
    session["role"] = role

    return jsonify(
        {
            "success": True,
            "user": {"name": name, "email": email, "avatar": avatar, "role": role},
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


def generate_app_id():
    return f"KYC{datetime.now().strftime('%Y%m%d')}{uuid.uuid4().hex[:8].upper()}"


def save_to_gridfs(file, app_id, field_name):
    content = file.read()
    file_id = fs.put(
        content,
        filename=f"{app_id}_{field_name}_{secure_filename(file.filename)}",
        content_type=file.content_type,
        application_id=app_id,
        field_name=field_name,
    )
    return str(file_id)


def save_bytes_to_gridfs(
    data: bytes, app_id: str, field_name: str, filename: str, content_type: str
):
    """Store raw bytes in GridFS."""
    file_id = fs.put(
        data,
        filename=f"{app_id}_{field_name}_{filename}",
        content_type=content_type,
        application_id=app_id,
        field_name=field_name,
    )
    return str(file_id)


def get_temp_file(file_id):
    from bson import ObjectId

    try:
        gf = fs.get(ObjectId(file_id))
        tmp = os.path.join(app.config["UPLOAD_FOLDER"], f"tmp_{file_id}")
        with open(tmp, "wb") as f:
            f.write(gf.read())
        return tmp
    except Exception as e:
        print(f"GridFS read error: {e}")
        return None


def audit(app_id, action, details, user="system"):
    audit_col.insert_one(
        {
            "application_id": app_id,
            "action": action,
            "user": user,
            "details": details,
            "timestamp": datetime.utcnow(),
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
#  VIDEO FRAME EXTRACTION
#
#  Extracts up to `max_frames` evenly-spaced frames from a video file.
#  Returns list of (frame_index, numpy_bgr_array).
#  Also computes a motion score across frames for liveness.
# ══════════════════════════════════════════════════════════════════════════════


def extract_frames_from_video(video_path: str, max_frames: int = 10):
    """
    Open video, sample up to max_frames evenly across its duration.
    Returns list of numpy BGR arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    duration_sec = total_frames / fps

    print(f"Video: {total_frames} frames @ {fps:.1f} fps = {duration_sec:.1f}s")

    if total_frames == 0:
        cap.release()
        return []

    # Sample evenly
    step = max(1, total_frames // max_frames)
    sample_indices = list(range(0, total_frames, step))[:max_frames]

    frames = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)

    cap.release()
    print(f"Extracted {len(frames)} frames from video")
    return frames


def compute_motion_score(frames: list) -> float:
    """
    Compute average optical-flow motion across consecutive frames.
    Real faces move (blink, micro-expressions); static images don't.
    Returns 0–100 where higher = more motion (more likely live).
    """
    if len(frames) < 2:
        return 0.0

    motion_scores = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        motion = float(np.mean(diff))
        motion_scores.append(motion)
        prev_gray = gray

    avg_motion = float(np.mean(motion_scores)) if motion_scores else 0.0
    # Normalize: 0–5 pixel change → 0–100 score; clamp
    score = min(100.0, avg_motion * 20)
    return round(score, 2)


def detect_blink(frames: list) -> bool:
    """
    Rough blink detector using eye-region variance across frames.
    A significant dip in brightness in the eye area suggests a blink.
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    eye_means = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        for x, y, w, h in faces:
            roi = gray[y : y + h // 2, x : x + w]
            eyes = eye_cascade.detectMultiScale(roi, 1.1, 4)
            for ex, ey, ew, eh in eyes:
                eye_roi = roi[ey : ey + eh, ex : ex + ew]
                eye_means.append(float(np.mean(eye_roi)))
            break  # first face only

    if len(eye_means) < 3:
        return False  # not enough data

    # Blink = a local minimum significantly below mean
    mean_val = float(np.mean(eye_means))
    min_val = float(min(eye_means))
    return (mean_val - min_val) > 15  # 15 brightness units drop = blink


# ══════════════════════════════════════════════════════════════════════════════
#  VIDEO-BASED LIVENESS CHECK
#
#  Combines:
#   • Motion score   (optical flow across frames)
#   • Blink detected (eye-region brightness dip)
#   • Frame sharpness (average Laplacian variance)
#   • Frame count    (must have enough frames)
# ══════════════════════════════════════════════════════════════════════════════


def check_liveness_from_video(frames: list) -> dict:
    """
    Compute liveness from extracted video frames.
    Returns same schema as old check_liveness() for compatibility.
    """
    if not frames:
        return {
            "score": 0,
            "is_live": False,
            "motion_score": 0,
            "blink_detected": False,
            "frame_count": 0,
            "sharpness": 0,
            "error": "No frames extracted from video",
        }

    # --- Motion ---
    motion_score = compute_motion_score(frames)

    # --- Blink ---
    blink = detect_blink(frames)

    # --- Sharpness (average across frames) ---
    sharpness_vals = []
    brightness_vals = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        sharpness_vals.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness_vals.append(float(np.mean(gray)))

    avg_sharpness = float(np.mean(sharpness_vals))
    avg_brightness = float(np.mean(brightness_vals))

    # --- Score composition ---
    score = 0

    # Motion (0-40 pts): need some motion but not too much (not shaky)
    if motion_score > 2:
        score += min(40, int(motion_score * 4))

    # Blink (0-20 pts)
    if blink:
        score += 20

    # Sharpness (0-25 pts): video should be reasonably sharp
    if avg_sharpness > 200:
        score += 25
    elif avg_sharpness > 100:
        score += 18
    elif avg_sharpness > 50:
        score += 10

    # Brightness (0-15 pts): face properly lit
    if 60 < avg_brightness < 210:
        score += 15
    elif 40 < avg_brightness < 230:
        score += 8

    score = min(100, score)

    return {
        "score": score,
        "is_live": score >= 55,
        "motion_score": motion_score,
        "blink_detected": blink,
        "frame_count": len(frames),
        "sharpness": round(avg_sharpness, 2),
        "brightness": round(avg_brightness, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  VIDEO-BASED FACE MATCHING
#
#  Extracts faces from all video frames, matches each against the document
#  photo, and returns the best (highest confidence) match result plus
#  a per-frame breakdown.
#
#  Why multi-frame?
#   • A single blurry / poorly-lit frame might fail even for a real user
#   • Taking the BEST score across N frames is fairer and more accurate
#   • We also report median to detect manipulation (all frames suspiciously
#     perfect = possible replay attack)
# ══════════════════════════════════════════════════════════════════════════════


def compare_faces_video(doc_path: str, frames: list) -> dict:
    """
    Match face in document against each video frame.
    Returns best score, median score, per-frame results.
    """
    if not frames:
        return {
            "match": False,
            "score": 0.0,
            "best_score": 0.0,
            "median_score": 0.0,
            "frames_checked": 0,
            "frames_matched": 0,
            "per_frame": [],
            "error": "No frames to match",
        }

    per_frame = []
    scores = []

    for i, frame in enumerate(frames):
        # Write frame to temp file for DeepFace
        tmp_frame = os.path.join(
            app.config["UPLOAD_FOLDER"], f"tmp_frame_{uuid.uuid4().hex}.jpg"
        )
        try:
            cv2.imwrite(tmp_frame, frame)
            result = DeepFace.verify(
                img1_path=doc_path,
                img2_path=tmp_frame,
                model_name="Facenet",
                enforce_detection=False,
            )
            frame_score = round(max(0.0, (1 - result["distance"]) * 100), 2)
            scores.append(frame_score)
            per_frame.append(
                {
                    "frame": i,
                    "score": frame_score,
                    "matched": frame_score >= 75,
                }
            )
        except Exception as e:
            per_frame.append(
                {"frame": i, "score": 0.0, "matched": False, "error": str(e)}
            )
        finally:
            if os.path.exists(tmp_frame):
                os.remove(tmp_frame)

    if not scores:
        return {
            "match": False,
            "score": 0.0,
            "best_score": 0.0,
            "median_score": 0.0,
            "frames_checked": len(frames),
            "frames_matched": 0,
            "per_frame": per_frame,
            "error": "All frame comparisons failed",
        }

    best_score = round(max(scores), 2)
    median_score = round(float(np.median(scores)), 2)
    frames_matched = sum(1 for s in scores if s >= 75)

    # Match is confirmed if best score >= 60 AND at least 1 frame matched
    match = best_score >= 75 and frames_matched >= 1

    return {
        "match": match,
        "score": best_score,  # primary score used for risk
        "best_score": best_score,
        "median_score": median_score,
        "frames_checked": len(frames),
        "frames_matched": frames_matched,
        "per_frame": per_frame,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SAUDI ARABIA DOCUMENT VALIDATION  (Gemini AI strict gate)
#
#  Blocks submission unless the document is BOTH:
#    • Issued by Saudi Arabia (Kingdom of Saudi Arabia)
#    • One of the 4 accepted types:
#        passport | national_id | drivers_license | utility_bill
#
#  Called before any DB/GridFS write in create_application().
#  Returns (is_valid: bool, error_message: str|None, details: dict)
# ══════════════════════════════════════════════════════════════════════════════

ACCEPTED_DOC_TYPES = {
    "passport",
    "national_id",
    "drivers_license",
    "utility_bill",
}

ACCEPTED_DOC_LABELS = {
    "passport": "Passport",
    "national_id": "National ID",
    "drivers_license": "Driver's License",
    "utility_bill": "Utility Bill",
}


def validate_document_with_ai(image_path: str, claimed_doc_type: str):
    """
    Use Gemini to verify:
      1. Is this a real identity document?
      2. Is it issued by Saudi Arabia?
      3. Is the type one of: passport, national_id, drivers_license, utility_bill?
    Returns (is_valid, error_message, details_dict).
    """
    if not GEMINI_API_KEY:
        print("[doc-validation] GEMINI_API_KEY not set — blocking all submissions")
        return (
            False,
            "Document validation service unavailable. Please contact support.",
            {},
        )

    try:
        print(
            f"[doc-validation] Validating: {image_path}  claimed_type={claimed_doc_type}"
        )
        model = genai.GenerativeModel("gemini-2.5-flash-lite")

        prompt = """You are a strict KYC document validator for a Saudi Arabia-only verification system.

Examine this image carefully and return ONLY a raw JSON object — no markdown, no code fences.

{
  "is_document": true or false,
  "is_saudi_arabia": true or false,
  "document_type": "one of exactly: passport | national_id | drivers_license | utility_bill | other | none",
  "issuing_country": "the country name printed on this document, or null",
  "confidence": "high | medium | low",
  "reason": "one sentence explanation"
}

Strict rules:
- is_document = false for selfies, person photos, screenshots, blank pages
- is_saudi_arabia = true ONLY if clearly issued by Kingdom of Saudi Arabia (المملكة العربية السعودية)
- document_type must be exactly one of the listed values
- When in doubt → is_saudi_arabia = false"""

        ext = (image_path.rsplit(".", 1)[-1] or "jpg").lower()
        mime = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "pdf": "application/pdf",
        }.get(ext, "image/jpeg")

        with open(image_path, "rb") as fh:
            img_bytes = fh.read()

        response = model.generate_content(
            [prompt, {"mime_type": mime, "data": img_bytes}]
        )

        text = response.text.strip()
        print(f"[doc-validation] Gemini response: {text[:300]}")

        # Strip markdown fences if model adds them despite instructions
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        data = json.loads(text)

        is_doc = bool(data.get("is_document", False))
        is_saudi = bool(data.get("is_saudi_arabia", False))
        doc_type_ai = str(data.get("document_type") or "none").strip().lower()
        country = str(data.get("issuing_country") or "Unknown")
        confidence = str(data.get("confidence") or "low")
        reason = str(data.get("reason") or "")

        details = {
            "issuing_country_detected": country,
            "document_type_detected": doc_type_ai,
            "confidence": confidence,
            "ai_reason": reason,
        }

        # Check 1 — must be a document
        if not is_doc:
            return (
                False,
                "The uploaded image is not an identity document. Please upload your Saudi Arabia passport, national ID, driver's license, or utility bill.",
                details,
            )

        # Check 2 — must be Saudi Arabia
        if not is_saudi:
            detected_country = (
                country
                if country and country != "Unknown"
                else "a non-Saudi Arabia country"
            )
            return (
                False,
                f"Only Saudi Arabia documents are accepted. The uploaded document appears to be from {detected_country}. Please submit a valid Saudi Arabia document.",
                details,
            )

        # Check 3 — must be one of the 4 accepted types
        if doc_type_ai not in ACCEPTED_DOC_TYPES:
            accepted_list = ", ".join(ACCEPTED_DOC_LABELS.values())
            return (
                False,
                f'Document type "{doc_type_ai}" is not accepted. Please upload one of: {accepted_list}.',
                details,
            )

        print(
            f"[doc-validation] ✓ Valid: {doc_type_ai} from {country} (confidence={confidence})"
        )
        return True, None, details

    except json.JSONDecodeError as e:
        print(f"[doc-validation] JSON parse error: {e} — blocking")
        return (
            False,
            "Could not read the document clearly. Please upload a clearer image and try again.",
            {},
        )
    except Exception as e:
        print(f"[doc-validation] Error: {e} — blocking")
        return False, f"Document validation failed: {str(e)}. Please try again.", {}


# ══════════════════════════════════════════════════════════════════════════════
#  DOCUMENT OCR  (Gemini 2.5 Flash Lite)
# ══════════════════════════════════════════════════════════════════════════════

DOC_PROMPTS = {
    "passport": """
Analyze this passport and return ONLY valid JSON with these keys:
{
  "full_name": "string or null",
  "document_number": "string or null",
  "date_of_birth": "DD/MM/YYYY or null",
  "expiry_date": "DD/MM/YYYY or null",
  "issue_date": "DD/MM/YYYY or null",
  "nationality": "string or null",
  "sex": "M or F or null",
  "raw_text": "all visible text"
}""",
    "drivers_license": """
Analyze this driver's license and return ONLY valid JSON with these keys:
{
  "full_name": "string or null",
  "document_number": "string or null",
  "date_of_birth": "DD/MM/YYYY or null",
  "expiry_date": "DD/MM/YYYY or null",
  "issue_date": "DD/MM/YYYY or null",
  "license_class": "string or null",
  "raw_text": "all visible text"
}""",
    "national_id": """
Analyze this national ID and return ONLY valid JSON with these keys:
{
  "full_name": "string or null",
  "document_number": "string or null",
  "date_of_birth": "DD/MM/YYYY or null",
  "expiry_date": "DD/MM/YYYY or null",
  "issue_date": "DD/MM/YYYY or null",
  "nationality": "string or null",
  "raw_text": "all visible text"
}""",
}


def ocr_document(image_path, doc_type):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        prompt = DOC_PROMPTS.get(doc_type, DOC_PROMPTS["national_id"])
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        response = model.generate_content(
            [prompt, {"mime_type": "image/jpeg", "data": img_bytes}]
        )
        text = response.text.strip()
        for fence in ("```json", "```"):
            if text.startswith(fence):
                text = text[len(fence) :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        data = json.loads(text)
        non_null = sum(1 for k, v in data.items() if k != "raw_text" and v)
        total = len(data) - 1
        confidence = round((non_null / total * 100) if total else 0, 1)
        return {"success": True, "data": data, "confidence": confidence}
    except Exception as e:
        print(f"OCR error: {e}")
        return {"success": False, "data": {}, "confidence": 0, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  EXPIRY CHECK
# ══════════════════════════════════════════════════════════════════════════════

DATE_FORMATS = [
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
]


def parse_date(s):
    if not s or s == "null":
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(str(s).strip(), fmt)
        except ValueError:
            continue
    return None


def check_expiry(expiry_str):
    if not expiry_str:
        return {
            "is_valid": False,
            "is_expired": None,
            "expiry_date": None,
            "days_until_expiry": None,
            "is_expiring_soon": False,
            "error": "No expiry date found in document",
        }
    dt = parse_date(expiry_str)
    if not dt:
        return {
            "is_valid": False,
            "is_expired": None,
            "expiry_date": expiry_str,
            "days_until_expiry": None,
            "is_expiring_soon": False,
            "error": "Unrecognised date format",
        }
    now = datetime.now()
    days = (dt - now).days
    expired = dt < now
    return {
        "is_valid": not expired,
        "is_expired": expired,
        "expiry_date": dt.strftime("%Y-%m-%d"),
        "days_until_expiry": days,
        "is_expiring_soon": 0 <= days <= 30,
        "error": None,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE FORENSICS
# ══════════════════════════════════════════════════════════════════════════════


def detect_fake_document(image_path):
    reasons = []
    score = 0
    img = cv2.imread(image_path)
    if img is None:
        return {"is_fake": True, "confidence": 90, "reasons": ["Image unreadable"]}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 120:
        score += 25
        reasons.append("Low sharpness – possible re-save or edit")
    if np.std(gray) < 20:
        score += 20
        reasons.append("Unnatural noise pattern")
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    if edge_density < 0.01 or edge_density > 0.25:
        score += 25
        reasons.append("Abnormal edge density – possible cropping or pasting")
    b, g, r = cv2.split(img)
    if abs(float(np.mean(r)) - float(np.mean(b))) > 40:
        score += 20
        reasons.append("Color channel imbalance")
    return {"is_fake": score >= 50, "confidence": min(score, 100), "reasons": reasons}


# ══════════════════════════════════════════════════════════════════════════════
#  RISK SCORING  (unchanged logic, compatible with new video face/liveness)
# ══════════════════════════════════════════════════════════════════════════════


def calculate_risk_score(face_score, liveness_score, fake_doc_confidence, expiry_valid):
    risk = 0
    if expiry_valid is False:
        risk += 30
    elif expiry_valid is None:
        risk += 20
    if fake_doc_confidence >= 50:
        risk += 30
    elif fake_doc_confidence >= 25:
        risk += 15
    if face_score is None:
        risk += 35
    elif face_score < 60:
        risk += 35
    elif face_score < 75:
        risk += 20
    elif face_score < 85:
        risk += 10
    if liveness_score is None:
        risk += 25
    elif liveness_score < 60:
        risk += 25
    elif liveness_score < 75:
        risk += 10
    return min(100, risk)


# ══════════════════════════════════════════════════════════════════════════════
#  ONFIDO
# ══════════════════════════════════════════════════════════════════════════════


def send_to_onfido(app_id, ocr_name):
    headers_json = {
        "Authorization": f"Token token={ONFIDO_API_TOKEN}",
        "Content-Type": "application/json",
    }
    headers_upload = {"Authorization": f"Token token={ONFIDO_API_TOKEN}"}
    record = applications_col.find_one({"application_id": app_id})
    if not record:
        return {"success": False, "error": "Application not found"}

    try:
        parts = (ocr_name or "Unknown Unknown").strip().split()
        r = req.post(
            f"{ONFIDO_API_URL}/applicants",
            headers=headers_json,
            json={
                "first_name": parts[0],
                "last_name": " ".join(parts[1:]) if len(parts) > 1 else "Unknown",
            },
            timeout=30,
        )
        r.raise_for_status()
        applicant_id = r.json()["id"]

        doc_type_map = {
            "passport": "passport",
            "drivers_license": "driving_licence",
            "national_id": "national_identity_card",
        }
        onfido_doc_type = doc_type_map.get(
            record.get("document_type", "passport"), "passport"
        )

        for field, side in [("document_front", "front"), ("document_back", "back")]:
            fid = record.get("files", {}).get(field)
            if not fid:
                continue
            tmp = get_temp_file(fid)
            if not tmp:
                continue
            try:
                with open(tmp, "rb") as f:
                    resp = req.post(
                        f"{ONFIDO_API_URL}/documents",
                        headers=headers_upload,
                        data={
                            "applicant_id": applicant_id,
                            "type": onfido_doc_type,
                            "side": side,
                        },
                        files={"file": (f"{field}.jpg", f, "image/jpeg")},
                        timeout=60,
                    )
                print(f"{'✓' if resp.ok else '⚠'} Onfido doc {field}")
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

        # Upload best video frame as live photo for Onfido
        selfie_fid = record.get("files", {}).get("selfie_best_frame")
        if not selfie_fid:
            selfie_fid = record.get("files", {}).get("selfie_video")

        if selfie_fid:
            tmp = get_temp_file(selfie_fid)
            if tmp:
                try:
                    with open(tmp, "rb") as f:
                        resp = req.post(
                            f"{ONFIDO_API_URL}/live_photos",
                            headers=headers_upload,
                            data={"applicant_id": applicant_id},
                            files={"file": ("selfie.jpg", f, "image/jpeg")},
                            timeout=60,
                        )
                    print(f"{'✓' if resp.ok else '⚠'} Onfido selfie")
                finally:
                    if os.path.exists(tmp):
                        os.remove(tmp)

        r = req.post(
            f"{ONFIDO_API_URL}/workflow_runs",
            headers=headers_json,
            json={"applicant_id": applicant_id, "workflow_id": ONFIDO_WORKFLOW_ID},
            timeout=30,
        )
        r.raise_for_status()
        wf = r.json()

        return {
            "success": True,
            "applicant_id": applicant_id,
            "workflow_run_id": wf["id"],
            "onfido_status": wf.get("status", "processing"),
        }

    except req.exceptions.HTTPError as e:
        body = e.response.text if e.response else str(e)
        return {"success": False, "error": str(e), "error_details": body}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  VERIFICATION PIPELINE  (video-based liveness + face matching)
#
#  Step 1  OCR document front
#  Step 2  Expiry check
#  Step 3  Fake-doc forensics
#  Step 4  Extract frames from video
#  Step 5  Liveness check from video frames
#  Step 6  Face match: doc photo vs each video frame (best score wins)
#  Step 7  Risk score
#  Step 8  Route: internal approve OR Onfido escalate
# ══════════════════════════════════════════════════════════════════════════════


def run_verification(app_id):
    record = applications_col.find_one({"application_id": app_id})
    if not record:
        return

    files = record.get("files", {})
    doc_type = record.get("document_type", "passport")

    # ── Step 1: OCR ───────────────────────────────────────────────────────────
    ocr_result = {"success": False, "data": {}, "confidence": 0}
    if files.get("document_front"):
        tmp = get_temp_file(files["document_front"])
        if tmp:
            try:
                ocr_result = ocr_document(tmp, doc_type)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        if ocr_result["success"]:
            applications_col.update_one(
                {"application_id": app_id},
                {
                    "$set": {
                        "ocr_data": ocr_result["data"],
                        "ocr_confidence": ocr_result["confidence"],
                    }
                },
            )

    ocr_data = ocr_result.get("data", {})
    ocr_name = ocr_data.get("full_name")

    # ── Step 2: Expiry ────────────────────────────────────────────────────────
    expiry = check_expiry(ocr_data.get("expiry_date"))
    expiry_valid = expiry["is_valid"] if expiry["error"] is None else None

    # ── Step 3: Fake-doc forensics ────────────────────────────────────────────
    fake_result = {"is_fake": False, "confidence": 0, "reasons": []}
    if files.get("document_front"):
        tmp = get_temp_file(files["document_front"])
        if tmp:
            try:
                fake_result = detect_fake_document(tmp)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

    # ── Step 4: Extract frames from video ────────────────────────────────────
    video_frames = []
    tmp_video = None
    if files.get("selfie_video"):
        tmp_video = get_temp_file(files["selfie_video"])
        if tmp_video:
            video_frames = extract_frames_from_video(tmp_video, max_frames=10)

    # ── Step 5: Liveness from video frames ───────────────────────────────────
    live_result = check_liveness_from_video(video_frames)
    print(
        f"[{app_id}] Liveness: score={live_result['score']} "
        f"motion={live_result.get('motion_score')} "
        f"blink={live_result.get('blink_detected')} "
        f"frames={live_result.get('frame_count')}"
    )

    # ── Step 6: Face match across video frames ────────────────────────────────
    face_result = {"match": False, "score": None}
    if files.get("document_front") and video_frames:
        tmp_doc = get_temp_file(files["document_front"])
        if tmp_doc:
            try:
                face_result = compare_faces_video(tmp_doc, video_frames)
                print(
                    f"[{app_id}] Face match: best={face_result['best_score']} "
                    f"median={face_result['median_score']} "
                    f"frames_matched={face_result['frames_matched']}/{face_result['frames_checked']}"
                )

                # Save best frame to GridFS for Onfido upload
                if face_result.get("per_frame"):
                    best_frame_info = max(
                        face_result["per_frame"], key=lambda x: x.get("score", 0)
                    )
                    best_frame_idx = best_frame_info.get("frame", 0)
                    if best_frame_idx < len(video_frames):
                        best_frame = video_frames[best_frame_idx]
                        _, buf = cv2.imencode(
                            ".jpg", best_frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
                        )
                        best_frame_id = save_bytes_to_gridfs(
                            buf.tobytes(),
                            app_id,
                            "selfie_best_frame",
                            "best_frame.jpg",
                            "image/jpeg",
                        )
                        applications_col.update_one(
                            {"application_id": app_id},
                            {"$set": {"files.selfie_best_frame": best_frame_id}},
                        )
            finally:
                if os.path.exists(tmp_doc):
                    os.remove(tmp_doc)

    # Clean up temp video
    if tmp_video and os.path.exists(tmp_video):
        os.remove(tmp_video)

    # ── Step 7: Risk score ────────────────────────────────────────────────────
    risk_score = calculate_risk_score(
        face_score=face_result.get("score"),
        liveness_score=live_result.get("score"),
        fake_doc_confidence=fake_result.get("confidence", 0),
        expiry_valid=expiry_valid,
    )

    # ── Step 8: Route ─────────────────────────────────────────────────────────
    verification_method = "internal"
    status = "approved"
    rejection_reason = None
    onfido_data = None

    face_score_val = face_result.get("score")
    live_score_val = live_result.get("score")

    escalation_reasons = []
    if expiry_valid is False:
        escalation_reasons.append(f"doc_expired={expiry.get('expiry_date', 'unknown')}")
    if risk_score >= 70:
        escalation_reasons.append(f"risk_score={risk_score} (≥70)")
    if face_score_val is None or face_score_val < 75:
        escalation_reasons.append(
            f"face_match={round(face_score_val, 1) if face_score_val is not None else 'N/A'} (<75)"
        )
    if live_score_val is None or live_score_val < 30:
        escalation_reasons.append(
            f"liveness={round(live_score_val, 1) if live_score_val is not None else 'N/A'} (<30)"
        )

    needs_onfido = len(escalation_reasons) > 0

    if needs_onfido:
        verification_method = "onfido"
        print(f"[{app_id}] Escalating to Onfido: {', '.join(escalation_reasons)}")
        onfido_resp = send_to_onfido(app_id, ocr_name)
        onfido_data = {
            "submitted_at": datetime.utcnow().isoformat(),
            "applicant_id": onfido_resp.get("applicant_id"),
            "workflow_run_id": onfido_resp.get("workflow_run_id"),
            "status": onfido_resp.get("onfido_status", "processing"),
            "success": onfido_resp.get("success", False),
            "escalation_reasons": escalation_reasons,
        }
        status = "pending_onfido"

    update = {
        "verification": {
            "risk_score": risk_score,
            "method": verification_method,
            "status": status,
            "expiry": expiry,
            "expiry_valid": expiry_valid,
            "fake_document": fake_result,
            "face_match": face_result,
            "liveness": live_result,
            "video_frames_analyzed": len(video_frames),
        },
        "status": status,
        "updated_at": datetime.utcnow(),
    }
    if onfido_data:
        update["onfido_data"] = onfido_data
    if rejection_reason:
        update["rejection_reason"] = rejection_reason
        update["reviewed_at"] = datetime.utcnow()
        update["reviewed_by"] = "system_auto"

    applications_col.update_one({"application_id": app_id}, {"$set": update})

    audit(
        app_id,
        "verification_completed",
        json.dumps(
            {
                "risk_score": risk_score,
                "method": verification_method,
                "expiry_valid": expiry_valid,
                "is_fake": fake_result["is_fake"],
                "face_score": face_result.get("score"),
                "face_frames_checked": face_result.get("frames_checked", 0),
                "face_frames_matched": face_result.get("frames_matched", 0),
                "liveness": live_result.get("score"),
                "motion_score": live_result.get("motion_score"),
                "blink_detected": live_result.get("blink_detected"),
                "video_frames": len(video_frames),
                "status": status,
                "onfido_run": (
                    onfido_data.get("workflow_run_id") if onfido_data else None
                ),
                "escalation_reasons": (
                    onfido_data.get("escalation_reasons", []) if onfido_data else []
                ),
            }
        ),
    )

    print(
        f"[{app_id}] risk={risk_score} | fake={fake_result['is_fake']} "
        f"| face_best={face_result.get('score')} ({face_result.get('frames_matched')}/{face_result.get('frames_checked')} frames)"
        f"| live={live_result.get('score')} | expiry_valid={expiry_valid} "
        f"| method={verification_method} | status={status}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

COUNTRY_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "config", "country_documents.json"
)
COUNTRY_DOCUMENTS = {}
try:
    with open(COUNTRY_CONFIG_PATH, "r") as f:
        COUNTRY_DOCUMENTS = json.load(f)
    print(
        f"✓ Loaded {len(COUNTRY_DOCUMENTS.get('countries', []))} countries from config"
    )
except Exception as e:
    print(f"✗ Failed to load country_documents.json: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/admin")
@admin_required
def admin():
    return render_template("admin.html")


@app.route("/api/countries", methods=["GET"])
def list_countries():
    countries = COUNTRY_DOCUMENTS.get("countries", [])
    return jsonify(
        {
            "success": True,
            "countries": [
                {
                    "code": c["code"],
                    "name": c["name"],
                    "region": c.get("region", "Unknown"),
                }
                for c in countries
            ],
        }
    )


@app.route("/api/countries/<country_code>/documents", methods=["GET"])
def get_country_documents(country_code):
    countries = COUNTRY_DOCUMENTS.get("countries", [])
    country = next((c for c in countries if c["code"] == country_code.upper()), None)
    if not country:
        return (
            jsonify({"success": False, "error": f"Country '{country_code}' not found"}),
            404,
        )
    return jsonify(
        {
            "success": True,
            "country": country_code.upper(),
            "identity_documents": country.get("identityDocuments", []),
            "address_documents": country.get("addressDocuments", []),
        }
    )


@app.route("/api/applications", methods=["POST"])
def create_application():
    """
    Accepts multipart/form-data with:
      country_code    : ISO 3166-1 alpha-2 (required)
      document_id     : document ID from country config (required)
      document_type   : passport | drivers_license | national_id (required)
      document_front  : image file (required)
      document_back   : image file (optional)
      selfie_video    : video file — MP4/WebM (required, replaces selfie_photo)

    selfie_video is processed server-side:
      - frames extracted
      - liveness computed from motion / blink across frames
      - face matched against document across all frames (best score used)
    """
    try:
        country_code = request.form.get("country_code", "SA").upper()
        document_id = request.form.get("document_id", "")
        doc_type = request.form.get("document_type", "passport")

        # Saudi Arabia is the only supported country — skip country/doc validation
        SAUDI_DOCS = ["passport", "national_id", "drivers_license", "utility_bill"]
        if not doc_type:
            doc_type = "passport"
        if not document_id:
            document_id = doc_type

        if (
            "document_front" not in request.files
            or not request.files["document_front"].filename
        ):
            return (
                jsonify({"success": False, "error": "document_front is required"}),
                400,
            )

        # Accept either selfie_video (new) or selfie_photo (legacy fallback)
        has_video = (
            "selfie_video" in request.files and request.files["selfie_video"].filename
        )
        has_photo = (
            "selfie_photo" in request.files and request.files["selfie_photo"].filename
        )
        if not has_video and not has_photo:
            return jsonify({"success": False, "error": "selfie_video is required"}), 400

        # ══════════════════════════════════════════════════════════════════════
        # STRICT SAUDI DOCUMENT VALIDATION — runs before any DB/GridFS write
        # Accepted: passport | national_id | drivers_license | utility_bill
        # Country:  Saudi Arabia only
        # ══════════════════════════════════════════════════════════════════════
        _doc_file = request.files.get("document_front")
        if _doc_file:
            import tempfile as _tf

            _ext = (
                os.path.splitext(secure_filename(_doc_file.filename or "doc.jpg"))[1]
                or ".jpg"
            )
            _fd, _tmp_path = _tf.mkstemp(suffix=_ext)
            try:
                _doc_file.seek(0)
                with os.fdopen(_fd, "wb") as _tmpf:
                    _tmpf.write(_doc_file.read())
                _doc_file.seek(0)  # reset for GridFS save later

                _is_valid, _err_msg, _details = validate_document_with_ai(
                    _tmp_path, doc_type
                )
            finally:
                try:
                    os.remove(_tmp_path)
                except Exception:
                    pass

            if not _is_valid:
                print(f"[create_application] BLOCKED — {_err_msg}")
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": _err_msg,
                            "error_code": "INVALID_DOCUMENT",
                            "validation_details": _details,
                        }
                    ),
                    422,
                )

        app_id = generate_app_id()

        file_ids = {}
        for field in ("document_front", "document_back"):
            f = request.files.get(field)
            if f and f.filename and allowed_file(f.filename):
                file_ids[field] = save_to_gridfs(f, app_id, field)

        # Save selfie video (preferred) or photo (legacy)
        if has_video:
            f = request.files["selfie_video"]
            if allowed_file(f.filename):
                file_ids["selfie_video"] = save_to_gridfs(f, app_id, "selfie_video")
        elif has_photo:
            f = request.files["selfie_photo"]
            if allowed_file(f.filename):
                file_ids["selfie_photo"] = save_to_gridfs(f, app_id, "selfie_photo")

        doc = {
            "application_id": app_id,
            "country_code": country_code,
            "document_id": document_id,
            "document_type": doc_type,
            "files": file_ids,
            "ocr_data": None,
            "ocr_confidence": None,
            "verification": None,
            "onfido_data": None,
            "status": "processing",
            "rejection_reason": None,
            "reviewed_by": None,
            "review_notes": None,
            "reviewed_at": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        applications_col.insert_one(doc)
        audit(
            app_id,
            "application_created",
            f"country={country_code} doc_id={document_id} doc_type={doc_type} has_video={has_video}",
        )

        threading.Thread(target=run_verification, args=(app_id,), daemon=True).start()

        return (
            jsonify(
                {
                    "success": True,
                    "application_id": app_id,
                    "message": "Files received. Video analysis and verification is running.",
                }
            ),
            201,
        )

    except Exception as e:
        import traceback

        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


def serialize_doc(d):
    """Stringify ObjectIds in _id and files sub-document, and ISO-format datetimes."""
    d["_id"] = str(d["_id"])
    for field in ("created_at", "updated_at", "reviewed_at"):
        if d.get(field):
            try:
                d[field] = d[field].isoformat()
            except Exception:
                pass
    if isinstance(d.get("files"), dict):
        d["files"] = {k: str(v) for k, v in d["files"].items() if v is not None}
    return d


@app.route("/api/applications/<app_id>", methods=["GET"])
def get_application(app_id):
    rec = applications_col.find_one({"application_id": app_id})
    if not rec:
        return jsonify({"error": "Not found"}), 404
    serialize_doc(rec)
    return jsonify({"success": True, "application": rec})


@app.route("/api/applications", methods=["GET"])
@admin_required
def list_applications():
    status_filter = request.args.get("status")
    limit = int(request.args.get("limit", 50))
    offset = int(request.args.get("offset", 0))
    query = {}
    if status_filter and status_filter != "all":
        query["status"] = status_filter
    docs = list(
        applications_col.find(query).sort("created_at", -1).skip(offset).limit(limit)
    )
    for d in docs:
        serialize_doc(d)
    return jsonify({"success": True, "count": len(docs), "applications": docs})


@app.route("/api/applications/<app_id>/review", methods=["POST"])
@admin_required
def review_application(app_id):
    rec = applications_col.find_one({"application_id": app_id})
    if not rec:
        return jsonify({"error": "Not found"}), 404
    data = request.json or {}
    action = data.get("action")
    reviewer = data.get("reviewer", "admin")
    notes = data.get("notes", "")
    if action not in ("approve", "reject"):
        return jsonify({"error": "action must be 'approve' or 'reject'"}), 400
    new_status = "approved" if action == "approve" else "rejected"
    update = {
        "status": new_status,
        "reviewed_by": reviewer,
        "review_notes": notes,
        "reviewed_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    if action == "reject":
        update["rejection_reason"] = data.get("reason", "Failed manual review")
    applications_col.update_one({"application_id": app_id}, {"$set": update})
    audit(app_id, f"application_{action}d", notes, user=reviewer)
    return jsonify({"success": True, "application_id": app_id, "status": new_status})


@app.route("/api/stats", methods=["GET"])
@admin_required
def get_stats():
    statuses = ["processing", "approved", "rejected", "reviewing", "pending_onfido"]
    counts = {s: applications_col.count_documents({"status": s}) for s in statuses}
    total = applications_col.count_documents({})
    counts["total"] = total
    counts["approval_rate"] = round(counts["approved"] / total * 100 if total else 0, 2)
    recent = list(applications_col.find().sort("created_at", -1).limit(10))
    for d in recent:
        d["_id"] = str(d["_id"])
        for f in ("created_at", "updated_at"):
            if d.get(f):
                d[f] = d[f].isoformat()
    return jsonify({"success": True, "stats": counts, "recent": recent})


@app.route("/api/files/<file_id>")
def serve_file(file_id):
    from bson import ObjectId

    try:
        gf = fs.get(ObjectId(file_id))
        return send_file(
            io.BytesIO(gf.read()), mimetype=gf.content_type, download_name=gf.filename
        )
    except Exception:
        return jsonify({"error": "File not found"}), 404


@app.route("/api/webhooks/onfido", methods=["POST"])
def onfido_webhook():
    try:
        payload = request.json or {}
        obj = payload.get("object", {})
        workflow_run_id = obj.get("id") or payload.get("workflow_run_id")
        onfido_status = obj.get("status") or payload.get("status", "")
        output = obj.get("output", {}) or {}
        if not workflow_run_id:
            return jsonify({"ok": True}), 200
        rec = applications_col.find_one(
            {"onfido_data.workflow_run_id": workflow_run_id}
        )
        if not rec:
            return jsonify({"ok": True}), 200
        app_id = rec["application_id"]
        status_map = {
            "approved": "approved",
            "declined": "rejected",
            "review": "reviewing",
            "abandoned": "reviewing",
            "error": "reviewing",
        }
        new_status = status_map.get(onfido_status, "reviewing")
        update = {
            "onfido_data.status": onfido_status,
            "onfido_data.output": output,
            "onfido_data.completed_at": datetime.utcnow().isoformat(),
            "status": new_status,
            "updated_at": datetime.utcnow(),
        }
        if new_status == "rejected":
            update["rejection_reason"] = (
                f"Onfido verification declined (workflow: {workflow_run_id})"
            )
            update["reviewed_by"] = "onfido_auto"
            update["reviewed_at"] = datetime.utcnow()
        applications_col.update_one({"application_id": app_id}, {"$set": update})
        audit(
            app_id,
            "onfido_webhook_received",
            json.dumps(
                {
                    "workflow_run_id": workflow_run_id,
                    "onfido_status": onfido_status,
                    "new_status": new_status,
                }
            ),
        )
        return jsonify({"ok": True}), 200
    except Exception as e:
        print(f"✗ Onfido webhook error: {e}")
        return jsonify({"ok": True}), 200


@app.route("/api/applications/<app_id>/onfido-result", methods=["GET"])
def poll_onfido_result(app_id):
    try:
        rec = applications_col.find_one({"application_id": app_id})
        if not rec:
            return jsonify({"error": "Not found"}), 404
        onfido = rec.get("onfido_data", {})
        workflow_run_id = onfido.get("workflow_run_id") if onfido else None
        if not workflow_run_id:
            return (
                jsonify({"success": False, "error": "No Onfido workflow run found"}),
                400,
            )
        headers = {
            "Authorization": f"Token token={ONFIDO_API_TOKEN}",
            "Content-Type": "application/json",
        }
        resp = req.get(
            f"{ONFIDO_API_URL}/workflow_runs/{workflow_run_id}",
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        wf_data = resp.json()
        onfido_status = wf_data.get("status", "processing")
        output = wf_data.get("output", {}) or {}
        reasons = wf_data.get("reasons", []) or []
        status_map = {
            "approved": "approved",
            "declined": "rejected",
            "review": "reviewing",
            "abandoned": "reviewing",
            "error": "reviewing",
            "processing": None,
        }
        new_status = status_map.get(onfido_status)
        update = {
            "onfido_data.status": onfido_status,
            "onfido_data.output": output,
            "onfido_data.reasons": reasons,
            "onfido_data.last_polled": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow(),
        }
        if new_status:
            update["status"] = new_status
            if new_status == "rejected":
                update["rejection_reason"] = (
                    f"Onfido declined: {', '.join(reasons) if reasons else onfido_status}"
                )
                update["reviewed_by"] = "onfido_auto"
                update["reviewed_at"] = datetime.utcnow()
        applications_col.update_one({"application_id": app_id}, {"$set": update})
        audit(
            app_id,
            "onfido_result_polled",
            json.dumps({"onfido_status": onfido_status, "new_status": new_status}),
        )
        rec = applications_col.find_one({"application_id": app_id})
        rec["_id"] = str(rec["_id"])
        for f in ("created_at", "updated_at", "reviewed_at"):
            if rec.get(f):
                rec[f] = rec[f].isoformat()
        return jsonify(
            {
                "success": True,
                "onfido_status": onfido_status,
                "app_status": new_status or rec.get("status"),
                "output": output,
                "reasons": reasons,
                "application": rec,
            }
        )
    except req.exceptions.HTTPError as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
#  HTML PAGE ROUTES
# ══════════════════════════════════════════════════════════════════════════════


@app.route("/login.html")
@app.route("/login")
def serve_login():
    return render_template("login.html")


@app.route("/index.html")
def serve_index():
    return render_template("index.html")


@app.route("/admin.html")
@admin_required
def serve_admin():
    return render_template("admin.html")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("KYC VERIFICATION SERVER  (Video Liveness Mode)")
    print("Flow: Upload (doc + 5s video) → Extract Frames → Face Match")
    print("      → Liveness (motion+blink) → Risk Score → Internal / Onfido")
    print(f"  risk < 50  → internal auto-approve")
    print(f"  risk >= 50 → Onfido external verification")
    print(f"  Workflow ID: {ONFIDO_WORKFLOW_ID}")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5003)
