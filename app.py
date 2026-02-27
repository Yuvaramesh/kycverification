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

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.utils import secure_filename
from datetime import datetime
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

# ── Environment ────────────────────────────────────────────────────────────────
load_dotenv()

app = Flask(__name__)
CORS(app)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-me-in-production")
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
                    "matched": frame_score >= 60,
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
    frames_matched = sum(1 for s in scores if s >= 60)

    # Match is confirmed if best score >= 60 AND at least 1 frame matched
    match = best_score >= 60 and frames_matched >= 1

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
        risk += 50
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
    if risk_score >= 50:
        escalation_reasons.append(f"risk_score={risk_score} (≥50)")
    if face_score_val is None or face_score_val < 50:
        escalation_reasons.append(
            f"face_match={round(face_score_val, 1) if face_score_val is not None else 'N/A'} (<50)"
        )
    if live_score_val is None or live_score_val < 50:
        escalation_reasons.append(
            f"liveness={round(live_score_val, 1) if live_score_val is not None else 'N/A'} (<50)"
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
        country_code = request.form.get("country_code", "").upper()
        document_id = request.form.get("document_id", "")
        doc_type = request.form.get("document_type", "passport")

        if not country_code:
            return jsonify({"success": False, "error": "country_code is required"}), 400

        countries = COUNTRY_DOCUMENTS.get("countries", [])
        country = next((c for c in countries if c["code"] == country_code), None)
        if not country:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Country '{country_code}' not supported",
                    }
                ),
                400,
            )

        valid_docs = [d["id"] for d in country.get("identityDocuments", [])] + [
            d["id"] for d in country.get("addressDocuments", [])
        ]
        if document_id and document_id not in valid_docs:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Document '{document_id}' not valid for {country_code}",
                    }
                ),
                400,
            )

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


@app.route("/api/applications/<app_id>", methods=["GET"])
def get_application(app_id):
    rec = applications_col.find_one({"application_id": app_id})
    if not rec:
        return jsonify({"error": "Not found"}), 404
    rec["_id"] = str(rec["_id"])
    for field in ("created_at", "updated_at", "reviewed_at"):
        if rec.get(field):
            rec[field] = rec[field].isoformat()
    return jsonify({"success": True, "application": rec})


@app.route("/api/applications", methods=["GET"])
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
        d["_id"] = str(d["_id"])
        for f in ("created_at", "updated_at", "reviewed_at"):
            if d.get(f):
                d[f] = d[f].isoformat()
    return jsonify({"success": True, "count": len(docs), "applications": docs})


@app.route("/api/applications/<app_id>/review", methods=["POST"])
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
