"""
KYC Verification Backend
Steps: Document Upload → Selfie Upload → Risk Scoring → Internal or Onfido Verification
"""

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import json
import uuid
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
import threading
from gridfs import GridFS
import io
import requests as req

# ── Environment ────────────────────────────────────────────────────────────────
load_dotenv()

app = Flask(__name__)
CORS(app)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-me-in-production")
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "pdf"}

# ── MongoDB ────────────────────────────────────────────────────────────────────
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGODB_DB_NAME", "kyc-app")]
fs = GridFS(db)
applications_col = db["applications"]
audit_col = db["audit_logs"]

# ── Drop stale indexes from old schema ────────────────────────────────────────
# The old app had a unique index on 'email'. Now there's no email field,
# so every insert would collide on email=null. Drop it on startup.
for _stale in ["email_1", "email"]:
    try:
        applications_col.drop_index(_stale)
        print(f"✓ Dropped stale index: {_stale}")
    except Exception:
        pass  # didn't exist — fine

# Ensure application_id stays unique
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
    """Store an uploaded file in GridFS and return its string ID."""
    content = file.read()
    file_id = fs.put(
        content,
        filename=f"{app_id}_{field_name}_{secure_filename(file.filename)}",
        content_type=file.content_type,
        application_id=app_id,
        field_name=field_name,
    )
    return str(file_id)


def get_temp_file(file_id):
    """Pull a file out of GridFS into a local temp path. Caller must delete it."""
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
    """Run Gemini OCR on a document image. Returns dict."""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        prompt = DOC_PROMPTS.get(doc_type, DOC_PROMPTS["national_id"])

        with open(image_path, "rb") as f:
            img_bytes = f.read()

        response = model.generate_content(
            [
                prompt,
                {"mime_type": "image/jpeg", "data": img_bytes},
            ]
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
    """Return structured expiry validation dict."""
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
#  LAZY IMPORTS (avoid loading TensorFlow at startup)
# ══════════════════════════════════════════════════════════════════════════════

def _get_cv2():
    import cv2
    return cv2

def _get_deepface():
    from deepface import DeepFace
    return DeepFace

# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE FORENSICS  (fake-doc detection)
# ══════════════════════════════════════════════════════════════════════════════


def detect_fake_document(image_path):
    """
    Lightweight image-forensics check.
    Returns: {"is_fake": bool, "confidence": 0-100, "reasons": [...]}
    """
    reasons = []
    score = 0

    cv2 = _get_cv2()
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
#  FACE MATCHING & LIVENESS
# ══════════════════════════════════════════════════════════════════════════════


def compare_faces(doc_path, selfie_path):
    try:
        result = _get_deepface().verify(
            img1_path=doc_path,
            img2_path=selfie_path,
            model_name="Facenet",
            enforce_detection=False,
        )
        score = round(max(0, (1 - result["distance"]) * 100), 2)
        return {"match": score >= 60, "score": score, "verified": result["verified"]}
    except Exception as e:
        print(f"Face match error: {e}")
        return {"match": False, "score": 0.0, "error": str(e)}


def check_liveness(selfie_path):
    try:
        cv2 = _get_cv2()
        img = cv2.imread(selfie_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = float(np.mean(gray))
        contrast = float(gray.std())

        s = 0
        s += 40 if sharpness > 100 else (20 if sharpness > 50 else 0)
        s += 30 if 50 < brightness < 200 else (15 if 30 < brightness < 220 else 0)
        s += 30 if contrast > 40 else (15 if contrast > 20 else 0)

        return {
            "score": min(100, s),
            "is_live": s >= 60,
            "sharpness": round(sharpness, 2),
            "brightness": round(brightness, 2),
        }
    except Exception as e:
        print(f"Liveness error: {e}")
        return {"score": 0, "is_live": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  RISK SCORING
#
#  Score 0–100  |  Higher = riskier
#  < 50  → internal (auto-approve)
#  >= 50 → Onfido external verification
#
#  Signals & weights
#  ─────────────────
#  Expired document          +50   (hard signal)
#  Expiry unverifiable       +20
#  Fake-doc confidence ≥ 50  +30   (strong signal)
#  Fake-doc confidence 25-49 +15
#  Face match < 60           +35
#  Face match 60-75          +20
#  Face match 75-85          +10
#  Liveness < 60             +25
#  Liveness 60-75            +10
# ══════════════════════════════════════════════════════════════════════════════


def calculate_risk_score(face_score, liveness_score, fake_doc_confidence, expiry_valid):
    risk = 0

    # Expiry
    if expiry_valid is False:
        risk += 50
    elif expiry_valid is None:
        risk += 20

    # Fake document forensics
    if fake_doc_confidence >= 50:
        risk += 30
    elif fake_doc_confidence >= 25:
        risk += 15

    # Face match (lower match = higher risk)
    if face_score is None:
        risk += 35
    elif face_score < 60:
        risk += 35
    elif face_score < 75:
        risk += 20
    elif face_score < 85:
        risk += 10

    # Liveness
    if liveness_score is None:
        risk += 25
    elif liveness_score < 60:
        risk += 25
    elif liveness_score < 75:
        risk += 10

    return min(100, risk)


# ══════════════════════════════════════════════════════════════════════════════
#  ONFIDO  — external verification triggered when risk >= 50
#  Onfido checks: document authenticity, expiry, face match, fake-doc detection
# ═══════════════════════════════════════════════════════════════════════════���══


def send_to_onfido(app_id, ocr_name):
    """
    Full Onfido verification flow:
      1. Create applicant (name from OCR)
      2. Upload document front + back
      3. Upload selfie as live photo
      4. Create workflow run → Onfido handles expiry, fake-doc, face checks
    """
    headers_json = {
        "Authorization": f"Token token={ONFIDO_API_TOKEN}",
        "Content-Type": "application/json",
    }
    headers_upload = {"Authorization": f"Token token={ONFIDO_API_TOKEN}"}

    record = applications_col.find_one({"application_id": app_id})
    if not record:
        return {"success": False, "error": "Application not found"}

    try:
        # ── 1. Create applicant ────────────────────────────────────────────
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
        print(f"✓ Onfido applicant: {applicant_id}")

        # ── 2. Upload document front & back ────────────────────────────────
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
                print(
                    f"{'✓' if resp.ok else '⚠'} Onfido doc {field}: {'' if resp.ok else resp.text}"
                )
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

        # ── 3. Upload selfie as live photo ─────────────────────────────────
        selfie_fid = record.get("files", {}).get("selfie_photo")
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
                    print(
                        f"{'✓' if resp.ok else '⚠'} Onfido selfie: {'' if resp.ok else resp.text}"
                    )
                finally:
                    if os.path.exists(tmp):
                        os.remove(tmp)

        # ── 4. Create workflow run ─────────────────────────────────────────
        r = req.post(
            f"{ONFIDO_API_URL}/workflow_runs",
            headers=headers_json,
            json={"applicant_id": applicant_id, "workflow_id": ONFIDO_WORKFLOW_ID},
            timeout=30,
        )
        r.raise_for_status()
        wf = r.json()
        print(f"✓ Onfido workflow run: {wf['id']} | status: {wf.get('status')}")

        return {
            "success": True,
            "applicant_id": applicant_id,
            "workflow_run_id": wf["id"],
            "onfido_status": wf.get("status", "processing"),
        }

    except req.exceptions.HTTPError as e:
        body = e.response.text if e.response else str(e)
        print(f"✗ Onfido HTTP error: {body}")
        return {"success": False, "error": str(e), "error_details": body}
    except Exception as e:
        print(f"✗ Onfido error: {e}")
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  VERIFICATION PIPELINE  (runs in background thread)
#
#  Step 1  OCR document front      → name, doc number, expiry date
#  Step 2  Expiry check            → valid / expired / expiring soon
#  Step 3  Fake-doc forensics      → image tampering signals
#  Step 4  Face match              → document photo vs selfie (DeepFace)
#  Step 5  Liveness check          → selfie image quality
#  Step 6  Risk score (0–100)      → combine all signals
#  Step 7  Route
#            ALL pass (risk<50 AND face≥50 AND live≥50) → internal auto-approve
#            ANY fail (risk≥50 OR face<50 OR live<50)   → Onfido external
#            doc expired                                  → hard reject
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

    # ── Step 4: Face match ────────────────────────────────────────────────────
    face_result = {"match": False, "score": None}
    if files.get("document_front") and files.get("selfie_photo"):
        tmp_doc = get_temp_file(files["document_front"])
        tmp_sel = get_temp_file(files["selfie_photo"])
        if tmp_doc and tmp_sel:
            try:
                face_result = compare_faces(tmp_doc, tmp_sel)
            finally:
                for p in (tmp_doc, tmp_sel):
                    if p and os.path.exists(p):
                        os.remove(p)

    # ── Step 5: Liveness ──────────────────────────────────────────────────────
    live_result = {"score": None, "is_live": False}
    if files.get("selfie_photo"):
        tmp = get_temp_file(files["selfie_photo"])
        if tmp:
            try:
                live_result = check_liveness(tmp)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

    # ── Step 6: Risk score ────────────────────────────────────────────────────
    risk_score = calculate_risk_score(
        face_score=face_result.get("score"),
        liveness_score=live_result.get("score"),
        fake_doc_confidence=fake_result.get("confidence", 0),
        expiry_valid=expiry_valid,
    )

    # ── Step 7: Route ─────────────────────────────────────────────────────────
    #
    #  Onfido escalation — triggered if ANY condition fails:
    #    • risk_score  >= 50   (combined weighted risk is high)
    #    • face_match  <  50   (poor identity match between doc and selfie)
    #    • liveness    <  50   (selfie quality / liveness check failed)
    #    • doc expired         (expired doc still goes to Onfido for authoritative check)
    #
    #  Internal auto-approve — only when ALL pass:
    #    • risk_score < 50  AND  face >= 50  AND  liveness >= 50  AND  doc valid
    #
    verification_method = "internal"
    status = "approved"
    rejection_reason = None
    onfido_data = None

    face_score_val = face_result.get("score")  # float or None
    live_score_val = live_result.get("score")  # float or None

    # Collect every failing condition — any failure → Onfido
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
        # One or more signals failed → escalate to Onfido for external authoritative check
        # (covers expired docs, poor face match, low liveness, and high risk score)
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

    # (else: ALL checks passed → internal auto-approve, status stays "approved")

    # ── Persist ───────────────────────────────────────────────────────────────
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
                "liveness": live_result.get("score"),
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
        f"| face={face_result.get('score')} | live={live_result.get('score')} "
        f"| expiry_valid={expiry_valid} | method={verification_method} | status={status}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════


# ── Load country documents configuration ──────────────────────────────────────
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


def validate_document_for_country(country_code, document_id):
    """
    Validates if a document is allowed for a specific country.
    Returns (is_valid, message, doc_info)
    """
    countries = COUNTRY_DOCUMENTS.get("countries", [])
    country = next((c for c in countries if c["code"] == country_code), None)

    if not country:
        return False, f"Country {country_code} not found", None

    all_docs = country.get("identityDocuments", []) + country.get(
        "addressDocuments", []
    )
    doc_info = next((d for d in all_docs if d["id"] == document_id), None)

    if not doc_info:
        return False, f"Document {document_id} not valid for {country_code}", None

    return True, "Valid", doc_info


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/admin")
def admin():
    return render_template("admin.html")


# ── GET /api/countries ────────────────────────────────────────────────────────
@app.route("/api/countries", methods=["GET"])
def list_countries():
    """
    Returns list of supported countries with their metadata.
    Response: { "success": true, "countries": [ { "code": "GB", "name": "...", "region": "..." }, ... ] }
    """
    countries = COUNTRY_DOCUMENTS.get("countries", [])
    country_list = [
        {
            "code": c["code"],
            "name": c["name"],
            "region": c.get("region", "Unknown"),
        }
        for c in countries
    ]
    return jsonify({"success": True, "countries": country_list})


# ── GET /api/countries/<country_code>/documents ────────────────────────────────
@app.route("/api/countries/<country_code>/documents", methods=["GET"])
def get_country_documents(country_code):
    """
    Returns identity and address documents for a specific country.
    Response: {
      "success": true,
      "country": "GB",
      "identity_documents": [ { "id": "...", "name": "...", "description": "..." }, ... ],
      "address_documents": [ { "id": "...", "name": "...", "description": "..." }, ... ]
    }
    """
    countries = COUNTRY_DOCUMENTS.get("countries", [])
    country = next((c for c in countries if c["code"] == country_code.upper()), None)

    if not country:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Country '{country_code}' not found",
                }
            ),
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


# ── POST /api/applications ────────────────────────────────────────────────────
@app.route("/api/applications", methods=["POST"])
def create_application():
    """
    Accepts multipart/form-data with:
      country_code    : ISO 3166-1 alpha-2 (GB, US, CA, etc.) (required)
      document_id     : document ID from country config (required)
      document_type   : passport | drivers_license | national_id | address (required)
      document_front  : image file  (required)
      document_back   : image file  (optional — recommended for ID/license)
      selfie_photo    : image file  (required for identity verification)

    Stores files in GridFS, creates the record, triggers background verification.
    No personal info or address fields are collected.
    """
    try:
        country_code = request.form.get("country_code", "").upper()
        document_id = request.form.get("document_id", "")
        doc_type = request.form.get("document_type", "passport")

        # Validate country and document exist in config
        if not country_code:
            return (
                jsonify({"success": False, "error": "country_code is required"}),
                400,
            )

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

        # Validate document_id exists for this country
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
        if (
            "selfie_photo" not in request.files
            or not request.files["selfie_photo"].filename
        ):
            return jsonify({"success": False, "error": "selfie_photo is required"}), 400

        app_id = generate_app_id()

        # Upload files to GridFS
        file_ids = {}
        for field in ("document_front", "document_back", "selfie_photo"):
            f = request.files.get(field)
            if f and f.filename and allowed_file(f.filename):
                file_ids[field] = save_to_gridfs(f, app_id, field)

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
            f"country={country_code} doc_id={document_id} doc_type={doc_type}",
        )

        # Start verification pipeline in background
        threading.Thread(target=run_verification, args=(app_id,), daemon=True).start()

        return (
            jsonify(
                {
                    "success": True,
                    "application_id": app_id,
                    "message": "Files received. Verification is running.",
                }
            ),
            201,
        )

    except Exception as e:
        import traceback

        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


# ── GET /api/applications/<id> ────────────────────────────────────────────────
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


# ── GET /api/applications ─────────────────────────────────────────────────────
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


# ── POST /api/applications/<id>/review ───────────────────────────────────────
@app.route("/api/applications/<app_id>/review", methods=["POST"])
def review_application(app_id):
    """Admin manual approve / reject."""
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


# ── GET /api/stats ────────────────────────────────────────────────────────────
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


# ── GET /api/files/<file_id> ──────────────────────────────────────────────────
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


# ── POST /api/webhooks/onfido  (Onfido calls this when workflow completes) ─────
@app.route("/api/webhooks/onfido", methods=["POST"])
def onfido_webhook():
    """
    Receives Onfido workflow completion events.
    Onfido POSTs a JSON payload when a workflow run finishes.
    We look up the application by workflow_run_id and update it.
    """
    try:
        payload = request.json or {}
        resource_type = payload.get("resource_type", "")
        action = payload.get("action", "")
        obj = payload.get("object", {})

        workflow_run_id = obj.get("id") or payload.get("workflow_run_id")
        onfido_status = obj.get("status") or payload.get("status", "")
        output = obj.get("output", {}) or {}

        print(
            f"Onfido webhook: resource={resource_type} action={action} run={workflow_run_id} status={onfido_status}"
        )

        if not workflow_run_id:
            return jsonify({"ok": True}), 200

        # Find application by workflow_run_id
        rec = applications_col.find_one(
            {"onfido_data.workflow_run_id": workflow_run_id}
        )
        if not rec:
            print(f"⚠ No application found for workflow_run_id: {workflow_run_id}")
            return jsonify({"ok": True}), 200

        app_id = rec["application_id"]

        # Map Onfido status → our status
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

        print(f"✓ [{app_id}] Onfido result applied: {onfido_status} → {new_status}")
        return jsonify({"ok": True}), 200

    except Exception as e:
        print(f"✗ Onfido webhook error: {e}")
        return jsonify({"ok": True}), 200  # always 200 so Onfido doesn't retry


# ── GET /api/applications/<id>/onfido-result  (poll Onfido API for result) ────
@app.route("/api/applications/<app_id>/onfido-result", methods=["GET"])
def poll_onfido_result(app_id):
    """
    Manually polls the Onfido API for the latest workflow run result
    and updates the application record. Called by admin dashboard.
    """
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

        # Poll Onfido API
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

        # Map to our status
        status_map = {
            "approved": "approved",
            "declined": "rejected",
            "review": "reviewing",
            "abandoned": "reviewing",
            "error": "reviewing",
            "processing": None,  # still running — don't change status
        }
        new_status = status_map.get(onfido_status)

        update = {
            "onfido_data.status": onfido_status,
            "onfido_data.output": output,
            "onfido_data.reasons": reasons,
            "onfido_data.last_polled": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow(),
        }

        if new_status:  # only update app status if Onfido has a final decision
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
            json.dumps(
                {
                    "onfido_status": onfido_status,
                    "new_status": new_status,
                }
            ),
        )

        # Re-fetch updated record
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
    print("KYC VERIFICATION SERVER")
    print("Flow: Upload (doc + selfie) → Risk Score → Internal / Onfido")
    print(f"  risk < 50  → internal auto-approve")
    print(f"  risk >= 50 → Onfido external verification")
    print(f"  Workflow ID: {ONFIDO_WORKFLOW_ID}")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5003)
