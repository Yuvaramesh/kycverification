# app.py - Complete Line-by-Line Explanation

## Overview
This Flask application implements a KYC (Know Your Customer) verification system that processes documents and selfies through internal AI verification and optionally escalates to Onfido for complex cases.

---

## SECTION 1: IMPORTS & SETUP (Lines 1-77)

### Lines 1-4: Module Docstring
```python
"""
KYC Verification Backend
Steps: Document Upload → Selfie Upload → Risk Scoring → Internal or Onfido Verification
"""
```
Describes the overall workflow of the application.

### Lines 6-22: Library Imports
```python
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
```

**What each import does:**
- `Flask, request, jsonify, render_template, send_file` - Web framework
- `CORS` - Enable cross-origin requests
- `MongoClient` - Connect to MongoDB database
- `secure_filename` - Sanitize uploaded filenames
- `datetime` - Track timestamps
- `os` - Environment variables and file operations
- `json` - Parse/serialize JSON
- `uuid` - Generate unique IDs
- `cv2` - OpenCV for image analysis (fake document detection)
- `numpy` - Numerical operations for image forensics
- `DeepFace` - AI model for facial recognition and comparison
- `load_dotenv` - Load environment variables from .env file
- `genai` - Google Gemini API for OCR
- `threading` - Run verification in background
- `GridFS` - Store large files in MongoDB
- `io` - In-memory file operations
- `requests` - HTTP calls to Onfido API

### Lines 25-33: Flask Configuration
```python
load_dotenv()  # Load .env file

app = Flask(__name__)  # Create Flask app
CORS(app)  # Enable CORS for all routes

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-me-in-production")
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "pdf"}
```

**Line by line:**
- `load_dotenv()` - Reads environment variables from `.env` file (MONGO_URI, GEMINI_API_KEY, ONFIDO_API_TOKEN, etc.)
- `Flask(__name__)` - Initializes Flask application
- `CORS(app)` - Allows frontend on different domain to call backend
- `SECRET_KEY` - For session encryption (defaults to unsafe value, change in production)
- `UPLOAD_FOLDER` - Temporary storage directory
- `MAX_CONTENT_LENGTH` - Max file upload size (16 MB)
- `ALLOWED_EXTENSIONS` - Only these file types can be uploaded

### Lines 36-59: MongoDB Setup
```python
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGODB_DB_NAME", "kyc-app")]
fs = GridFS(db)
applications_col = db["applications"]
audit_col = db["audit_logs"]

# Drop stale indexes from old schema
for _stale in ["email_1", "email"]:
    try:
        applications_col.drop_index(_stale)
        print(f"✓ Dropped stale index: {_stale}")
    except Exception:
        pass

# Ensure application_id stays unique
try:
    applications_col.create_index(
        "application_id", unique=True, name="application_id_unique"
    )
    print("✓ Index ensured: application_id (unique)")
except Exception:
    pass
```

**What this does:**
- Connects to MongoDB using connection string from environment
- Gets database named "kyc-app" (or whatever MONGODB_DB_NAME is set to)
- `GridFS(db)` - Enables large file storage (documents and selfies)
- Gets references to collections: `applications` and `audit_logs`
- Drops old `email` index (from previous schema that stored email)
- Creates unique index on `application_id` to prevent duplicates

### Lines 62-76: External APIs Configuration
```python
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✓ Gemini configured")
else:
    print("✗ GEMINI_API_KEY missing")

ONFIDO_API_TOKEN = os.getenv("ONFIDO_API_TOKEN")
ONFIDO_API_URL = os.getenv("ONFIDO_API_URL")
ONFIDO_WORKFLOW_ID = os.getenv("ONFIDO_WORKFLOW_ID")
print("✓ Onfido configured")

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
```

**What this does:**
- Loads Gemini API key and configures it (used for document OCR)
- Loads Onfido credentials (API token, URL, workflow ID)
- Creates `uploads` folder if it doesn't exist (for temporary file storage)

---

## SECTION 2: UTILITY HELPER FUNCTIONS (Lines 84-133)

### Lines 84-88: allowed_file()
```python
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )
```

**What it does:**
- Checks if uploaded file has an allowed extension (png, jpg, jpeg, pdf)
- `filename.rsplit(".", 1)[1]` - Gets the file extension
- Returns True if extension is in the whitelist

**Example:**
```
allowed_file("passport.jpg") → True
allowed_file("malware.exe") → False
```

### Lines 91-92: generate_app_id()
```python
def generate_app_id():
    return f"KYC{datetime.now().strftime('%Y%m%d')}{uuid.uuid4().hex[:8].upper()}"
```

**What it does:**
- Creates unique application ID with format: `KYC20260224A1B2C3D4`
- `KYC` - Prefix
- `20260224` - Today's date (YYYYMMDD)
- `A1B2C3D4` - First 8 characters of random UUID (uppercase)

**Why:** Date prefix allows chronological sorting, UUID ensures uniqueness

### Lines 95-105: save_to_gridfs()
```python
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
```

**What it does:**
- Stores uploaded file in MongoDB GridFS (can handle files > 16MB)
- `file.read()` - Gets file content as bytes
- `fs.put()` - Stores in MongoDB with metadata
- `secure_filename()` - Removes unsafe characters from filename
- Returns MongoDB ObjectId as string

**Why GridFS:** MongoDB documents have 16MB limit, GridFS splits large files automatically

### Lines 108-120: get_temp_file()
```python
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
```

**What it does:**
- Retrieves file from GridFS using file ID
- Creates temporary file locally (needed for OpenCV and Gemini)
- Returns path to temporary file
- Returns None if file doesn't exist

**Why:** OpenCV and Gemini need local file paths, not streaming data

### Lines 123-132: audit()
```python
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
```

**What it does:**
- Logs every action for compliance/debugging
- Stores: application ID, action name, user who performed it, details, timestamp
- Goes into `audit_logs` collection

**Examples:**
```
action="application_created"
action="verification_completed"
action="onfido_webhook_received"
```

---

## SECTION 3: DOCUMENT OCR (Lines 139-210)

### Lines 139-174: DOC_PROMPTS Dictionary
```python
DOC_PROMPTS = {
    "passport": """Analyze this passport and return ONLY valid JSON...""",
    "drivers_license": """Analyze this driver's license...""",
    "national_id": """Analyze this national ID...""",
}
```

**What it does:**
- Defines prompts for Gemini AI to extract information from documents
- Each document type has different fields to extract
- Prompts explicitly request JSON response format

**Fields extracted:**
- **Passport:** full_name, document_number, date_of_birth, expiry_date, issue_date, nationality, sex
- **Driver's License:** full_name, document_number, date_of_birth, expiry_date, license_class
- **National ID:** full_name, document_number, date_of_birth, expiry_date, nationality

### Lines 177-210: ocr_document()
```python
def ocr_document(image_path, doc_type):
    """Run Gemini OCR on a document image. Returns dict."""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        prompt = DOC_PROMPTS.get(doc_type, DOC_PROMPTS["national_id"])

        with open(image_path, "rb") as f:
            img_bytes = f.read()

        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": img_bytes},
        ])

        text = response.text.strip()
        # Remove markdown code fences
        for fence in ("```json", "```"):
            if text.startswith(fence):
                text = text[len(fence):]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        data = json.loads(text)
        # Calculate confidence
        non_null = sum(1 for k, v in data.items() if k != "raw_text" and v)
        total = len(data) - 1
        confidence = round((non_null / total * 100) if total else 0, 1)

        return {"success": True, "data": data, "confidence": confidence}

    except Exception as e:
        print(f"OCR error: {e}")
        return {"success": False, "data": {}, "confidence": 0, "error": str(e)}
```

**What it does step-by-step:**
1. Load Gemini 2.5 Flash Lite model (fast, cheap for document analysis)
2. Get correct prompt based on document type (passport/license/ID)
3. Read image file as bytes
4. Send to Gemini: prompt + image
5. Parse Gemini response (removes markdown code fences if present)
6. Convert JSON string to Python dict
7. Calculate confidence: % of fields that have values
8. Return results or error

**Example Response:**
```json
{
  "success": true,
  "data": {
    "full_name": "John Doe",
    "document_number": "AB123456",
    "date_of_birth": "01/01/1990",
    "expiry_date": "31/12/2030",
    "raw_text": "... all visible text ..."
  },
  "confidence": 85.7
}
```

---

## SECTION 4: DATE PARSING & EXPIRY CHECK (Lines 217-275)

### Lines 217-228: DATE_FORMATS
```python
DATE_FORMATS = [
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y-%m-%d",
    ...
]
```

**What it does:**
- List of common date formats to try when parsing dates
- Tries each format until one works

### Lines 231-239: parse_date()
```python
def parse_date(s):
    if not s or s == "null":
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(str(s).strip(), fmt)
        except ValueError:
            continue
    return None
```

**What it does:**
- Tries to parse a date string using each format
- Returns Python datetime object if successful
- Returns None if can't parse any format

### Lines 242-275: check_expiry()
```python
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
```

**What it does:**
1. If no expiry date, return error
2. Parse the expiry date string
3. If parsing fails, return error
4. Compare expiry date with today
5. Calculate days until expiry
6. Check if expiring soon (within 30 days)
7. Return structured result

**Example Return:**
```json
{
  "is_valid": true,
  "is_expired": false,
  "expiry_date": "2030-12-31",
  "days_until_expiry": 1807,
  "is_expiring_soon": false,
  "error": null
}
```

---

## SECTION 5: FAKE DOCUMENT DETECTION (Lines 283-317)

### Lines 283-317: detect_fake_document()
```python
def detect_fake_document(image_path):
    """Lightweight image-forensics check."""
    reasons = []
    score = 0

    img = cv2.imread(image_path)
    if img is None:
        return {"is_fake": True, "confidence": 90, "reasons": ["Image unreadable"]}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Check 1: Laplacian sharpness
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 120:
        score += 25
        reasons.append("Low sharpness – possible re-save or edit")

    # Check 2: Noise pattern
    if np.std(gray) < 20:
        score += 20
        reasons.append("Unnatural noise pattern")

    # Check 3: Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    if edge_density < 0.01 or edge_density > 0.25:
        score += 25
        reasons.append("Abnormal edge density")

    # Check 4: Color channel imbalance
    b, g, r = cv2.split(img)
    if abs(float(np.mean(r)) - float(np.mean(b))) > 40:
        score += 20
        reasons.append("Color channel imbalance")

    return {"is_fake": score >= 50, "confidence": min(score, 100), "reasons": reasons}
```

**What it does:**

Detects four signs of fake/tampered documents:

1. **Sharpness (Laplacian variance < 120):**
   - Legitimate documents are sharp
   - Photoshopped/re-saved images are blurry
   - +25 points if blurry

2. **Noise Pattern (std dev < 20):**
   - Real photos have natural noise variations
   - Fake/edited images have unnatural noise
   - +20 points if unnatural

3. **Edge Density (< 1% or > 25%):**
   - Documents have consistent edge patterns
   - Cropped or pasted elements show abnormal edges
   - +25 points if abnormal

4. **Color Channel Imbalance (R-B > 40):**
   - Genuine images have balanced RGB channels
   - Edited images often have color shifts
   - +20 points if imbalanced

**Scoring:**
- < 50 = Genuine document
- >= 50 = Likely fake

---

## SECTION 6: FACE MATCHING & LIVENESS (Lines 325-361)

### Lines 325-337: compare_faces()
```python
def compare_faces(doc_path, selfie_path):
    try:
        result = DeepFace.verify(
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
```

**What it does:**
1. Uses DeepFace (AI face recognition library) to compare two faces
2. `img1_path` = Face in document (passport/ID)
3. `img2_path` = Selfie taken by user
4. Facenet = Advanced face recognition model
5. `enforce_detection=False` = Don't fail if face detection is uncertain
6. Returns distance between faces (0 = identical, 1 = completely different)
7. Converts distance to percentage score (100% = perfect match, 0% = no match)
8. `match` = True if score >= 60% (more than 60% similar)

**Example:**
```json
{
  "match": true,
  "score": 87.5,
  "verified": true
}
```

### Lines 340-361: check_liveness()
```python
def check_liveness(selfie_path):
    try:
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
```

**What it does:**

Checks if selfie is a real, live person (not a photo of a photo):

1. **Sharpness Check (Laplacian variance):**
   - Sharp face = Live person (0-40 points)
   - Blurry face = Possible fake/photo

2. **Brightness Check (mean pixel value):**
   - Optimal: 50-200 (30 points)
   - Acceptable: 30-220 (15 points)
   - Tells if lighting is reasonable

3. **Contrast Check (standard deviation):**
   - High contrast = Real face (30 points)
   - Low contrast = Flat/printed photo (0-15 points)

**Scoring:**
- >= 60 = Likely live person
- < 60 = Possibly fake (printed photo, screen replay, etc.)

---

## SECTION 7: RISK SCORING (Lines 385-418)

### Lines 385-418: calculate_risk_score()
```python
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

    # Face match
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
```

**Scoring System (0-100, higher = riskier):**

| Issue | Points |
|-------|--------|
| Document expired | +50 |
| Expiry date unverifiable | +20 |
| Fake doc confidence >= 50% | +30 |
| Fake doc confidence 25-50% | +15 |
| Face match None (can't detect) | +35 |
| Face match < 60% (poor match) | +35 |
| Face match 60-75% (weak match) | +20 |
| Face match 75-85% (good match) | +10 |
| Liveness None (can't check) | +25 |
| Liveness < 60% (likely fake) | +25 |
| Liveness 60-75% (weak liveness) | +10 |

**Decision:**
- Risk < 50 = Low risk → **Internal auto-approve**
- Risk >= 50 = High risk → **Escalate to Onfido**

---

## SECTION 8: ONFIDO INTEGRATION (Lines 427-543)

### Lines 427-543: send_to_onfido()
```python
def send_to_onfido(app_id, ocr_name):
    """Full Onfido verification flow"""
    headers_json = {
        "Authorization": f"Token token={ONFIDO_API_TOKEN}",
        "Content-Type": "application/json",
    }
    headers_upload = {"Authorization": f"Token token={ONFIDO_API_TOKEN}"}

    record = applications_col.find_one({"application_id": app_id})
    if not record:
        return {"success": False, "error": "Application not found"}

    try:
        # 1. Create applicant
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

        # 2. Upload document front & back
        doc_type_map = {
            "passport": "passport",
            "drivers_license": "driving_licence",
            "national_id": "national_identity_card",
        }
        onfido_doc_type = doc_type_map.get(record.get("document_type"), "passport")

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
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

        # 3. Upload selfie as live photo
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
                finally:
                    if os.path.exists(tmp):
                        os.remove(tmp)

        # 4. Create workflow run
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

    except Exception as e:
        print(f"✗ Onfido error: {e}")
        return {"success": False, "error": str(e)}
```

**What it does step-by-step:**

**Step 1: Create Applicant**
- Split OCR name into first/last name
- POST to `/applicants` endpoint
- Onfido returns `applicant_id`

**Step 2: Upload Documents**
- Maps our document types to Onfido types:
  - passport → passport
  - drivers_license → driving_licence
  - national_id → national_identity_card
- Uploads document front and back separately
- Each has "side" parameter (front/back)

**Step 3: Upload Live Photo**
- Uploads selfie as "live_photo" (Onfido's liveness check)

**Step 4: Create Workflow Run**
- Initiates Onfido verification workflow
- Returns workflow_run_id for tracking

---

## SECTION 9: VERIFICATION PIPELINE (Lines 562-745)

### Lines 562-745: run_verification()

**Overall Flow:**
```
User uploads document + selfie
    ↓
Create application in MongoDB
    ↓
Start background thread running run_verification()
    ↓
Step 1: OCR document → Extract full_name, document_number, expiry_date
    ↓
Step 2: Check expiry → Is document expired?
    ↓
Step 3: Detect fake document → Image forensics analysis
    ↓
Step 4: Compare faces → Document photo vs selfie
    ↓
Step 5: Liveness check → Is selfie a real person?
    ↓
Step 6: Calculate risk score → 0-100
    ↓
Step 7: Route decision
    │
    ├─→ Risk < 50 AND Face >= 50 AND Liveness >= 50 → APPROVED (internal)
    └─→ Risk >= 50 OR Face < 50 OR Liveness < 50 → ESCALATE TO ONFIDO
```

**Step 1: OCR (Lines 570-589)**
```python
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
            {"$set": {"ocr_data": ocr_result["data"], "ocr_confidence": ocr_result["confidence"]}}
        )
```

- Gets document front from GridFS
- Runs Gemini OCR
- Stores extracted data in database
- Deletes temporary file

**Step 2: Expiry Check (Line 595)**
```python
expiry = check_expiry(ocr_data.get("expiry_date"))
expiry_valid = expiry["is_valid"] if expiry["error"] is None else None
```

- Checks if document is expired
- Sets `expiry_valid` to True/False/None

**Step 3: Fake Detection (Lines 600-607)**
```python
fake_result = {"is_fake": False, "confidence": 0, "reasons": []}
if files.get("document_front"):
    tmp = get_temp_file(files["document_front"])
    if tmp:
        try:
            fake_result = detect_fake_document(tmp)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
```

- Analyzes document image for tampering signs
- Returns fake score and reasons

**Step 4: Face Match (Lines 610-620)**
```python
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
```

- Compares face in document with face in selfie
- Returns match score (0-100%)

**Step 5: Liveness (Lines 623-631)**
```python
live_result = {"score": None, "is_live": False}
if files.get("selfie_photo"):
    tmp = get_temp_file(files["selfie_photo"])
    if tmp:
        try:
            live_result = check_liveness(tmp)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
```

- Checks if selfie is genuine (not a printed photo)
- Returns liveness score (0-100%)

**Step 6: Risk Score (Lines 634-639)**
```python
risk_score = calculate_risk_score(
    face_score=face_result.get("score"),
    liveness_score=live_result.get("score"),
    fake_doc_confidence=fake_result.get("confidence", 0),
    expiry_valid=expiry_valid,
)
```

- Combines all signals into single risk score (0-100)

**Step 7: Routing Decision (Lines 641-691)**
```python
escalation_reasons = []
if expiry_valid is False:
    escalation_reasons.append(f"doc_expired={expiry.get('expiry_date')}")
if risk_score >= 50:
    escalation_reasons.append(f"risk_score={risk_score}")
if face_score_val is None or face_score_val < 50:
    escalation_reasons.append(f"face_match={face_score_val} (<50)")
if live_score_val is None or live_score_val < 50:
    escalation_reasons.append(f"liveness={live_score_val} (<50)")

needs_onfido = len(escalation_reasons) > 0

if needs_onfido:
    verification_method = "onfido"
    onfido_resp = send_to_onfido(app_id, ocr_name)
    onfido_data = {...}
    status = "pending_onfido"
```

**Decision Logic:**
- If ANY condition fails → escalate to Onfido
- Otherwise → auto-approve

**Step 8: Store Results (Lines 695-717)**
```python
update = {
    "verification": {
        "risk_score": risk_score,
        "method": verification_method,
        "status": status,
        "expiry": expiry,
        "fake_document": fake_result,
        "face_match": face_result,
        "liveness": live_result,
    },
    "status": status,
    "updated_at": datetime.utcnow(),
}
if onfido_data:
    update["onfido_data"] = onfido_data

applications_col.update_one({"application_id": app_id}, {"$set": update})
```

- Stores all verification results in database
- Stores Onfido data if escalated

---

## SECTION 10: API ROUTES (Lines 786-1259)

### Route 1: GET / (Line 786)
```python
@app.route("/")
def index():
    return render_template("index.html")
```
- Serves frontend HTML file

### Route 2: GET /api/countries (Lines 797-812)
```python
@app.route("/api/countries", methods=["GET"])
def list_countries():
    countries = COUNTRY_DOCUMENTS.get("countries", [])
    country_list = [
        {"code": c["code"], "name": c["name"], "region": c.get("region")}
        for c in countries
    ]
    return jsonify({"success": True, "countries": country_list})
```
- Returns list of supported countries (from country_documents.json)

### Route 3: GET /api/countries/<country_code>/documents (Lines 816-850)
```python
@app.route("/api/countries/<country_code>/documents", methods=["GET"])
def get_country_documents(country_code):
    countries = COUNTRY_DOCUMENTS.get("countries", [])
    country = next((c for c in countries if c["code"] == country_code.upper()), None)
    
    if not country:
        return jsonify({"success": False, "error": "Country not found"}), 404
    
    return jsonify({
        "success": True,
        "country": country_code.upper(),
        "identity_documents": country.get("identityDocuments", []),
        "address_documents": country.get("addressDocuments", []),
    })
```
- Returns documents allowed for a specific country

### Route 4: POST /api/applications (Lines 854-971)
```python
@app.route("/api/applications", methods=["POST"])
def create_application():
    country_code = request.form.get("country_code", "").upper()
    document_id = request.form.get("document_id", "")
    doc_type = request.form.get("document_type", "passport")
    
    # Validate country and document
    countries = COUNTRY_DOCUMENTS.get("countries", [])
    country = next((c for c in countries if c["code"] == country_code), None)
    if not country:
        return jsonify({"success": False, "error": f"Country '{country_code}' not supported"}), 400
    
    # Validate files
    if "document_front" not in request.files:
        return jsonify({"success": False, "error": "document_front is required"}), 400
    if "selfie_photo" not in request.files:
        return jsonify({"success": False, "error": "selfie_photo is required"}), 400
    
    # Generate ID and upload files
    app_id = generate_app_id()
    file_ids = {}
    for field in ("document_front", "document_back", "selfie_photo"):
        f = request.files.get(field)
        if f and f.filename and allowed_file(f.filename):
            file_ids[field] = save_to_gridfs(f, app_id, field)
    
    # Create document
    doc = {
        "application_id": app_id,
        "country_code": country_code,
        "document_id": document_id,
        "document_type": doc_type,
        "files": file_ids,
        "ocr_data": None,
        "verification": None,
        "onfido_data": None,
        "status": "processing",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    applications_col.insert_one(doc)
    
    # Start verification in background
    threading.Thread(target=run_verification, args=(app_id,), daemon=True).start()
    
    return jsonify({
        "success": True,
        "application_id": app_id,
        "message": "Files received. Verification is running.",
    }), 201
```

**What it does:**
1. Validates country and document type
2. Validates required files are uploaded
3. Generates unique application ID
4. Uploads files to GridFS
5. Creates application document in MongoDB
6. Starts verification in background thread
7. Returns application ID to frontend

### Route 5: GET /api/applications/<app_id> (Lines 975-986)
```python
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
```
- Returns single application record with all verification results

### Route 6: GET /api/applications (Lines 990-1009)
```python
@app.route("/api/applications", methods=["GET"])
def list_applications():
    status_filter = request.args.get("status")
    limit = int(request.args.get("limit", 50))
    offset = int(request.args.get("offset", 0))
    
    query = {}
    if status_filter and status_filter != "all":
        query["status"] = status_filter
    
    docs = list(applications_col.find(query).sort("created_at", -1).skip(offset).limit(limit))
    
    return jsonify({"success": True, "count": len(docs), "applications": docs})
```
- Lists applications with optional status filter
- Supports pagination (limit, offset)
- Sorted by creation date (newest first)

### Route 7: POST /api/applications/<app_id>/review (Lines 1013-1042)
```python
@app.route("/api/applications/<app_id>/review", methods=["POST"])
def review_application(app_id):
    rec = applications_col.find_one({"application_id": app_id})
    if not rec:
        return jsonify({"error": "Not found"}), 404
    
    data = request.json or {}
    action = data.get("action")  # "approve" or "reject"
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
```
- Admin endpoint to manually review and approve/reject applications

### Route 8: GET /api/stats (Lines 1046-1061)
```python
@app.route("/api/stats", methods=["GET"])
def get_stats():
    statuses = ["processing", "approved", "rejected", "reviewing", "pending_onfido"]
    counts = {s: applications_col.count_documents({"status": s}) for s in statuses}
    total = applications_col.count_documents({})
    counts["total"] = total
    counts["approval_rate"] = round(counts["approved"] / total * 100 if total else 0, 2)
    
    recent = list(applications_col.find().sort("created_at", -1).limit(10))
    
    return jsonify({"success": True, "stats": counts, "recent": recent})
```
- Returns statistics (approval rate, counts by status, recent applications)

### Route 9: GET /api/files/<file_id> (Lines 1065-1075)
```python
@app.route("/api/files/<file_id>")
def serve_file(file_id):
    try:
        gf = fs.get(ObjectId(file_id))
        return send_file(io.BytesIO(gf.read()), mimetype=gf.content_type, download_name=gf.filename)
    except Exception:
        return jsonify({"error": "File not found"}), 404
```
- Downloads file from GridFS
- Used to view document images and selfies

### Route 10: POST /api/webhooks/onfido (Lines 1079-1156)
```python
@app.route("/api/webhooks/onfido", methods=["POST"])
def onfido_webhook():
    payload = request.json or {}
    resource_type = payload.get("resource_type", "")
    action = payload.get("action", "")
    obj = payload.get("object", {})
    
    workflow_run_id = obj.get("id")
    onfido_status = obj.get("status", "")
    
    # Find application by workflow_run_id
    rec = applications_col.find_one({"onfido_data.workflow_run_id": workflow_run_id})
    if not rec:
        return jsonify({"ok": True}), 200
    
    app_id = rec["application_id"]
    
    # Map Onfido status to our status
    status_map = {
        "approved": "approved",
        "declined": "rejected",
        "review": "reviewing",
        "abandoned": "reviewing",
    }
    new_status = status_map.get(onfido_status, "reviewing")
    
    # Update application
    update = {
        "onfido_data.status": onfido_status,
        "onfido_data.completed_at": datetime.utcnow().isoformat(),
        "status": new_status,
        "updated_at": datetime.utcnow(),
    }
    
    applications_col.update_one({"application_id": app_id}, {"$set": update})
    
    return jsonify({"ok": True}), 200
```
- Webhook that Onfido calls when verification completes
- Updates application status based on Onfido's result

### Route 11: GET /api/applications/<app_id>/onfido-result (Lines 1160-1258)
```python
@app.route("/api/applications/<app_id>/onfido-result", methods=["GET"])
def poll_onfido_result(app_id):
    rec = applications_col.find_one({"application_id": app_id})
    
    onfido = rec.get("onfido_data", {})
    workflow_run_id = onfido.get("workflow_run_id")
    
    if not workflow_run_id:
        return jsonify({"success": False, "error": "No Onfido workflow run found"}), 400
    
    # Poll Onfido API
    headers = {"Authorization": f"Token token={ONFIDO_API_TOKEN}"}
    resp = req.get(f"{ONFIDO_API_URL}/workflow_runs/{workflow_run_id}", headers=headers)
    resp.raise_for_status()
    wf_data = resp.json()
    
    onfido_status = wf_data.get("status", "processing")
    
    # Map to our status
    status_map = {
        "approved": "approved",
        "declined": "rejected",
        "review": "reviewing",
    }
    new_status = status_map.get(onfido_status)
    
    # Update if final decision reached
    if new_status:
        update = {"status": new_status, "updated_at": datetime.utcnow()}
        applications_col.update_one({"application_id": app_id}, {"$set": update})
    
    return jsonify({
        "success": True,
        "onfido_status": onfido_status,
        "app_status": new_status or rec.get("status"),
        "application": rec,
    })
```
- Admin endpoint to manually check Onfido result (polling)
- Used if webhook doesn't come through

---

## SECTION 11: MAIN ENTRY (Lines 1262-1270)

```python
if __name__ == "__main__":
    print("=" * 60)
    print("KYC VERIFICATION SERVER")
    print("Flow: Upload (doc + selfie) → Risk Score → Internal / Onfido")
    print(f"  risk < 50  → internal auto-approve")
    print(f"  risk >= 50 → Onfido external verification")
    print(f"  Workflow ID: {ONFIDO_WORKFLOW_ID}")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5003)
```

- Prints banner when server starts
- Starts Flask on port 5003
- `debug=True` enables auto-reload on code changes
- `host="0.0.0.0"` allows external connections

---

## Summary: Complete Data Flow

```
1. User Upload
   → POST /api/applications with country_code, doc_front, doc_back, selfie
   
2. Validation
   → Check country exists in country_documents.json
   → Check document_id is valid for that country
   
3. File Storage
   → Upload all files to MongoDB GridFS
   → Store file IDs in database
   
4. Background Verification (in separate thread)
   → Step 1: OCR document with Gemini AI
   → Step 2: Check if document is expired
   → Step 3: Analyze for fake document signs
   → Step 4: Compare faces (DeepFace)
   → Step 5: Check liveness (image quality analysis)
   → Step 6: Calculate risk score (0-100)
   → Step 7: Route decision
   
5. Routing Decision
   IF risk < 50 AND face >= 50 AND liveness >= 50:
     → Status = "approved" (internal)
     → Send to admin dashboard
   ELSE:
     → Escalate to Onfido
     → Status = "pending_onfido"
     → Wait for Onfido webhook
   
6. Onfido Webhook
   → Onfido calls /api/webhooks/onfido when complete
   → Updates status: approved / rejected / reviewing
   → Application document in database gets final decision
   
7. Admin Review
   → Admin views application details
   → Can manually approve/reject using /api/applications/<id>/review
   → Can download document files using /api/files/<id>
```

This is the complete architecture of the KYC system!

