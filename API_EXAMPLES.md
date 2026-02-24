# API Examples & Testing Guide

## Testing the Country-Based Verification System

### 1. Get All Countries

#### Request
```bash
curl -X GET http://localhost:5000/api/countries
```

#### Response
```json
{
  "success": true,
  "countries": [
    {
      "code": "GB",
      "name": "United Kingdom",
      "region": "Europe"
    },
    {
      "code": "US",
      "name": "United States",
      "region": "North America"
    },
    {
      "code": "CA",
      "name": "Canada",
      "region": "North America"
    },
    {
      "code": "AU",
      "name": "Australia",
      "region": "Oceania"
    },
    {
      "code": "DE",
      "name": "Germany",
      "region": "Europe"
    },
    {
      "code": "FR",
      "name": "France",
      "region": "Europe"
    },
    {
      "code": "IN",
      "name": "India",
      "region": "Asia"
    },
    {
      "code": "SG",
      "name": "Singapore",
      "region": "Asia"
    },
    {
      "code": "DZ",
      "name": "Algeria",
      "region": "Africa"
    }
  ]
}
```

### 2. Get Documents for Specific Country

#### Request (UK)
```bash
curl -X GET http://localhost:5000/api/countries/GB/documents
```

#### Response
```json
{
  "success": true,
  "country": "GB",
  "identity_documents": [
    {
      "id": "passport",
      "name": "Passport",
      "description": "UK Passport issued by the Home Office",
      "docType": "passport"
    },
    {
      "id": "drivers_license",
      "name": "Driving License",
      "description": "UK/DVLA Driving License",
      "docType": "drivers_license"
    },
    {
      "id": "national_id",
      "name": "National ID Card",
      "description": "UK National Identity Card",
      "docType": "national_id"
    }
  ],
  "address_documents": [
    {
      "id": "utility_bill",
      "name": "Utility Bill",
      "description": "Gas, electricity, water, landline or broadband bill (less than 3 months old)",
      "docType": "address"
    },
    {
      "id": "bank_statement",
      "name": "Bank or Building Society Statement",
      "description": "Bank statement or building society statement (less than 3 months old)",
      "docType": "address"
    },
    {
      "id": "council_tax",
      "name": "Council Tax Letter",
      "description": "Council tax statement or letter from local government (less than 3 months old)",
      "docType": "address"
    },
    {
      "id": "mortgage_statement",
      "name": "Mortgage Statement",
      "description": "Mortgage statement from a financial lending institution (less than 3 months old)",
      "docType": "address"
    }
  ]
}
```

#### Request (Canada - Different Documents)
```bash
curl -X GET http://localhost:5000/api/countries/CA/documents
```

#### Response
```json
{
  "success": true,
  "country": "CA",
  "identity_documents": [
    {
      "id": "passport",
      "name": "Canadian Passport",
      "description": "Canadian Passport",
      "docType": "passport"
    },
    {
      "id": "drivers_license",
      "name": "Driver's License",
      "description": "Provincial/Territorial Driver's License",
      "docType": "drivers_license"
    },
    {
      "id": "national_id",
      "name": "Provincial ID",
      "description": "Provincial/Territorial ID Card",
      "docType": "national_id"
    }
  ],
  "address_documents": [
    {
      "id": "utility_bill",
      "name": "Utility Bill",
      "description": "Gas, electricity, water, or internet bill (less than 3 months old)",
      "docType": "address"
    },
    {
      "id": "bank_statement",
      "name": "Bank Statement",
      "description": "Bank or credit union statement (less than 3 months old)",
      "docType": "address"
    },
    {
      "id": "benefits_letter",
      "name": "Benefits Letter",
      "description": "Government benefits letter (less than 3 months old)",
      "docType": "address"
    }
  ]
}
```

### 3. Submit Application with Country

#### Request (with sample files)
```bash
curl -X POST http://localhost:5000/api/applications \
  -F "country_code=GB" \
  -F "document_id=passport" \
  -F "document_type=passport" \
  -F "document_front=@/path/to/passport_front.jpg" \
  -F "document_back=@/path/to/passport_back.jpg" \
  -F "selfie_photo=@/path/to/selfie.jpg"
```

#### Response
```json
{
  "success": true,
  "application_id": "KYC20260224ABC12345",
  "message": "Files received. Verification is running."
}
```

### 4. Check Application Status

#### Request
```bash
curl -X GET http://localhost:5000/api/applications/KYC20260224ABC12345
```

#### Response (While Processing)
```json
{
  "success": true,
  "application": {
    "application_id": "KYC20260224ABC12345",
    "country_code": "GB",
    "document_id": "passport",
    "document_type": "passport",
    "status": "processing",
    "created_at": "2026-02-24T10:30:00.000Z",
    "updated_at": "2026-02-24T10:30:05.000Z"
  }
}
```

#### Response (After Verification - Approved)
```json
{
  "success": true,
  "application": {
    "application_id": "KYC20260224ABC12345",
    "country_code": "GB",
    "document_id": "passport",
    "document_type": "passport",
    "status": "approved",
    "verification": {
      "risk_score": 25,
      "method": "internal",
      "expiry_valid": true,
      "expiry": {
        "is_valid": true,
        "is_expired": false,
        "expiry_date": "2030-05-15",
        "days_until_expiry": 1536,
        "is_expiring_soon": false
      },
      "fake_document": {
        "is_fake": false,
        "confidence": 8,
        "reasons": []
      },
      "face_match": {
        "match": true,
        "score": 87.5,
        "verified": true
      },
      "liveness": {
        "score": 95,
        "is_live": true,
        "sharpness": 245.3,
        "brightness": 145.2
      }
    },
    "created_at": "2026-02-24T10:30:00.000Z",
    "updated_at": "2026-02-24T10:30:45.000Z",
    "reviewed_at": null
  }
}
```

#### Response (Escalated to Onfido)
```json
{
  "success": true,
  "application": {
    "application_id": "KYC20260224XYZ98765",
    "country_code": "US",
    "document_id": "drivers_license",
    "document_type": "drivers_license",
    "status": "pending_onfido",
    "verification": {
      "risk_score": 62,
      "method": "onfido",
      "face_match": {
        "match": false,
        "score": 45.2
      }
    },
    "onfido_data": {
      "applicant_id": "onfido_app_12345",
      "workflow_run_id": "onfido_run_67890",
      "status": "processing",
      "escalation_reasons": ["face_match_below_threshold"]
    },
    "created_at": "2026-02-24T10:32:00.000Z",
    "updated_at": "2026-02-24T10:32:35.000Z"
  }
}
```

### 5. Test Invalid Country

#### Request
```bash
curl -X GET http://localhost:5000/api/countries/XX/documents
```

#### Response
```json
{
  "success": false,
  "error": "Country 'XX' not found"
}
```

### 6. Submit with Invalid Document for Country

#### Request
```bash
curl -X POST http://localhost:5000/api/applications \
  -F "country_code=GB" \
  -F "document_id=pan_card" \
  -F "document_type=national_id" \
  -F "document_front=@passport.jpg" \
  -F "selfie_photo=@selfie.jpg"
```

#### Response
```json
{
  "success": false,
  "error": "Document 'pan_card' not valid for GB"
}
```

## Testing Scenarios

### Scenario 1: UK User with Passport
```bash
# 1. Get UK documents
curl http://localhost:5000/api/countries/GB/documents

# 2. Submit with passport (valid for UK)
curl -X POST http://localhost:5000/api/applications \
  -F "country_code=GB" \
  -F "document_id=passport" \
  -F "document_type=passport" \
  -F "document_front=@uk_passport.jpg" \
  -F "document_back=@uk_passport_back.jpg" \
  -F "selfie_photo=@selfie.jpg"

# 3. Check status
curl http://localhost:5000/api/applications/KYC20260224ABC12345
```

### Scenario 2: US User with Driver's License
```bash
# 1. Get US documents
curl http://localhost:5000/api/countries/US/documents

# 2. Submit with driver's license (valid for US)
curl -X POST http://localhost:5000/api/applications \
  -F "country_code=US" \
  -F "document_id=drivers_license" \
  -F "document_type=drivers_license" \
  -F "document_front=@us_dl_front.jpg" \
  -F "document_back=@us_dl_back.jpg" \
  -F "selfie_photo=@selfie.jpg"

# 3. Check status
curl http://localhost:5000/api/applications/KYC20260224XYZ12345
```

### Scenario 3: Cross-Validation (Should Fail)
```bash
# Try to submit Indian PAN card to UK (should fail)
curl -X POST http://localhost:5000/api/applications \
  -F "country_code=GB" \
  -F "document_id=pan_card" \
  -F "document_type=national_id" \
  -F "document_front=@pan_card.jpg" \
  -F "selfie_photo=@selfie.jpg"

# Response: Error - pan_card not valid for GB
```

## JavaScript Frontend Example

```javascript
// 1. Load countries on page init
async function loadCountries() {
  const res = await fetch('/api/countries');
  const data = await res.json();
  
  const select = document.getElementById('countrySelect');
  data.countries.forEach(country => {
    const option = document.createElement('option');
    option.value = country.code;
    option.textContent = country.name;
    select.appendChild(option);
  });
}

// 2. When user selects country
async function onCountryChange() {
  const countryCode = document.getElementById('countrySelect').value;
  
  const res = await fetch(`/api/countries/${countryCode}/documents`);
  const data = await res.json();
  
  // Render document buttons
  renderDocumentButtons(data.identity_documents);
}

// 3. Submit form with country
async function submitApplication() {
  const country = document.getElementById('countrySelect').value;
  const documentId = document.getElementById('selectedDocument').value;
  
  const fd = new FormData();
  fd.append('country_code', country);
  fd.append('document_id', documentId);
  fd.append('document_type', 'passport');
  fd.append('document_front', files['document_front']);
  fd.append('selfie_photo', files['selfie_photo']);
  
  const res = await fetch('/api/applications', {
    method: 'POST',
    body: fd
  });
  
  const data = await res.json();
  console.log('Application ID:', data.application_id);
}
```

## Key Differences from Onfido

| Aspect | Onfido | Internal Verification |
|--------|--------|----------------------|
| Documents per country | Yes | Yes (this system) |
| OCR accuracy | High | High (Gemini) |
| Face matching | Yes | Yes (DeepFace) |
| Liveness detection | Yes | Yes (image analysis) |
| API key required | Yes | No |
| Cost per check | $1-5 | $0 (internal) |
| Processing time | 10-30 seconds | 5-10 seconds |
| Approval rate | 70-80% | 70-80% (target) |

## Monitoring & Analytics

### Check approval rates by country
```bash
curl http://localhost:5000/api/applications?status=approved | \
  jq '[.applications[] | .country_code] | group_by(.) | map({country: .[0], count: length})'
```

### Monitor escalation to Onfido
```bash
curl http://localhost:5000/api/applications | \
  jq '[.applications[] | select(.status == "pending_onfido")] | length'
```

### Average risk scores by country
```bash
curl http://localhost:5000/api/applications | \
  jq '[.applications[] | {country: .country_code, risk: .verification.risk_score}] | \
  group_by(.country) | map({country: .[0].country, avg_risk: (map(.risk) | add / length)})'
```

---

**Next:** Test with the provided examples and monitor the metrics to ensure your internal verification rates stay at 70-80% while Onfido costs drop accordingly.
