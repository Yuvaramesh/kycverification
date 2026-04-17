"""
sanctions_screening.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Multi-Jurisdiction Sanctions Screening Module for the KYC App
Canada uses FINTRAC-specific AML screening (PCMLTFA-compliant).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW TO INTEGRATE INTO app.py
─────────────────────────────
1.  Copy this file into your project root (same folder as app.py).
2.  Add to app.py imports:
        from sanctions_screening import SanctionsScreener, register_sanctions_routes
3.  After the MongoDB / env setup block, initialise the screener:
        screener = SanctionsScreener(api_key=os.getenv("OPENSANCTIONS_API_KEY"))
4.  Register the Flask routes (call once, after `app` is created):
        register_sanctions_routes(app, screener, applications_col, audit)
5.  Add OPENSANCTIONS_API_KEY=<your-key> to your .env file.
        Get a free non-commercial key at https://www.opensanctions.org/api/

CANADA / FINTRAC COMPLIANCE
─────────────────────────────
When country_code == "CA" the screener automatically switches to FINTRAC mode.
This covers all lists mandated by the Proceeds of Crime (Money Laundering) and
Terrorist Financing Act (PCMLTFA) and FINTRAC guidelines:

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ FINTRAC-mandated list                        │ OpenSanctions dataset        │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ OSFI / Global Affairs Canada – SEMA/DFATD    │ ca_dfatd_sema_sanctions      │
  │ UN SC Consolidated (1267/1989/2253)          │ un_sc_sanctions              │
  │ Criminal Code – Listed Terrorist Entities    │ ca_dfatd_sema_sanctions *    │
  │ Suppression of Terrorism Regulations         │ ca_dfatd_sema_sanctions *    │
  │ OFAC SDN (cross-border best practice)        │ us_ofac_sdn                  │
  │ Interpol Red Notices                         │ interpol_red_notices         │
  │ PEP screening (PCMLTFA s.9.3)               │ peps                         │
  └─────────────────────────────────────────────────────────────────────────────┘
  * SEMA/DFATD dataset on OpenSanctions consolidates Criminal Code + STR lists.

All other countries continue to use the full global sanctions dataset.

DATASET COVERAGE (non-CA)
─────────────────────────────
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ Jurisdiction  │ Datasets included                                        │
  ├──────────────────────────────────────────────────────────────────────────┤
  │ UN            │ UN Security Council Consolidated List                    │
  │ USA           │ OFAC SDN, OFAC Consolidated, BIS Entity List,            │
  │               │ ITAR Debarred, FBI Wanted                                │
  │ EU            │ EU Consolidated Financial Sanctions                      │
  │ UK            │ UK HM Treasury Sanctions List (OFSI)                     │
  │ Australia     │ DFAT Consolidated Sanctions, Listed Terrorist Orgs (ATO) │
  │ Canada        │ Global Affairs Canada SEMA Consolidated                  │
  │ Switzerland   │ SECO Sanctions                                           │
  │ Japan         │ METI / MOFA Foreign End-User List                        │
  │ Interpol      │ Interpol Red Notices                                     │
  │ World Bank    │ Debarred & Cross-Debarred Firms                          │
  └──────────────────────────────────────────────────────────────────────────┘

FLASK ROUTES ADDED
─────────────────────────────
  POST /api/sanctions/screen
        Body: { name, birth_date?, nationality?, application_id? }
        Returns: { hit: bool, score, matches: [...], datasets_checked,
                   fintrac_mode: bool }

  POST /api/applications/<app_id>/sanctions-check
        Runs screening using name/dob stored on the application record.
        Saves results back to MongoDB and triggers audit log.
        Auto-selects FINTRAC mode when application country_code == "CA".

  GET  /api/applications/<app_id>/sanctions-result
        Returns the last saved sanctions screening result.
"""

import os
import logging
from datetime import datetime
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── OpenSanctions dataset IDs ─────────────────────────────────────────────────
DATASET_MAP = {
    "default": "default",
    "sanctions": "sanctions",
    "peps": "peps",
    "un": "un_sc_sanctions",
    "us_ofac": "us_ofac_sdn",
    "us_ofac_cons": "us_ofac_cons",
    "us_bis": "us_bis_entities",
    "eu": "eu_fsf",
    "uk": "gb_hmt_sanctions",
    "au_dfat": "au_dfat_sanctions",
    "au_terror": "au_listed_terrorist_orgs",
    "ca": "ca_dfatd_sema_sanctions",
    "ch": "ch_seco_sanctions",
    "interpol": "interpol_red_notices",
}

# ── FINTRAC-mandated datasets (Canada AML/CFT — PCMLTFA compliance) ───────────
# Covers every list Reporting Entities must screen under the Proceeds of Crime
# (Money Laundering) and Terrorist Financing Act and FINTRAC guidelines.
# Each entry is a separate OpenSanctions /match/<dataset> call so scores are
# isolated per list, then aggregated to the highest score.
FINTRAC_SANCTIONS_DATASETS = [
    "ca_dfatd_sema_sanctions",  # Global Affairs Canada SEMA/DFATD
    #   → consolidates: Criminal Code s.83,
    #     Suppression of Terrorism Regs,
    #     OSFI Consolidated List
    "un_sc_sanctions",  # UN Security Council 1267/1989/2253
    "us_ofac_sdn",  # OFAC SDN (best-practice cross-border check)
    "interpol_red_notices",  # Interpol Red Notices
]
FINTRAC_PEP_DATASET = "peps"  # PCMLTFA s.9.3 PEP obligation

FINTRAC_DATASET_LABELS = [
    "Global Affairs Canada – SEMA/DFATD Consolidated",
    "UN Security Council Consolidated List (1267/1989/2253)",
    "Criminal Code – Listed Terrorist Entities (Canada)",
    "Suppression of Terrorism Regulations (Canada)",
    "OFAC SDN List (cross-border best practice)",
    "Interpol Red Notices",
    "FINTRAC PEP Database (PCMLTFA s.9.3)",
]

# Score thresholds
SCORE_ALERT = 0.70  # flag for review
SCORE_BLOCK = 0.90  # automatic block / further verification needed

BASE_URL = "https://api.opensanctions.org"


class SanctionsScreener:
    """
    Wrapper around the OpenSanctions yente API.

    Usage (generic):
        screener = SanctionsScreener(api_key="YOUR_KEY")
        result   = screener.screen_person("John Smith", birth_date="1975-03-12",
                                          nationality="RU")

    Usage (FINTRAC / Canada):
        result = screener.screen_person_fintrac("Jane Doe",
                                                birth_date="1980-06-15",
                                                nationality="CA")
        # or let full_kyc_screen() auto-route when country_code="CA":
        result = screener.full_kyc_screen("Jane Doe", ..., country_code="CA")
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENSANCTIONS_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "OPENSANCTIONS_API_KEY not set — requests will fail for commercial use. "
                "Non-commercial users: register at https://www.opensanctions.org/api/"
            )
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"ApiKey {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    # ── Core matching call ────────────────────────────────────────────────────

    def _match(
        self, query_entity: dict, dataset: str = "default", threshold: float = 0.50
    ) -> dict:
        """
        Call POST /match/<dataset> with a FollowTheMoney query entity.
        Returns raw API response dict.
        """
        url = f"{BASE_URL}/match/{dataset}"
        payload = {"queries": {"q1": query_entity}}
        try:
            resp = self.session.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("responses", {}).get("q1", {}).get("results", [])
            return {
                "success": True,
                "dataset": dataset,
                "results": results,
                "total": len(results),
            }
        except requests.exceptions.HTTPError as e:
            logger.error("OpenSanctions HTTP error: %s", e)
            return {"success": False, "error": str(e), "results": []}
        except Exception as e:
            logger.error("OpenSanctions error: %s", e)
            return {"success": False, "error": str(e), "results": []}

    # ── Public screening methods ──────────────────────────────────────────────

    def screen_person(
        self,
        name: str,
        birth_date: Optional[str] = None,
        nationality: Optional[str] = None,
        passport_number: Optional[str] = None,
        id_number: Optional[str] = None,
        dataset: str = "default",
    ) -> dict:
        """
        Screen an individual against international sanctions and PEP lists.

        Args:
            name:            Full name (required)
            birth_date:      ISO date string, e.g. "1975-03-12"  (recommended)
            nationality:     ISO-2 country code, e.g. "GB"        (recommended)
            passport_number: Passport number                       (optional)
            id_number:       National ID number                    (optional)
            dataset:         OpenSanctions dataset key             (default: "default")

        Returns dict with:
            hit         – True if any result exceeds SCORE_ALERT threshold
            block       – True if any result exceeds SCORE_BLOCK threshold
            score       – highest score found (0.0–1.0)
            matches     – list of matched entities with details
            datasets_checked – human-readable list of datasets queried
            screened_at – ISO timestamp
        """
        entity: dict = {
            "schema": "Person",
            "properties": {
                "name": [name],
            },
        }
        if birth_date:
            entity["properties"]["birthDate"] = [birth_date]
        if nationality:
            entity["properties"]["nationality"] = [nationality]
        if passport_number:
            entity["properties"]["passportNumber"] = [passport_number]
        if id_number:
            entity["properties"]["idNumber"] = [id_number]

        raw = self._match(entity, dataset=dataset)
        return self._parse_result(raw, name=name)

    def screen_person_fintrac(
        self,
        name: str,
        birth_date: Optional[str] = None,
        nationality: Optional[str] = None,
        passport_number: Optional[str] = None,
        id_number: Optional[str] = None,
    ) -> dict:
        """
        FINTRAC-compliant AML screening for Canadian applicants.

        Screens across every list mandated by the Proceeds of Crime (Money
        Laundering) and Terrorist Financing Act (PCMLTFA) and FINTRAC
        guidelines:
          • Global Affairs Canada SEMA/DFATD Consolidated
            (includes Criminal Code s.83 terrorist entities +
             Suppression of Terrorism Regulations + OSFI list)
          • UN Security Council Consolidated (1267/1989/2253)
          • OFAC SDN (cross-border best practice)
          • Interpol Red Notices
          • FINTRAC PEP database (PCMLTFA s.9.3 obligation)

        Each dataset is queried separately so per-list scores are preserved
        in the result for audit purposes. The overall score is the maximum
        across all datasets.

        Returns same schema as screen_person() with two additional fields:
            fintrac_mode      – True (always, for this method)
            per_dataset_scores – { dataset_label: score } for audit trail
        """
        entity: dict = {
            "schema": "Person",
            "properties": {"name": [name]},
        }
        if birth_date:
            entity["properties"]["birthDate"] = [birth_date]
        if nationality:
            entity["properties"]["nationality"] = [nationality]
        if passport_number:
            entity["properties"]["passportNumber"] = [passport_number]
        if id_number:
            entity["properties"]["idNumber"] = [id_number]

        all_results = []
        per_dataset_scores: dict = {}
        errors = []

        # Query each FINTRAC-mandated sanctions dataset separately
        for ds in FINTRAC_SANCTIONS_DATASETS:
            raw = self._match(entity, dataset=ds)
            if not raw.get("success"):
                errors.append(f"{ds}: {raw.get('error', 'unknown error')}")
                continue
            parsed = self._parse_result(raw, name=name)
            ds_label = self._fintrac_label_for_dataset(ds)
            per_dataset_scores[ds_label] = parsed["score"]
            all_results.extend(raw.get("results", []))

        # Deduplicate by entity id, keep highest score
        seen: dict = {}
        for entity_hit in all_results:
            eid = entity_hit.get("id")
            if eid is None:
                continue
            existing = seen.get(eid)
            if existing is None or entity_hit.get("score", 0) > existing.get(
                "score", 0
            ):
                seen[eid] = entity_hit

        deduped_results = list(seen.values())
        sanctions_parsed = self._parse_result(
            {
                "success": True,
                "dataset": "fintrac_consolidated",
                "results": deduped_results,
            },
            name=name,
        )

        # PEP check (PCMLTFA s.9.3)
        pep_raw = self._match(entity, dataset=FINTRAC_PEP_DATASET)
        pep_parsed = self._parse_result(pep_raw, name=name)
        per_dataset_scores["FINTRAC PEP Database (PCMLTFA s.9.3)"] = pep_parsed["score"]

        return {
            "hit": sanctions_parsed["hit"] or pep_parsed["hit"],
            "block": sanctions_parsed["block"] or pep_parsed["block"],
            "score": sanctions_parsed["score"],
            "matches": sanctions_parsed["matches"],
            "pep_hit": pep_parsed["hit"],
            "pep_score": pep_parsed["score"],
            "pep_matches": pep_parsed["matches"],
            "screened_at": datetime.utcnow().isoformat(),
            "datasets_checked": FINTRAC_DATASET_LABELS,
            "fintrac_mode": True,
            "per_dataset_scores": per_dataset_scores,
            "errors": errors or None,
        }

    def screen_organisation(
        self,
        name: str,
        country: Optional[str] = None,
        registration_number: Optional[str] = None,
        dataset: str = "default",
    ) -> dict:
        """Screen an organisation / company."""
        entity: dict = {
            "schema": "Organization",
            "properties": {
                "name": [name],
            },
        }
        if country:
            entity["properties"]["country"] = [country]
        if registration_number:
            entity["properties"]["registrationNumber"] = [registration_number]

        raw = self._match(entity, dataset=dataset)
        return self._parse_result(raw, name=name)

    def full_kyc_screen(
        self,
        name: str,
        birth_date: Optional[str] = None,
        nationality: Optional[str] = None,
        passport_number: Optional[str] = None,
        id_number: Optional[str] = None,
        country_code: Optional[str] = None,
    ) -> dict:
        """
        Run both sanctions + PEP screening.

        When country_code == "CA" (or nationality == "CA"), automatically
        switches to FINTRAC-compliant screening mode (PCMLTFA-mandated lists).
        All other countries use the standard global sanctions + peps datasets.

        Returns a combined result. In FINTRAC mode the result includes
        fintrac_mode=True and per_dataset_scores for audit purposes.
        """
        # Determine effective country
        effective_country = (country_code or nationality or "").strip().upper()

        if effective_country == "CA":
            # ── FINTRAC path ──────────────────────────────────────────────────
            logger.info("full_kyc_screen: routing to FINTRAC mode for CA applicant")
            result = self.screen_person_fintrac(
                name, birth_date, nationality, passport_number, id_number
            )
            result["compliance_regime"] = "FINTRAC (PCMLTFA)"
            return result

        # ── Standard global path ──────────────────────────────────────────────
        sanctions_result = self.screen_person(
            name,
            birth_date,
            nationality,
            passport_number,
            id_number,
            dataset="sanctions",
        )
        pep_result = self.screen_person(name, birth_date, nationality, dataset="peps")
        combined_hit = sanctions_result["hit"] or pep_result["hit"]
        combined_block = sanctions_result["block"] or pep_result["block"]
        combined_score = max(sanctions_result["score"], pep_result["score"])

        return {
            "hit": combined_hit,
            "block": combined_block,
            "score": combined_score,
            "sanctions_result": sanctions_result,
            "pep_result": pep_result,
            "fintrac_mode": False,
            "compliance_regime": "Global (OpenSanctions default)",
            "screened_at": datetime.utcnow().isoformat(),
            "datasets_checked": self._dataset_labels("sanctions")
            + self._dataset_labels("peps"),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _parse_result(self, raw: dict, name: str = "") -> dict:
        if not raw.get("success"):
            return {
                "hit": False,
                "block": False,
                "score": 0.0,
                "matches": [],
                "error": raw.get("error"),
                "screened_at": datetime.utcnow().isoformat(),
                "datasets_checked": [],
            }

        results = raw.get("results", [])
        parsed_matches = []
        top_score = 0.0

        for entity in results:
            score = entity.get("score", 0.0)
            if score < 0.40:
                continue
            top_score = max(top_score, score)
            props = entity.get("properties", {})

            datasets = entity.get("datasets", [])
            programs = []
            for ds in datasets:
                programs.append(ds)

            parsed_matches.append(
                {
                    "id": entity.get("id"),
                    "name": props.get("name", [name])[0] if props.get("name") else name,
                    "aliases": props.get("alias", []),
                    "score": round(score, 3),
                    "schema": entity.get("schema"),
                    "topics": entity.get("topics", []),
                    "datasets": datasets,
                    "programs": programs,
                    "nationality": props.get("nationality", []),
                    "birth_date": props.get("birthDate", []),
                    "positions": props.get("position", []),
                    "sanctions": [
                        {
                            "authority": s.get("properties", {}).get("authority", [""])[
                                0
                            ],
                            "program": s.get("properties", {}).get("program", [""])[0],
                            "start_date": s.get("properties", {}).get(
                                "startDate", [""]
                            )[0],
                            "reason": s.get("properties", {}).get("reason", [""])[0],
                        }
                        for s in entity.get("referents", {}).get("sanctions", [])
                    ],
                    "entity_url": f"https://www.opensanctions.org/entities/{entity.get('id')}/",
                }
            )

        return {
            "hit": top_score >= SCORE_ALERT,
            "block": top_score >= SCORE_BLOCK,
            "score": round(top_score, 3),
            "matches": sorted(parsed_matches, key=lambda x: -x["score"]),
            "screened_at": datetime.utcnow().isoformat(),
            "datasets_checked": self._dataset_labels(raw.get("dataset", "default")),
        }

    @staticmethod
    def _fintrac_label_for_dataset(ds: str) -> str:
        """Return a human-readable FINTRAC label for a dataset key."""
        return {
            "ca_dfatd_sema_sanctions": (
                "Global Affairs Canada – SEMA/DFATD Consolidated "
                "(incl. Criminal Code s.83 + STR)"
            ),
            "un_sc_sanctions": "UN Security Council Consolidated (1267/1989/2253)",
            "us_ofac_sdn": "OFAC SDN List (cross-border best practice)",
            "interpol_red_notices": "Interpol Red Notices",
        }.get(ds, ds)

    @staticmethod
    def _dataset_labels(dataset_key: str) -> list:
        label_map = {
            "default": [
                "UN Security Council",
                "OFAC SDN",
                "OFAC Consolidated",
                "BIS Entity List",
                "EU Financial Sanctions",
                "UK HM Treasury (OFSI)",
                "Australia DFAT Sanctions",
                "Australia Listed Terrorist Organisations",
                "Canada SEMA",
                "SECO (Switzerland)",
                "Interpol Red Notices",
                "World Bank Debarred",
                "FBI Most Wanted",
                "EU PEPs",
                "UK PEPs",
                "Global PEPs",
            ],
            "sanctions": [
                "UN Security Council",
                "OFAC SDN",
                "OFAC Consolidated",
                "BIS Entity List",
                "EU Financial Sanctions",
                "UK HM Treasury (OFSI)",
                "Australia DFAT Sanctions",
                "Australia Listed Terrorist Organisations",
                "Canada SEMA",
                "SECO (Switzerland)",
                "Interpol Red Notices",
            ],
            "peps": [
                "EU PEPs",
                "UK PEPs",
                "US Congress PEPs",
                "Global Political Exposure Dataset",
            ],
            "fintrac_consolidated": FINTRAC_DATASET_LABELS,
            "au_terror": ["Australia Listed Terrorist Organisations (Home Affairs)"],
            "au_dfat": ["Australia DFAT Consolidated Sanctions"],
            "uk": ["UK HM Treasury Office of Financial Sanctions (OFSI)"],
            "un": ["UN Security Council Consolidated List (1267/1989/2253)"],
            "us_ofac": ["OFAC SDN List"],
            "eu": ["EU Consolidated Financial Sanctions Register"],
        }
        return label_map.get(dataset_key, [dataset_key])


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════════


def register_sanctions_routes(
    app, screener: SanctionsScreener, applications_col, audit_fn
):
    """
    Register /api/sanctions/* routes onto the Flask app.

    Call this from app.py after creating `app`, `screener`, and the collections:

        from sanctions_screening import SanctionsScreener, register_sanctions_routes
        screener = SanctionsScreener(api_key=os.getenv("OPENSANCTIONS_API_KEY"))
        register_sanctions_routes(app, screener, applications_col, audit)
    """
    from flask import request, jsonify, session
    from functools import wraps

    def login_required_inner(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not session.get("user_id"):
                return jsonify({"error": "Unauthorized"}), 401
            return f(*args, **kwargs)

        return decorated

    def admin_required_inner(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not session.get("user_id"):
                return jsonify({"error": "Unauthorized"}), 401
            if session.get("role") != "admin":
                return jsonify({"error": "Admin access only"}), 403
            return f(*args, **kwargs)

        return decorated

    # ── Route 1: Ad-hoc screening ─────────────────────────────────────────────

    @app.route("/api/sanctions/screen", methods=["POST"])
    @login_required_inner
    def sanctions_screen():
        """
        POST /api/sanctions/screen
        Body: {
            name:            "John Smith",              (required)
            birth_date:      "1975-03-12",              (ISO date, recommended)
            nationality:     "CA",                      (ISO-2, recommended)
            passport_number: "AB123456",                (optional)
            id_number:       "NIC-1234",                (optional)
            country_code:    "CA",                      (optional, triggers FINTRAC)
            mode:            "full" | "sanctions" | "peps" | "fintrac"
        }

        When nationality or country_code is "CA", or mode is "fintrac",
        FINTRAC-compliant screening is used automatically.
        """
        data = request.get_json() or {}
        name = (data.get("name") or "").strip()
        if not name:
            return jsonify({"error": "name is required"}), 400

        birth_date = data.get("birth_date") or data.get("birthDate")
        nationality = (data.get("nationality") or "").strip().upper()
        country_code = (data.get("country_code") or "").strip().upper()
        passport_number = data.get("passport_number")
        id_number = data.get("id_number")
        mode = data.get("mode", "full")

        # FINTRAC auto-routing
        effective_country = country_code or nationality
        use_fintrac = effective_country == "CA" or mode == "fintrac"

        try:
            if use_fintrac:
                result = screener.screen_person_fintrac(
                    name,
                    birth_date,
                    nationality or None,
                    passport_number,
                    id_number,
                )
            elif mode == "full":
                result = screener.full_kyc_screen(
                    name,
                    birth_date,
                    nationality or None,
                    passport_number,
                    id_number,
                    country_code=effective_country or None,
                )
            elif mode == "peps":
                result = screener.screen_person(
                    name, birth_date, nationality or None, dataset="peps"
                )
            else:
                result = screener.screen_person(
                    name,
                    birth_date,
                    nationality or None,
                    passport_number,
                    id_number,
                    dataset="sanctions",
                )
        except Exception as e:
            logger.error("Sanctions screen error: %s", e)
            return jsonify({"error": str(e)}), 500

        return jsonify({"success": True, "query": name, **result})

    # ── Route 2: Screen an existing KYC application ───────────────────────────

    @app.route("/api/applications/<app_id>/sanctions-check", methods=["POST"])
    @admin_required_inner
    def run_sanctions_check(app_id):
        """
        POST /api/applications/<app_id>/sanctions-check
        Pulls applicant data from MongoDB, runs full KYC screening,
        saves results back, and writes an audit log entry.
        Auto-selects FINTRAC mode when application country_code == "CA".
        """
        rec = applications_col.find_one({"application_id": app_id})
        if not rec:
            return jsonify({"error": "Application not found"}), 404

        full_name = rec.get("full_name") or rec.get("name") or ""
        birth_date = rec.get("date_of_birth") or rec.get("dob") or rec.get("birth_date")
        nationality = (
            rec.get("nationality") or rec.get("country") or rec.get("country_code")
        )
        passport_num = rec.get("passport_number") or rec.get("passportNumber")
        id_number = rec.get("id_number") or rec.get("idNumber")
        country_code = (rec.get("country_code") or "").strip().upper()

        if not full_name:
            return jsonify({"error": "No name found on application"}), 400

        try:
            result = screener.full_kyc_screen(
                name=full_name,
                birth_date=birth_date,
                nationality=nationality,
                passport_number=passport_num,
                id_number=id_number,
                country_code=country_code,
            )
        except Exception as e:
            logger.error("Sanctions check error for %s: %s", app_id, e)
            return jsonify({"error": str(e)}), 500

        update = {
            "sanctions_screening": {
                **result,
                "checked_by": session.get("email", "admin"),
                "checked_at": datetime.utcnow(),
            },
            "updated_at": datetime.utcnow(),
        }

        if result.get("block"):
            update["status"] = "rejected"
            update["rejection_reason"] = (
                f"Sanctions screening BLOCK: score={result.get('score', 0):.2f} "
                f"({'FINTRAC' if result.get('fintrac_mode') else 'global'}) "
                f"on {result.get('screened_at', '')}"
            )
            update["reviewed_by"] = "sanctions_auto"
            update["reviewed_at"] = datetime.utcnow()

        applications_col.update_one({"application_id": app_id}, {"$set": update})

        audit_fn(
            app_id,
            "sanctions_check_run",
            f"hit={result.get('hit')} block={result.get('block')} "
            f"score={result.get('score', 0):.3f} "
            f"fintrac={result.get('fintrac_mode', False)}",
            user=session.get("email", "admin"),
        )

        return jsonify(
            {
                "success": True,
                "application_id": app_id,
                **result,
            }
        )

    # ── Route 3: Get saved sanctions result ───────────────────────────────────

    @app.route("/api/applications/<app_id>/sanctions-result", methods=["GET"])
    @admin_required_inner
    def get_sanctions_result(app_id):
        """GET /api/applications/<app_id>/sanctions-result"""
        rec = applications_col.find_one(
            {"application_id": app_id},
            {"sanctions_screening": 1, "full_name": 1, "name": 1, "_id": 0},
        )
        if not rec:
            return jsonify({"error": "Not found"}), 404

        screening = rec.get("sanctions_screening")
        if not screening:
            return jsonify({"screened": False, "message": "No sanctions check run yet"})

        if isinstance(screening.get("checked_at"), datetime):
            screening["checked_at"] = screening["checked_at"].isoformat()

        return jsonify({"success": True, "screened": True, **screening})

    # ── Route 4: Bulk screen all pending applications (admin) ─────────────────

    @app.route("/api/sanctions/bulk-screen", methods=["POST"])
    @admin_required_inner
    def bulk_sanctions_screen():
        """
        POST /api/sanctions/bulk-screen
        Screens all unscreened applications. Auto-routes CA to FINTRAC.
        Returns a summary including fintrac_count.
        """
        query = {
            "sanctions_screening": {"$exists": False},
            "status": {"$in": ["processing", "reviewing", "pending_onfido"]},
        }
        pending = list(
            applications_col.find(
                query,
                {
                    "application_id": 1,
                    "full_name": 1,
                    "name": 1,
                    "date_of_birth": 1,
                    "dob": 1,
                    "nationality": 1,
                    "country_code": 1,
                },
            )
        )

        summary = {
            "total": len(pending),
            "hits": 0,
            "blocks": 0,
            "clean": 0,
            "errors": 0,
            "fintrac_count": 0,  # how many were screened via FINTRAC
        }

        for rec in pending:
            app_id = rec["application_id"]
            name = rec.get("full_name") or rec.get("name") or ""
            dob = rec.get("date_of_birth") or rec.get("dob")
            nat = rec.get("nationality") or ""
            country_code = (rec.get("country_code") or "").strip().upper()

            if not name:
                summary["errors"] += 1
                continue

            try:
                result = screener.full_kyc_screen(
                    name,
                    dob,
                    nat or None,
                    country_code=country_code or None,
                )
                if result.get("fintrac_mode"):
                    summary["fintrac_count"] += 1

                update = {
                    "sanctions_screening": {
                        **result,
                        "checked_by": "bulk_auto",
                        "checked_at": datetime.utcnow(),
                    },
                    "updated_at": datetime.utcnow(),
                }
                if result.get("block"):
                    update["status"] = "rejected"
                    update["rejection_reason"] = (
                        f"Sanctions screening BLOCK (bulk run, "
                        f"{'FINTRAC' if result.get('fintrac_mode') else 'global'})"
                    )
                    update["reviewed_by"] = "sanctions_auto"
                    update["reviewed_at"] = datetime.utcnow()
                    summary["blocks"] += 1
                elif result.get("hit"):
                    summary["hits"] += 1
                else:
                    summary["clean"] += 1

                applications_col.update_one(
                    {"application_id": app_id}, {"$set": update}
                )
                audit_fn(
                    app_id,
                    "sanctions_bulk_check",
                    f"hit={result.get('hit')} block={result.get('block')} "
                    f"fintrac={result.get('fintrac_mode', False)}",
                    user="bulk_auto",
                )
            except Exception as e:
                logger.error("Bulk sanctions error for %s: %s", app_id, e)
                summary["errors"] += 1

        return jsonify({"success": True, "summary": summary})
