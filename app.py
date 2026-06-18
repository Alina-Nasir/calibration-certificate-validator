# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from datetime import datetime
import io
import os
import re
import json
import difflib
from PIL import Image
import fitz
from google import genai
import tempfile
from typing import Optional, Dict, Any, Tuple, List
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Calibration Certificate Validator", page_icon="🔍", layout="wide"
)

st.markdown(
    """
<style>
.main-header {
    background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
    padding: 25px; border-radius: 15px; margin-bottom: 25px;
    color: white; text-align: center;
}
.success-box { background-color: #D1FAE5; padding: 15px; border-radius: 10px;
    border-left: 5px solid #10B981; margin-bottom: 15px; color: #065F46; font-weight: 500; }
.warning-box { background-color: #FEF3C7; padding: 15px; border-radius: 10px;
    border-left: 5px solid #F59E0B; margin-bottom: 15px; color: #92400E; font-weight: 500; }
.error-box { background-color: #FEE2E2; padding: 15px; border-radius: 10px;
    border-left: 5px solid #EF4444; margin-bottom: 15px; color: #991B1B; font-weight: 500; }
.info-box { background-color: #DBEAFE; padding: 15px; border-radius: 10px;
    border-left: 5px solid #3B82F6; margin-bottom: 15px; color: #1E40AF; font-weight: 500; }
</style>
""",
    unsafe_allow_html=True,
)

# Session state
if "master_df" not in st.session_state:
    st.session_state.master_df = None
if "validator" not in st.session_state:
    st.session_state.validator = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class CertificateValidator:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"
        self.master_df = None
        self.processed_certs = set()

    # ---------- Master loading ----------
    def load_master_excel(self, file) -> Tuple[bool, str]:
        try:
            df = pd.read_excel(file, sheet_name=0)
            # Collapse internal newlines/extra spaces in headers, then strip.
            df.columns = (
                df.columns.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
            )
            self.master_df = df
            return True, f"Loaded {len(self.master_df)} records"
        except Exception as e:
            return False, str(e)

    def get_certificate_column(self) -> Optional[str]:
        if self.master_df is None:
            return None
        for col in self.master_df.columns:
            if "calibration certificate no" in col.lower():
                return col
        for col in self.master_df.columns:
            if "certificate no" in col.lower():
                return col
        return None

    def get_serial_column(self) -> Optional[str]:
        """Real serial number lives in the 'Sr. No.' column, not 'File Number/Serial Number'."""
        if self.master_df is None:
            return None
        for col in self.master_df.columns:
            if "sr. no" in col.lower() or "sr no" in col.lower():
                return col
        for col in self.master_df.columns:
            if "serial" in col.lower():
                return col
        return None

    def col(self, *candidates) -> Optional[str]:
        """Find first existing column matching any candidate substring (case-insensitive)."""
        if self.master_df is None:
            return None
        for cand in candidates:
            for c in self.master_df.columns:
                if cand.lower() in c.lower():
                    return c
        return None

    # ---------- Normalization ----------
    def normalize_text(self, text) -> str:
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return ""
        text = str(text).strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[°%]", "", text)
        return text

    def _alnum(self, t) -> str:
        """Lowercase alphanumeric-only form (for serial/ID matching)."""
        return re.sub(r"[^a-z0-9]", "", self.normalize_text(t))

    def normalize_date(self, date_str) -> Optional[str]:
        if (
            date_str is None
            or (isinstance(date_str, float) and pd.isna(date_str))
            or date_str == ""
        ):
            return None
        try:
            s = str(date_str).strip()
            if " " in s and re.match(r"\d{4}-\d{2}-\d{2}", s):
                s = s.split(" ")[0]
            if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
                return s
            # OCR cleanups: O->0, trailing s after digit -> 5, drop punctuation
            s = re.sub(r"(?<=\d)[oO]", "0", s)
            s = re.sub(r"(?<=\d)[sS](?=[\s.,]|$)", "5", s)
            s = s.replace(".", "").replace(",", "")
            s = re.sub(r"\s+", " ", s).strip()
            formats = [
                "%d-%b-%y",
                "%d-%b-%Y",
                "%d/%m/%Y",
                "%d/%m/%y",
                "%d-%m-%Y",
                "%d %B %Y",
                "%B %d %Y",
                "%Y-%m-%d",
                "%b %d %Y",
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
                except ValueError:
                    continue
            return pd.to_datetime(s, dayfirst=True).strftime("%Y-%m-%d")
        except Exception:
            return None
                                                        
    # ---------- Comparators ----------
    def compare_range(self, extracted, master) -> bool:
        if not extracted or master is None or pd.isna(master):
            return False
        e = re.findall(r"\d+(?:\.\d+)?", str(extracted))
        m = re.findall(r"\d+(?:\.\d+)?", str(master))
        if e and m:
            return sorted(e) == sorted(m)
        e_clean = re.sub(r"[\s/]", "", str(extracted).lower())
        m_clean = re.sub(r"[\s/]", "", str(master).lower())
        return e_clean == m_clean

    def compare_least_count(self, extracted, master) -> bool:
        if not extracted or master is None or pd.isna(master):
            return False

        def units(t):
            t = str(t).lower()
            t = re.sub(r"amper|ampere|amp", "a", t)
            t = re.sub(r"volt", "v", t)
            return set(re.findall(r"[a-z]+", t))

        e_num = set(re.findall(r"\d+(?:\.\d+)?", str(extracted).lower()))
        m_num = set(re.findall(r"\d+(?:\.\d+)?", str(master).lower()))
        nums_ok = (
            bool(e_num)
            and bool(m_num)
            and (e_num.issubset(m_num) or m_num.issubset(e_num))
        )
        units_ok = bool(units(extracted) & units(master))
        return nums_ok and units_ok

    def compare_serial(self, extracted, master) -> bool:
        """Match serials ignoring case, spaces and dashes (TC-25 == TC 25 == tc25)."""
        if not extracted or master is None or pd.isna(master):
            return False
        return self._alnum(extracted) == self._alnum(master)

    def compare_contains(self, extracted, master) -> bool:
        """Match if one normalized value contains the other (RICI vs RICI Company Ltd.)."""
        if not extracted or master is None or pd.isna(master):
            return False
        e, m = self.normalize_text(extracted), self.normalize_text(master)
        if not e or not m:
            return False
        return e == m or e in m or m in e

    def compare_text_fuzzy(self, extracted, master) -> bool:
        """Match two text values fuzzyly, allowing for typos and containment."""
        if not extracted or master is None or pd.isna(master):
            return False
        e_norm = self.normalize_text(extracted)
        m_norm = self.normalize_text(master)
        if not e_norm or not m_norm:
            return False

        # 1. Exact match after normalization
        if e_norm == m_norm:
            return True

        # 2. Fuzzy similarity on the entire string
        ratio = difflib.SequenceMatcher(None, e_norm, m_norm).ratio()
        if ratio >= 0.8:
            return True

        # 3. Word-by-word fuzzy subset match
        e_words = e_norm.split()
        m_words = m_norm.split()
        if not e_words or not m_words:
            return False

        shorter_words = e_words if len(e_words) <= len(m_words) else m_words
        longer_words = m_words if len(e_words) <= len(m_words) else e_words

        all_matched = True
        for sw in shorter_words:
            word_matched = False
            for lw in longer_words:
                if sw == lw:
                    word_matched = True
                    break
                if len(sw) > 3 and len(lw) > 3:
                    if sw.startswith(lw) or lw.startswith(sw):
                        word_matched = True
                        break
                    if difflib.SequenceMatcher(None, sw, lw).ratio() >= 0.8:
                        word_matched = True
                        break
            if not word_matched:
                all_matched = False
                break

        return all_matched

    def compare_acceptance(self, extracted, master) -> bool:
        """Compare acceptance criteria by shared numbers + key unit tokens (RDG/FSD)."""
        if not extracted or master is None or pd.isna(master):
            return False
        e_num = set(re.findall(r"\d+(?:\.\d+)?", str(extracted)))
        m_num = set(re.findall(r"\d+(?:\.\d+)?", str(master)))
        e_tok = set(re.findall(r"rdg|fsd", str(extracted).lower()))
        m_tok = set(re.findall(r"rdg|fsd", str(master).lower()))
        nums_ok = bool(e_num & m_num)
        toks_ok = (e_tok == m_tok) if (e_tok or m_tok) else True
        return nums_ok and toks_ok

    # ---------- PDF / extraction ----------
    def pdf_to_images(self, pdf_file) -> List[Image.Image]:
        images = []
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        try:
            doc = fitz.open(tmp_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                mat = fitz.Matrix(2.5, 2.5)
                pix = page.get_pixmap(matrix=mat)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                images.append(img)
            doc.close()
        finally:
            os.unlink(tmp_path)
        return images

    def extract_certificate_number(self, image) -> Optional[str]:
        prompt = """
        Look at this calibration certificate page. Find the certificate number,
        usually labeled "Certificate No:" near the top. Correct obvious OCR errors
        (letter O -> digit 0). Return ONLY the number (4 to 6 digits), nothing else.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=[prompt, image]
            )
            result = response.text.strip()
            match = re.search(r"\b(\d{4,6})\b", result)
            return match.group(1) if match else None
        except Exception as e:
            st.warning(f"Certificate number extraction failed: {e}")
            return None

    def extract_certificate_details(self, image) -> Dict[str, Any]:
        prompt = """
        Analyze this calibration certificate page. Extract fields EXACTLY as printed.
        Rules:
        - Correct obvious OCR digit/letter confusion (O->0, "0s"->"05").
        - Dates: return ISO format YYYY-MM-DD.
        - Serial / identification / sl_no numbers: return only the alphanumeric token, no labels.
        - acceptance_criteria: look in the Results/Remarks section, e.g. "2.5% (RDG)" or "± 1.5 % FSD". Return the full criteria string.
        - measurements: from the Calibration Results table, return one object per row with:
          "reference" (the standard / reference / certified reading),
          "uuc" (the unit under calibration / instrument reading), and
          "error" (the printed reading error, if present).
        - If a field is not visible, return an empty string "".
        Return ONLY valid JSON with this exact structure:
        {
            "acceptance_criteria": "",
            "measurements": [
                {"reference": 0, "uuc": 0, "error": 0}
            ],
            "certificate_number": "",
            "customer_name": "",
            "customer_address": "",
            "instrument_description": "",
            "manufacturer": "",
            "calibration_agency": "",
            "model_no": "",
            "serial_no": "",
            "sl_no": "",
            "identification_no": "",
            "capacity_range": "",
            "resolution": "",
            "acceptance_criteria": "",
            "calibration_method": "",
            "calibration_site": "",
            "receipt_date": "",
            "calibration_date": "",
            "issue_date": "",
            "due_date": "",
            "calibrated_by": "",
            "reviewed_by": "",
            "approved_by": "",
            "temperature": "",
            "humidity": ""
        }
        """
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=[prompt, image]
            )
            json_match = re.search(r"\{.*\}", response.text.strip(), re.DOTALL)
            return json.loads(json_match.group()) if json_match else {}
        except Exception as e:
            st.warning(f"Detail extraction failed: {e}")
            return {}

    def process_pdf_pages(self, pdf_file) -> List[Dict]:
        """Detect certificates per page; merge multi-page detail so fields on the
        results page (e.g. Acceptance Criteria) are not lost."""
        images = self.pdf_to_images(pdf_file)
        if not images:
            return []

        page_certificates = []
        current = None  # currently open certificate being assembled

        for i, img in enumerate(images):
            cert_num = self.extract_certificate_number(img)
            details = self.extract_certificate_details(img)

            if cert_num and cert_num not in self.processed_certs:
                # New certificate starts here.
                self.processed_certs.add(cert_num)
                current = {
                    "page_num": i + 1,
                    "image": img,
                    "cert_number": cert_num,
                    "details": details or {},
                }
                page_certificates.append(current)
            elif current is not None and details:
                # Continuation page: fill only empty fields of the open certificate.
                for k, v in details.items():
                    if v and not current["details"].get(k):
                        current["details"][k] = v

        return page_certificates

    # ---------- Master lookup ----------
    def find_in_master(self, certificate_number: str) -> Optional[pd.Series]:
        if self.master_df is None:
            return None
        cert_col = self.get_certificate_column()
        if not cert_col:
            return None
        mask = (
            self.master_df[cert_col].astype(str).str.strip()
            == str(certificate_number).strip()
        )
        matches = self.master_df[mask]
        return matches.iloc[0] if len(matches) > 0 else None

    def get_expiry_status(self, due_date) -> Dict:
        if (
            due_date is None
            or (isinstance(due_date, float) and pd.isna(due_date))
            or due_date == ""
        ):
            return {
                "status": "No Date",
                "color": "gray",
                "days_left": None,
                "message": "No due date",
            }
        try:
            normalized = self.normalize_date(due_date)
            due = (
                datetime.strptime(normalized, "%Y-%m-%d").date()
                if normalized
                else pd.to_datetime(due_date, dayfirst=True).date()
            )
            today = datetime.now().date()
            days_left = (due - today).days
            if days_left < 0:
                return {
                    "status": "EXPIRED",
                    "color": "red",
                    "days_left": days_left,
                    "message": f"Expired {abs(days_left)} days ago",
                }
            if days_left <= 7:
                return {
                    "status": "CRITICAL",
                    "color": "darkred",
                    "days_left": days_left,
                    "message": f"Expires in {days_left} days",
                }
            if days_left <= 30:
                return {
                    "status": "EXPIRING SOON",
                    "color": "orange",
                    "days_left": days_left,
                    "message": f"Expires in {days_left} days",
                }
            return {
                "status": "VALID",
                "color": "green",
                "days_left": days_left,
                "message": f"Valid - {days_left} days left",
            }
        except Exception:
            return {
                "status": "Invalid Date",
                "color": "gray",
                "days_left": None,
                "message": "Invalid date",
            }

    # ---------- Comparison table ----------
    def create_comparison_table(
        self, cert_details: Dict, master_record: pd.Series
    ) -> pd.DataFrame:
        comparisons = []
        cert_col = self.get_certificate_column()
        serial_col = self.get_serial_column()

        # Certificate Number first
        master_cert_value = master_record.get(cert_col, "") if cert_col else ""
        extracted_cert_value = cert_details.get("certificate_number", "")
        comparisons.append(
            {
                "Field": "Certificate Number",
                "Extracted from Certificate": str(extracted_cert_value)
                if extracted_cert_value
                else "❌ Not found",
                "Master Database": str(master_cert_value)
                if pd.notna(master_cert_value)
                else "❌ Not in DB",
                "Status": "✅ Match"
                if str(extracted_cert_value).strip() == str(master_cert_value).strip()
                else "❌ Mismatch",
            }
        )

        # (master_column, display_name, cert_details_key)
        field_map = [
            (
                self.col("Instrument") or "Instrument",
                "Instrument",
                "instrument_description",
            ),
            (self.col("Make") or "Make", "Make", "manufacturer"),
            (self.col("Sl No", "S.No", "Sr No") or "Sl No", "Sl No", "sl_no"),
            (self.col("Range") or "Range", "Range", "capacity_range"),
            (self.col("Least count") or "Least count", "Least Count", "resolution"),
            (
                self.col("Unique Identity") or "Unique Identity No.",
                "Unique ID",
                "identification_no",
            ),
            (serial_col or "Sr. No.", "Serial Number", "serial_no"),
            (
                self.col("Cal Date") or "Cal Date",
                "Calibration Date",
                "calibration_date",
            ),
            (self.col("Due Date") or "Due Date", "Due Date", "due_date"),
            (self.col("User Location") or "User Location", "Location", None),
            (
                self.col("Acceptance Criteria") or "Acceptance Criteria",
                "Acceptance Criteria",
                "acceptance_criteria",
            ),
            (
                self.col("Calibration Agency") or "Calibration Agency",
                "Agency",
                "calibration_agency",
            ),
            (self.col("Remarks") or "Remarks", "Remarks", None),
        ]

        for db_field, display_name, cert_key in field_map:
            master_value = (
                master_record.get(db_field, "")
                if db_field in master_record.index
                else ""
            )
            master_display = (
                str(master_value)
                if pd.notna(master_value) and master_value != ""
                else "❌ Not in DB"
            )

            extracted_value = ""
            if cert_key:
                extracted_value = cert_details.get(cert_key, "") or ""
            extracted_display = (
                str(extracted_value) if extracted_value else "❌ Not in Certificate"
            )

            if display_name == "Range":
                status = (
                    "✅ Match"
                    if self.compare_range(extracted_value, master_value)
                    else "❌ Mismatch"
                )
            elif display_name == "Least Count":
                status = (
                    "✅ Match"
                    if self.compare_least_count(extracted_value, master_value)
                    else "❌ Mismatch"
                )
            elif display_name in ("Serial Number", "Sl No"):
                status = (
                    "✅ Match"
                    if self.compare_serial(extracted_value, master_value)
                    else "❌ Mismatch"
                )
            elif display_name == "Agency":
                status = (
                    "✅ Match"
                    if self.compare_text_fuzzy(extracted_value, master_value)
                    else "❌ Mismatch"
                )
            elif display_name == "Acceptance Criteria":
                status = (
                    "✅ Match"
                    if self.compare_acceptance(extracted_value, master_value)
                    else "❌ Mismatch"
                )
            elif display_name in ("Calibration Date", "Due Date"):
                e_date = self.normalize_date(extracted_value)
                m_date = self.normalize_date(master_value)
                status = (
                    ("✅ Match" if e_date == m_date else "❌ Mismatch")
                    if (e_date and m_date)
                    else "⚠️ Date issue"
                )
            else:
                has_master = pd.notna(master_value) and master_value != ""
                if extracted_value and has_master:
                    status = (
                        "✅ Match"
                        if self.compare_text_fuzzy(extracted_value, master_value)
                        else "❌ Mismatch"
                    )
                elif extracted_value and not has_master:
                    status = "⚠️ Missing in Master DB"
                elif not extracted_value and has_master:
                    status = "⚠️ Missing in Certificate"
                else:
                    status = "⚪ Both Missing"

            comparisons.append(
                {
                    "Field": display_name,
                    "Extracted from Certificate": extracted_display,
                    "Master Database": master_display,
                    "Status": status,
                }
            )

        # Additional certificate-only info
        if cert_details:
            comparisons.append(
                {
                    "Field": "--- ADDITIONAL CERTIFICATE INFO ---",
                    "Extracted from Certificate": "---",
                    "Master Database": "---",
                    "Status": "---",
                }
            )
            extra_fields = [
                ("customer_name", "Customer Name"),
                ("customer_address", "Customer Address"),
                ("model_no", "Model No"),
                ("calibration_method", "Calibration Method"),
                ("calibration_site", "Calibration Site"),
                ("receipt_date", "Receipt Date"),
                ("issue_date", "Issue Date"),
                ("calibrated_by", "Calibrated By"),
                ("reviewed_by", "Reviewed By"),
                ("approved_by", "Approved By"),
                ("temperature", "Temperature"),
                ("humidity", "Humidity"),
            ]
            for field, display_name in extra_fields:
                value = cert_details.get(field, "")
                if value:
                    comparisons.append(
                        {
                            "Field": display_name,
                            "Extracted from Certificate": str(value),
                            "Master Database": "Not in Master DB",
                            "Status": "ℹ️ Extra Info",
                        }
                    )

        return pd.DataFrame(comparisons)


class CalibrationValidationEngine:
    """Adds a calibration-aware validation layer on top of already-extracted data.
    Does NOT re-extract or modify any existing fields. Pure deterministic parsing."""

    # ---------- TASK 1: Acceptance criteria parser ----------
    @staticmethod
    def parse_acceptance_criteria(raw) -> Optional[List[Dict[str, Any]]]:
        if (
            raw is None
            or (isinstance(raw, float) and pd.isna(raw))
            or str(raw).strip() == ""
        ):
            return None

        raw_clean = str(raw).lower()
        raw_clean = raw_clean.replace("±", "±").replace("+-", "±").replace("+/-", "±")

        parts = []
        # Split on '±', ';', or commas that separate distinct rules
        split_parts = re.split(
            r"±|;|,\s*(?=\d|±|above|up to|below|under|over|\()", raw_clean
        )
        for p in split_parts:
            p_str = p.strip()
            if p_str:
                parts.append(p_str)

        rules = []

        for part in parts:
            # Find all numbers in this part
            num_matches = list(re.finditer(r"(\d+(?:\.\d+)?)", part))
            if not num_matches:
                continue

            # The first number is the tolerance value
            tol_val = float(num_matches[0].group(1))
            tol_span = num_matches[0].span()

            # Extract the text after the first number
            after_tol = part[tol_span[1] :].strip()

            # Find where the condition starts in after_tol
            cond_start_idx = len(after_tol)

            # Check for condition keywords
            cond_kw_match = re.search(
                r"\b(up to|above|below|under|over|<=|>=|<|>)\b|<=|>=|<|>", after_tol
            )
            if cond_kw_match:
                cond_start_idx = min(cond_start_idx, cond_kw_match.start())

            # Check for any number in after_tol
            num_in_after = re.search(r"\d", after_tol)
            if num_in_after:
                cond_start_idx = min(cond_start_idx, num_in_after.start())

            # Extract unit and condition text
            unit_text = after_tol[:cond_start_idx].strip()
            cond_phrase = after_tol[cond_start_idx:].strip()

            # Determine tolerance type and unit
            t = "ABSOLUTE"
            unit = "mm"
            has_percent = "%" in unit_text or "percent" in unit_text

            if (
                "fsd" in unit_text
                or "full scale" in unit_text
                or "fs" in re.findall(r"\bfs\b", unit_text)
            ):
                t, unit = "FSD", "percent"
            elif (
                "rdg" in unit_text
                or "reading" in unit_text
                or ("rd" in re.findall(r"\brd\b", unit_text) and has_percent)
            ):
                t, unit = "RDG", "percent"
            elif "digit" in unit_text or "dgt" in unit_text or "lsd" in unit_text:
                t, unit = "DIGIT", "digit"
            elif "mm" in unit_text:
                t, unit = "ABSOLUTE", "mm"
            elif has_percent:
                t, unit = "RDG", "percent"

            # Parse condition
            cond_type = None
            cond_val = None
            cond_val2 = None

            if cond_phrase:
                cond_nums = [
                    float(n) for n in re.findall(r"\d+(?:\.\d+)?", cond_phrase)
                ]
                if len(cond_nums) >= 2:
                    cond_type = "range"
                    cond_val = cond_nums[0]
                    cond_val2 = cond_nums[1]
                elif len(cond_nums) == 1:
                    if any(
                        kw in cond_phrase
                        for kw in ["up to", "<=", "below", "under", "<"]
                    ):
                        cond_type = "lte"
                        cond_val = cond_nums[0]
                    elif any(kw in cond_phrase for kw in ["above", ">=", "over", ">"]):
                        cond_type = "gt"
                        cond_val = cond_nums[0]
                    else:
                        cond_type = "lte"
                        cond_val = cond_nums[0]

            rules.append(
                {
                    "type": t,
                    "value": tol_val,
                    "unit": unit,
                    "cond_type": cond_type,
                    "cond_val": cond_val,
                    "cond_val2": cond_val2,
                }
            )

        return rules if rules else None

    # ---------- helpers ----------
    @staticmethod
    def _to_float(x) -> Optional[float]:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        m = re.search(r"-?\d+(?:\.\d+)?", str(x))
        return float(m.group()) if m else None

    @staticmethod
    def _max_range(range_str) -> Optional[float]:
        """Extract the largest numeric value from a range like '150-1300A/16-46V' -> 1300."""
        nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(range_str or ""))]
        return max(nums) if nums else None

    # ---------- TASK 2 + 3: per-row validation engine ----------
    @classmethod
    def validate_measurements(
        cls,
        parsed_ac: Optional[Any],
        measurements: List[Dict[str, Any]],
        max_range: Optional[float] = None,
        least_count: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """measurements: list of rows, each with at least 'reference' and 'error'.
        Returns the validation result list (TASK output schema)."""
        results = []
        if not parsed_ac or not measurements:
            return results

        if isinstance(parsed_ac, dict):
            rules = [parsed_ac]
        elif isinstance(parsed_ac, list):
            rules = parsed_ac
        else:
            return results

        for row in measurements:
            reference = cls._to_float(row.get("reference"))
            uuc = cls._to_float(row.get("uuc"))

            if uuc is not None and reference is not None:
                error = uuc - reference
            else:
                error = cls._to_float(row.get("error"))

            if error is None:
                continue

            applicable_rule = None
            if reference is not None:
                for rule in rules:
                    cond_type = rule.get("cond_type")
                    cond_val = rule.get("cond_val")
                    cond_val2 = rule.get("cond_val2")

                    if cond_type == "lte" and reference <= cond_val:
                        applicable_rule = rule
                        break
                    elif cond_type == "gt" and reference > cond_val:
                        applicable_rule = rule
                        break
                    elif cond_type == "range" and cond_val <= reference <= cond_val2:
                        applicable_rule = rule
                        break

            if applicable_rule is None:
                for rule in rules:
                    if rule.get("cond_type") is None:
                        applicable_rule = rule
                        break

            if applicable_rule is None and rules:
                applicable_rule = rules[0]

            if applicable_rule is None:
                results.append(
                    {
                        "reference": reference,
                        "uuc": uuc if uuc is not None else "N/A",
                        "error": round(error, 4),
                        "allowed_error": None,
                        "status": "UNKNOWN",
                    }
                )
                continue

            t = applicable_rule["type"]
            v = applicable_rule["value"]

            if t == "RDG" and reference is not None:
                allowed = (v / 100.0) * abs(reference)
            elif t == "FSD" and max_range is not None:
                allowed = (v / 100.0) * abs(max_range)
            elif t == "ABSOLUTE":
                allowed = v
            elif t == "DIGIT" and least_count is not None:
                allowed = v * least_count
            else:
                results.append(
                    {
                        "reference": reference,
                        "uuc": uuc if uuc is not None else "N/A",
                        "error": round(error, 4),
                        "allowed_error": None,
                        "status": "UNKNOWN",
                    }
                )
                continue

            status = "PASS" if abs(error) <= allowed else "FAIL"
            results.append(
                {
                    "reference": reference,
                    "uuc": uuc if uuc is not None else "N/A",
                    "error": round(error, 4),
                    "allowed_error": round(allowed, 4),
                    "status": status,
                }
            )
        return results

    # ---------- TASK 4: final decision ----------
    @staticmethod
    def final_decision(measurement_validation: List[Dict[str, Any]]) -> str:
        if not measurement_validation:
            return "UNKNOWN"
        statuses = [r["status"] for r in measurement_validation]
        if any(s == "FAIL" for s in statuses):
            return "FAIL"
        if all(s == "PASS" for s in statuses):
            return "PASS"
        return "UNKNOWN"

    # ---------- convenience wrapper ----------
    @classmethod
    def run(cls, cert_details: Dict[str, Any]) -> Dict[str, Any]:
        """Build the full validation block from extracted cert_details.
        Expects cert_details to optionally contain a 'measurements' list with
        rows of {'reference': ..., 'error': ...}."""
        parsed_ac = cls.parse_acceptance_criteria(
            cert_details.get("acceptance_criteria")
        )
        max_range = cls._max_range(cert_details.get("capacity_range"))
        least_count = cls._to_float(cert_details.get("resolution"))
        measurements = cert_details.get("measurements", []) or []

        mv = cls.validate_measurements(parsed_ac, measurements, max_range, least_count)
        decision = cls.final_decision(mv)
        return {
            "parsed_acceptance_criteria": parsed_ac,
            "measurement_validation": mv,
            "final_decision": decision,
        }


def process_single_pdf(validator: CertificateValidator, pdf_file) -> List[Dict]:
    try:
        certificates = validator.process_pdf_pages(pdf_file)
        results = []
        due_col = validator.col("Due Date") or "Due Date"
        instr_col = validator.col("Instrument") or "Instrument"
        make_col = validator.col("Make") or "Make"
        serial_col = validator.get_serial_column() or "Sr. No."
        cal_col = validator.col("Cal Date") or "Cal Date"
        loc_col = validator.col("User Location") or "User Location"
        agency_col = validator.col("Calibration Agency") or "Calibration Agency"
        remarks_col = validator.col("Remarks") or "Remarks"

        for cert in certificates:
            cert_number = cert["cert_number"]
            cert_details = cert.get("details", {})
            master_record = validator.find_in_master(cert_number)
            if master_record is not None:
                due_date = master_record.get(due_col)
                expiry = validator.get_expiry_status(due_date)
                comparison_df = validator.create_comparison_table(
                    cert_details, master_record
                )

                # --- NEW: calibration-aware validation layer ---
                validation_block = CalibrationValidationEngine.run(cert_details)
                # -----------------------------------------------

                results.append(
                    {
                        "parsed_acceptance_criteria": validation_block[
                            "parsed_acceptance_criteria"
                        ],
                        "measurement_validation": validation_block[
                            "measurement_validation"
                        ],
                        "final_decision": validation_block["final_decision"],
                        "filename": pdf_file.name,
                        "certificate_number": cert_number,
                        "page_num": cert["page_num"],
                        "found_in_master": True,
                        "comparison_df": comparison_df,
                        "cert_details": cert_details,
                        "instrument": master_record.get(instr_col, "N/A"),
                        "make": master_record.get(make_col, "N/A"),
                        "serial_no": master_record.get(serial_col, "N/A"),
                        "cal_date": master_record.get(cal_col, "N/A"),
                        "due_date": due_date,
                        "expiry_status": expiry["status"],
                        "expiry_message": expiry["message"],
                        "days_left": expiry["days_left"],
                        "location": master_record.get(loc_col, "N/A"),
                        "agency": master_record.get(agency_col, "N/A"),
                        "remarks": master_record.get(remarks_col, "N/A"),
                        "error": None,
                    }
                )
            else:
                results.append(
                    {
                        "filename": pdf_file.name,
                        "certificate_number": cert_number,
                        "page_num": cert["page_num"],
                        "found_in_master": False,
                        "error": f"Certificate {cert_number} not found in master database",
                    }
                )
        return results
    except Exception as e:
        return [
            {
                "filename": pdf_file.name,
                "certificate_number": None,
                "found_in_master": False,
                "error": str(e),
            }
        ]


def _autosize(worksheet):
    for column in worksheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except Exception:
                pass
        worksheet.column_dimensions[column_letter].width = min(max_length + 2, 50)


def generate_batch_excel_report(results: List[Dict]) -> bytes:
    report_data = [
        {
            "Filename": r.get("filename", ""),
            "Certificate Number": r.get("certificate_number", ""),
            "Page Number": r.get("page_num", ""),
            "Status": "FOUND" if r.get("found_in_master") else "NOT FOUND",
            "Expiry Status": r.get("expiry_status", ""),
            "Instrument": r.get("instrument", ""),
            "Make": r.get("make", ""),
            "Serial No": r.get("serial_no", ""),
            "Calibration Date": r.get("cal_date", ""),
            "Due Date": r.get("due_date", ""),
            "Days Left": r.get("days_left", ""),
            "Location": r.get("location", ""),
            "Agency": r.get("agency", ""),
            "Remarks": r.get("remarks", ""),
            "Error": r.get("error", ""),
        }
        for r in results
    ]
    df = pd.DataFrame(report_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Batch Results", index=False)
        _autosize(writer.sheets["Batch Results"])
    output.seek(0)
    return output.getvalue()


def generate_expiry_excel_report(
    validator: CertificateValidator, report_type: str
) -> bytes:
    master_df = validator.master_df
    today = datetime.now().date()
    report_data = []
    cert_col = validator.get_certificate_column()
    instr_col = validator.col("Instrument") or "Instrument"
    make_col = validator.col("Make") or "Make"
    serial_col = validator.get_serial_column() or "Sr. No."
    due_col = validator.col("Due Date") or "Due Date"
    loc_col = validator.col("User Location") or "User Location"

    for _, row in master_df.iterrows():
        due_date = row.get(due_col)
        if pd.notna(due_date):
            try:
                due = pd.to_datetime(due_date, dayfirst=True).date()
                days_left = (due - today).days
                if report_type == "expired" and days_left < 0:
                    report_data.append(
                        {
                            "Certificate No": str(row.get(cert_col, ""))
                            if cert_col and pd.notna(row.get(cert_col))
                            else "",
                            "Instrument": str(row.get(instr_col, ""))
                            if pd.notna(row.get(instr_col))
                            else "",
                            "Make": str(row.get(make_col, ""))
                            if pd.notna(row.get(make_col))
                            else "",
                            "Serial No": str(row.get(serial_col, ""))
                            if pd.notna(row.get(serial_col))
                            else "",
                            "Due Date": due.strftime("%Y-%m-%d"),
                            "Days Overdue": abs(days_left),
                            "Location": str(row.get(loc_col, ""))
                            if pd.notna(row.get(loc_col))
                            else "",
                        }
                    )
                elif report_type == "expiring" and 0 <= days_left <= 60:
                    status = (
                        "Critical"
                        if days_left <= 7
                        else "Warning"
                        if days_left <= 30
                        else "Info"
                    )
                    report_data.append(
                        {
                            "Certificate No": str(row.get(cert_col, ""))
                            if cert_col and pd.notna(row.get(cert_col))
                            else "",
                            "Instrument": str(row.get(instr_col, ""))
                            if pd.notna(row.get(instr_col))
                            else "",
                            "Make": str(row.get(make_col, ""))
                            if pd.notna(row.get(make_col))
                            else "",
                            "Serial No": str(row.get(serial_col, ""))
                            if pd.notna(row.get(serial_col))
                            else "",
                            "Due Date": due.strftime("%Y-%m-%d"),
                            "Days Left": days_left,
                            "Status": status,
                            "Location": str(row.get(loc_col, ""))
                            if pd.notna(row.get(loc_col))
                            else "",
                        }
                    )
            except Exception:
                pass
    df = pd.DataFrame(report_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=report_type, index=False)
        _autosize(writer.sheets[report_type])
    output.seek(0)
    return output.getvalue()


def main():
    st.markdown(
        """
    <div class="main-header">
        <h1>🔍 Calibration Certificate Validator</h1>
        <p>Upload PDFs → Auto-detects certificates → Full comparison with master database → Tracks expiry</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("OfficeFlowAI")
        if os.path.exists("OfficeFlow Ai-01-01.png"):
            st.image("OfficeFlow Ai-01-01.png", width=120)
        st.divider()
        st.header("📊 Master Database")
        master_file = st.file_uploader(
            "Upload Trail master list.xlsx", type=["xlsx", "xls"]
        )

        if master_file:
            if not GEMINI_API_KEY:
                st.error("GEMINI_API_KEY not set in environment.")
                return
            validator = CertificateValidator(GEMINI_API_KEY)
            success, msg = validator.load_master_excel(master_file)
            if success:
                st.session_state.master_df = validator.master_df
                st.session_state.validator = validator
                st.success(msg)
                cert_col = validator.get_certificate_column()
                if cert_col:
                    st.metric("Total Records", len(st.session_state.master_df))
                    st.metric(
                        "Unique Certificates",
                        st.session_state.master_df[cert_col].nunique(),
                    )
                else:
                    st.warning(
                        "Could not find a certificate-number column in the master file."
                    )
            else:
                st.error(msg)

    if st.session_state.master_df is None or st.session_state.validator is None:
        st.info("👈 Please upload the master Excel file first")
        return

    validator = st.session_state.validator
    validator.processed_certs = set()

    # ==================== EXPIRY REPORTS ====================
    st.header("📊 Certificate Expiry Reports")
    today = datetime.now().date()
    cert_col = validator.get_certificate_column()
    due_col = validator.col("Due Date") or "Due Date"
    instr_col = validator.col("Instrument") or "Instrument"
    make_col = validator.col("Make") or "Make"
    serial_col = validator.get_serial_column() or "Sr. No."
    loc_col = validator.col("User Location") or "User Location"

    expired_list, critical_list, warning_list, info_list = [], [], [], []

    if due_col in validator.master_df.columns:
        for _, row in validator.master_df.iterrows():
            due = row.get(due_col)
            if pd.notna(due):
                try:
                    due_date = pd.to_datetime(due, dayfirst=True).date()
                    days_left = (due_date - today).days
                    record = {
                        "Certificate No": str(row.get(cert_col, ""))
                        if cert_col and pd.notna(row.get(cert_col))
                        else "",
                        "Instrument": str(row.get(instr_col, ""))
                        if pd.notna(row.get(instr_col))
                        else "",
                        "Make": str(row.get(make_col, ""))
                        if pd.notna(row.get(make_col))
                        else "",
                        "Serial No": str(row.get(serial_col, ""))
                        if pd.notna(row.get(serial_col))
                        else "",
                        "Due Date": due_date.strftime("%Y-%m-%d"),
                        "Days Left": days_left,
                        "Location": str(row.get(loc_col, ""))
                        if pd.notna(row.get(loc_col))
                        else "",
                    }
                    if days_left < 0:
                        record["Status"] = "🔴 EXPIRED"
                        expired_list.append(record)
                    elif days_left <= 7:
                        record["Status"] = "🔴 CRITICAL"
                        critical_list.append(record)
                    elif days_left <= 30:
                        record["Status"] = "🟠 WARNING"
                        warning_list.append(record)
                    elif days_left <= 60:
                        record["Status"] = "🟡 INFO"
                        info_list.append(record)
                except Exception:
                    pass

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Expired", len(expired_list))
    c2.metric("🔴 Critical (1-7 days)", len(critical_list))
    c3.metric("🟠 Warning (8-30 days)", len(warning_list))
    c4.metric("🟡 Info (31-60 days)", len(info_list))

    d1, d2 = st.columns(2)
    with d1:
        if expired_list:
            st.download_button(
                label=f"📥 Download Expired Certificates ({len(expired_list)})",
                data=generate_expiry_excel_report(validator, "expired"),
                file_name=f"expired_certificates_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )
    with d2:
        if critical_list or warning_list or info_list:
            st.download_button(
                label=f"📥 Download Expiring Soon ({len(critical_list) + len(warning_list) + len(info_list)})",
                data=generate_expiry_excel_report(validator, "expiring"),
                file_name=f"expiring_certificates_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )

    st.divider()

    for title, lst, empty_msg in [
        ("🔴 Expired Certificates", expired_list, "✅ No expired certificates found"),
        (
            "🔴 Critical - Expiring in 1-7 Days",
            critical_list,
            "✅ No certificates expiring in 1-7 days",
        ),
        (
            "🟠 Warning - Expiring in 8-30 Days",
            warning_list,
            "✅ No certificates expiring in 8-30 days",
        ),
        (
            "🟡 Info - Expiring in 31-60 Days",
            info_list,
            "✅ No certificates expiring in 31-60 days",
        ),
    ]:
        if lst:
            st.subheader(title)
            df = pd.DataFrame(lst).sort_values("Days Left")
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.info(empty_msg)

    st.divider()

    # ==================== VALIDATION ====================
    st.header("📄 Certificate Validation")
    upload_mode = st.radio(
        "Select mode:",
        ["Single PDF File", "Multiple PDF Files (Bulk Upload)"],
        horizontal=True,
    )

    if upload_mode == "Single PDF File":
        cert_file = st.file_uploader("Choose PDF file", type=["pdf"], key="single_cert")
        if cert_file and st.button(
            "🚀 Extract & Compare", type="primary", width="stretch"
        ):
            validator.processed_certs = set()
            with st.spinner("Processing PDF - detecting certificates page by page..."):
                results = process_single_pdf(validator, cert_file)
            if results:
                st.success(f"✅ Found {len(results)} certificate(s) in this PDF")
                for idx, result in enumerate(results):
                    st.markdown("---")
                    st.markdown(
                        f"### Certificate {idx + 1}: {result.get('certificate_number', 'Unknown')}"
                    )
                    st.info(f"📄 Found on Page: {result.get('page_num', 'Unknown')}")
                    if result.get("found_in_master"):
                        box = {
                            "EXPIRED": "error-box",
                            "CRITICAL": "error-box",
                            "EXPIRING SOON": "warning-box",
                            "VALID": "success-box",
                        }.get(result.get("expiry_status", ""), "info-box")
                        st.markdown(
                            f'<div class="{box}"><strong>📅 {result.get("expiry_message", "")}</strong></div>',
                            unsafe_allow_html=True,
                        )
                        st.subheader(
                            "📊 Full Comparison: Certificate vs Master Database"
                        )
                        if result.get("comparison_df") is not None:
                            st.dataframe(
                                result["comparison_df"],
                                width="stretch",
                                hide_index=True,
                            )
                        decision = result.get("final_decision")
                        if decision:
                            box = (
                                "success-box"
                                if decision == "PASS"
                                else "error-box"
                                if decision == "FAIL"
                                else "info-box"
                            )
                            st.markdown(
                                f'<div class="{box}"><strong>🧪 Final Decision: {decision}</strong></div>',
                                unsafe_allow_html=True,
                            )
                        mv = result.get("measurement_validation")
                        if mv:
                            st.subheader("🧪 Measurement Validation")
                            st.dataframe(
                                pd.DataFrame(mv), width="stretch", hide_index=True
                            )

                    else:
                        st.error(
                            f"❌ {result.get('error', 'Certificate not found in master database')}"
                        )
                st.balloons()
    else:
        cert_files = st.file_uploader(
            "Choose multiple PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="bulk_certs",
            help="Each PDF may contain multiple certificates",
        )
        if cert_files:
            st.info(f"📁 {len(cert_files)} file(s) selected")
            if st.button("🚀 Process All Files", type="primary", width="stretch"):
                all_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i, cert_file in enumerate(cert_files):
                    status_text.text(f"Processing {cert_file.name}...")
                    # Reset dedup per file so the same cert in different files is still reported
                    validator.processed_certs = set()
                    all_results.extend(process_single_pdf(validator, cert_file))
                    progress_bar.progress((i + 1) / len(cert_files))
                status_text.text("✅ Processing complete!")

                st.subheader("📊 Results Summary")
                results_data = []
                for r in all_results:
                    days_left = r.get("days_left")
                    if days_left is not None:
                        if days_left <= 7:
                            dot = "🔴"
                        elif days_left <= 30:
                            dot = "🟠"
                        elif days_left <= 60:
                            dot = "🟡"
                        else:
                            dot = "🟢"
                        days_display = f"{dot} {days_left} days"
                    else:
                        days_display = "N/A"
                    results_data.append(
                        {
                            "File": r["filename"],
                            "Certificate No": r.get("certificate_number", "N/A"),
                            "Page": r.get("page_num", "N/A"),
                            "Status": "✅ Found"
                            if r.get("found_in_master")
                            else "❌ Not Found",
                            "Expiry Status": r.get("expiry_status", "N/A"),
                            "Days Remaining": days_display,
                            "Instrument": r.get("instrument", ""),
                            "Due Date": r.get("due_date", ""),
                        }
                    )
                st.dataframe(
                    pd.DataFrame(results_data), width="stretch", hide_index=True
                )

                st.download_button(
                    label="📥 Download Complete Report (Excel)",
                    data=generate_batch_excel_report(all_results),
                    file_name=f"certificate_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch",
                )

                found_results = [r for r in all_results if r.get("found_in_master")]
                if found_results:
                    st.subheader("📋 Detailed Comparison Results")
                    for r in found_results:
                        with st.expander(
                            f"📄 Certificate {r['certificate_number']} - {r['instrument']} (Page {r.get('page_num', 'N/A')})"
                        ):
                            box = {
                                "EXPIRED": "error-box",
                                "CRITICAL": "error-box",
                                "EXPIRING SOON": "warning-box",
                                "VALID": "success-box",
                            }.get(r.get("expiry_status", ""), "info-box")
                            st.markdown(
                                f'<div class="{box}"><strong>📅 {r.get("expiry_message", "")}</strong></div>',
                                unsafe_allow_html=True,
                            )
                            st.subheader(
                                "📊 Full Comparison: Certificate vs Master Database"
                            )
                            if r.get("comparison_df") is not None:
                                st.dataframe(
                                    r["comparison_df"], width="stretch", hide_index=True
                                )
                            decision = r.get("final_decision")
                            if decision:
                                box_dec = (
                                    "success-box"
                                    if decision == "PASS"
                                    else "error-box"
                                    if decision == "FAIL"
                                    else "info-box"
                                )
                                st.markdown(
                                    f'<div class="{box_dec}"><strong>🧪 Final Decision: {decision}</strong></div>',
                                    unsafe_allow_html=True,
                                )
                            mv = r.get("measurement_validation")
                            if mv:
                                st.subheader("🧪 Measurement Validation")
                                st.dataframe(
                                    pd.DataFrame(mv), width="stretch", hide_index=True
                                )


if __name__ == "__main__":
    main()
