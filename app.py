import streamlit as st
import pandas as pd
from datetime import datetime
import io
import os
import re
import json
from PIL import Image
import fitz
from google import genai
import tempfile
from typing import Optional, Dict, Any, Tuple, List
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Calibration Certificate Validator",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 25px;
    color: white;
    text-align: center;
}
.success-box {
    background-color: #D1FAE5;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #10B981;
    margin-bottom: 15px;
    color: #065F46;
    font-weight: 500;
}
.warning-box {
    background-color: #FEF3C7;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #F59E0B;
    margin-bottom: 15px;
    color: #92400E;
    font-weight: 500;
}
.error-box {
    background-color: #FEE2E2;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #EF4444;
    margin-bottom: 15px;
    color: #991B1B;
    font-weight: 500;
}
.info-box {
    background-color: #DBEAFE;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #3B82F6;
    margin-bottom: 15px;
    color: #1E40AF;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# Session state
if 'master_df' not in st.session_state:
    st.session_state.master_df = None
if 'processed_certificates' not in st.session_state:
    st.session_state.processed_certificates = set()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class CertificateValidator:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"  
        self.master_df = None
        self.processed_certs = set()
    
    def load_master_excel(self, file) -> Tuple[bool, str]:
        try:
            self.master_df = pd.read_excel(file, sheet_name=0)
            self.master_df.columns = self.master_df.columns.str.strip()
            return True, f"Loaded {len(self.master_df)} records"
        except Exception as e:
            return False, str(e)
    
    def get_certificate_column(self) -> Optional[str]:
        for col in self.master_df.columns:
            if 'certificate' in col.lower():
                return col
        return None
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison (remove extra spaces, special chars, convert to lowercase)"""
        if not text or pd.isna(text):
            return ""
        text = str(text).strip().lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[°%]', '', text)
        return text
    
    def normalize_date(self, date_str) -> Optional[str]:
        """Normalize date to YYYY-MM-DD format for comparison"""
        if pd.isna(date_str) or not date_str:
            return None
        try:
            date_str = str(date_str).strip()
            
            # Remove time part if present (e.g., "2026-03-02 00:00:00" -> "2026-03-02")
            if ' ' in date_str and '-' in date_str:
                date_str = date_str.split(' ')[0]
            
            # Already in YYYY-MM-DD
            if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                return date_str
            
            # Try different formats
            formats = [
                '%d-%b-%y',      # 01-Mar-26
                '%d-%b-%Y',      # 01-Mar-2026
                '%d/%m/%Y',      # 01/03/2026
                '%d/%m/%y',      # 01/03/26
                '%d-%m-%Y',      # 01-03-2026
                '%d %B %Y',      # 01 March 2026
                '%B %d, %Y',     # March 01, 2026
                '%B %d %Y',      # March 01 2026
                '%Y-%m-%d',      # 2026-03-01
            ]
            
            for fmt in formats:
                try:
                    parsed = datetime.strptime(date_str, fmt)
                    return parsed.strftime('%Y-%m-%d')
                except:
                    continue
            
            # Last resort
            parsed = pd.to_datetime(date_str)
            return parsed.strftime('%Y-%m-%d')
        except:
            return None
    
    def compare_range(self, extracted: str, master: str) -> bool:
        """Compare range values intelligently"""
        if not extracted or not master or pd.isna(master):
            return False
        
        extracted = str(extracted).lower().strip()
        master = str(master).lower().strip()
        
        # Extract all numbers from both strings
        extracted_numbers = re.findall(r'\d+', extracted)
        master_numbers = re.findall(r'\d+', master)
        
        # If both have the same set of numbers, consider it a match
        if extracted_numbers and master_numbers:
            if set(extracted_numbers) == set(master_numbers):
                return True
        
        # Remove spaces, slashes, and common separators
        extracted_clean = re.sub(r'[\s/]', '', extracted)
        master_clean = re.sub(r'[\s/]', '', master)
        
        return extracted_clean == master_clean
    
    def compare_least_count(self, extracted: str, master: str) -> bool:
        """Compare least count/resolution values intelligently"""
        if not extracted or not master or pd.isna(master):
            return False
        
        extracted = str(extracted).lower().strip()
        master = str(master).lower().strip()
        
        # Extract numbers
        extracted_numbers = re.findall(r'(\d+(?:\.\d+)?)', extracted)
        master_numbers = re.findall(r'(\d+(?:\.\d+)?)', master)
        
        # Extract units
        extracted_units = re.findall(r'[a-z]+', extracted)
        master_units = re.findall(r'[a-z]+', master)
        
        # Normalize units: amper/ampere -> a, volt -> v
        def normalize_units(units):
            normalized = []
            for u in units:
                u = re.sub(r'amper|ampere', 'a', u)
                u = re.sub(r'volt', 'v', u)
                normalized.append(u)
            return normalized
        
        extracted_units_norm = normalize_units(extracted_units)
        master_units_norm = normalize_units(master_units)
        
        # Check if numbers match (ignoring order)
        numbers_match = False
        if extracted_numbers and master_numbers:
            if sorted(extracted_numbers) == sorted(master_numbers):
                numbers_match = True
        
        # Check if units match (ignoring order)
        units_match = False
        if extracted_units_norm and master_units_norm:
            if sorted(extracted_units_norm) == sorted(master_units_norm):
                units_match = True
        
        # Check if the values are swapped (e.g., "20A/1V" vs "1V/20A")
        extracted_pairs = list(zip(extracted_numbers, extracted_units_norm)) if len(extracted_numbers) == len(extracted_units_norm) else []
        master_pairs = list(zip(master_numbers, master_units_norm)) if len(master_numbers) == len(master_units_norm) else []
        
        pairs_match = False
        if extracted_pairs and master_pairs:
            if sorted(extracted_pairs) == sorted(master_pairs):
                pairs_match = True
        
        return numbers_match and (units_match or pairs_match)
    
    def pdf_to_images(self, pdf_file) -> list:
        images = []
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            doc = fitz.open(tmp_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                zoom = 2.5
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                images.append(img)
            doc.close()
        finally:
            os.unlink(tmp_path)
        return images
    
    def extract_certificate_number(self, image) -> Optional[str]:
        prompt = """
        Look at this calibration certificate page. Find the 5-digit certificate number.
        It is usually near the top or labeled as "Certificate No:".
        Return ONLY the 5-digit number, nothing else.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt, image]
            )
            result = response.text.strip()
            match = re.search(r'\b(\d{5})\b', result)
            return match.group(1) if match else None
        except:
            return None
    
    def extract_certificate_details(self, image) -> Dict[str, Any]:
        prompt = """
        You are analyzing a calibration certificate page. Extract ALL visible information.
        
        Return a JSON object with this exact structure:
        {
            "certificate_number": "the 5-digit certificate number",
            "customer_name": "customer name",
            "customer_address": "customer address",
            "instrument_description": "equipment description",
            "manufacturer": "manufacturer name",
            "model_no": "model number",
            "serial_no": "serial number",
            "identification_no": "unique identification number",
            "capacity_range": "capacity/range",
            "resolution": "resolution",
            "calibration_site": "on-site or off-site",
            "receipt_date": "receipt date",
            "calibration_date": "calibration date",
            "issue_date": "issue date",
            "due_date": "due date",
            "calibrated_by": "calibrated by",
            "reviewed_by": "reviewed by",
            "approved_by": "approved by",
            "temperature": "temperature during test",
            "humidity": "humidity during test"
        }
        
        Return ONLY the JSON, no other text.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt, image]
            )
            result_text = response.text.strip()
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {}
        except Exception as e:
            return {}
    
    def process_pdf_pages(self, pdf_file) -> List[Dict]:
        images = self.pdf_to_images(pdf_file)
        if not images:
            return []
        
        page_certificates = []
        
        for i, img in enumerate(images):
            cert_num = self.extract_certificate_number(img)
            if cert_num and cert_num not in self.processed_certs:
                self.processed_certs.add(cert_num)
                cert_details = self.extract_certificate_details(img)
                page_certificates.append({
                    'page_num': i + 1,
                    'image': img,
                    'cert_number': cert_num,
                    'details': cert_details
                })
        
        return page_certificates
    
    def find_in_master(self, certificate_number: str) -> Optional[pd.Series]:
        if self.master_df is None:
            return None
        cert_col = self.get_certificate_column()
        if cert_col:
            mask = self.master_df[cert_col].astype(str).str.strip() == str(certificate_number).strip()
            matches = self.master_df[mask]
            if len(matches) > 0:
                return matches.iloc[0]
        return None
    
    def get_expiry_status(self, due_date) -> Dict:
        if pd.isna(due_date) or due_date == '':
            return {'status': 'No Date', 'color': 'gray', 'days_left': None, 'message': 'No due date'}
        
        try:
            normalized = self.normalize_date(due_date)
            if normalized:
                due = datetime.strptime(normalized, '%Y-%m-%d').date()
            else:
                due = pd.to_datetime(due_date).date()
            
            today = datetime.now().date()
            days_left = (due - today).days
            
            if days_left < 0:
                return {'status': 'EXPIRED', 'color': 'red', 'days_left': days_left, 'message': f'Expired {abs(days_left)} days ago'}
            elif days_left <= 7:
                return {'status': 'CRITICAL', 'color': 'darkred', 'days_left': days_left, 'message': f'Expires in {days_left} days'}
            elif days_left <= 30:
                return {'status': 'EXPIRING SOON', 'color': 'orange', 'days_left': days_left, 'message': f'Expires in {days_left} days'}
            else:
                return {'status': 'VALID', 'color': 'green', 'days_left': days_left, 'message': f'Valid - {days_left} days left'}
        except Exception as e:
            return {'status': 'Invalid Date', 'color': 'gray', 'days_left': None, 'message': f'Invalid date: {due_date}'}
    
    def create_comparison_table(self, cert_details: Dict, master_record: pd.Series) -> pd.DataFrame:
        """Create a full comparison table between certificate and master record"""
        
        comparisons = []
        
        master_fields = [
            ('Sl No', 'Sl No'),
            ('Instrument', 'Instrument'),
            ('Make', 'Make'),
            ('Range', 'Range'),
            ('Least count', 'Least Count'),
            ('Unique Identity No.', 'Unique ID'),
            ('Instrument / Equipment / Software (Sr. No.)', 'Serial Number'),
            ('Calibration Certificate No.', 'Certificate Number'),
            ('Cal Date', 'Calibration Date'),
            ('Due Date', 'Due Date'),
            ('User Location', 'Location'),
            ('Acceptance Criteria', 'Acceptance Criteria'),
            ('Calibration Agency', 'Agency'),
            ('File Number/Serial Number', 'File Number'),
            ('Remarks', 'Remarks')
        ]
        
        for db_field, display_name in master_fields:
            master_value = master_record.get(db_field, '')
            master_display = str(master_value) if pd.notna(master_value) else '❌ Not in DB'
            
            extracted_value = ''
            if db_field == 'Instrument' and cert_details.get('instrument_description'):
                extracted_value = cert_details.get('instrument_description', '')
            elif db_field == 'Make' and cert_details.get('manufacturer'):
                extracted_value = cert_details.get('manufacturer', '')
            elif db_field == 'Instrument / Equipment / Software (Sr. No.)' and cert_details.get('serial_no'):
                extracted_value = cert_details.get('serial_no', '')
            elif db_field == 'Unique Identity No.' and cert_details.get('identification_no'):
                extracted_value = cert_details.get('identification_no', '')
            elif db_field == 'Range' and cert_details.get('capacity_range'):
                extracted_value = cert_details.get('capacity_range', '')
            elif db_field == 'Least count' and cert_details.get('resolution'):
                extracted_value = cert_details.get('resolution', '')
            elif db_field == 'Cal Date' and cert_details.get('calibration_date'):
                extracted_value = cert_details.get('calibration_date', '')
            elif db_field == 'Due Date' and cert_details.get('due_date'):
                extracted_value = cert_details.get('due_date', '')
            elif db_field == 'Calibration Certificate No.':
                extracted_value = cert_details.get('certificate_number', '')
            
            extracted_display = extracted_value if extracted_value else '❌ Not in Certificate'
            
            # ============ INTELLIGENT COMPARISON ============
            
            # Special handling for Range
            if db_field == 'Range':
                is_match = self.compare_range(extracted_value, master_value)
                status = '✅ Match' if is_match else '❌ Mismatch'
            
            # Special handling for Least Count
            elif db_field == 'Least count':
                is_match = self.compare_least_count(extracted_value, master_value)
                status = '✅ Match' if is_match else '❌ Mismatch'
            
            # Special handling for Dates
            elif db_field in ['Cal Date', 'Due Date']:
                extracted_date = self.normalize_date(extracted_value)
                master_date = self.normalize_date(master_value)
                
                if extracted_date and master_date:
                    if extracted_date == master_date:
                        status = '✅ Match'
                    else:
                        status = '❌ Mismatch'
                elif extracted_value and master_value:
                    # Try direct string comparison after cleaning
                    extracted_clean = re.sub(r'[^\w\s]', '', str(extracted_value).lower())
                    master_clean = re.sub(r'[^\w\s]', '', str(master_value).lower())
                    if extracted_clean == master_clean:
                        status = '✅ Match'
                    else:
                        status = '⚠️ Date format issue'
                else:
                    status = '⚪ Missing'
            
            # Default comparison with normalization
            else:
                extracted_normalized = self.normalize_text(extracted_value)
                master_normalized = self.normalize_text(master_value)
                
                if extracted_value and master_value and pd.notna(master_value):
                    if extracted_normalized == master_normalized:
                        status = '✅ Match'
                    else:
                        status = '❌ Mismatch'
                elif extracted_value and (not master_value or pd.isna(master_value)):
                    status = '⚠️ Missing in Master DB'
                elif (not extracted_value) and master_value and pd.notna(master_value):
                    status = '⚠️ Missing in Certificate'
                else:
                    status = '⚪ Both Missing'
            
            comparisons.append({
                'Field': display_name,
                'Extracted from Certificate': extracted_display,
                'Master Database': master_display,
                'Status': status
            })
        
        # Certificate Details section
        if cert_details:
            comparisons.append({'Field': '--- ADDITIONAL CERTIFICATE INFO ---', 'Extracted from Certificate': '---', 'Master Database': '---', 'Status': '---'})
            
            cert_detail_fields = [
                ('customer_name', 'Customer Name'),
                ('customer_address', 'Customer Address'),
                ('model_no', 'Model No'),
                ('calibration_site', 'Calibration Site'),
                ('receipt_date', 'Receipt Date'),
                ('issue_date', 'Issue Date'),
                ('calibrated_by', 'Calibrated By'),
                ('reviewed_by', 'Reviewed By'),
                ('approved_by', 'Approved By'),
                ('temperature', 'Temperature'),
                ('humidity', 'Humidity')
            ]
            
            for field, display_name in cert_detail_fields:
                value = cert_details.get(field, '')
                if value:
                    comparisons.append({
                        'Field': display_name,
                        'Extracted from Certificate': value,
                        'Master Database': 'Not in Master DB',
                        'Status': 'ℹ️ Extra Info'
                    })
        
        return pd.DataFrame(comparisons)

def process_single_pdf(validator, pdf_file) -> List[Dict]:
    try:
        certificates = validator.process_pdf_pages(pdf_file)
        results = []
        
        for cert in certificates:
            cert_number = cert['cert_number']
            cert_details = cert.get('details', {})
            master_record = validator.find_in_master(cert_number)
            
            if master_record is not None:
                due_date = master_record.get('Due Date')
                expiry = validator.get_expiry_status(due_date)
                comparison_df = validator.create_comparison_table(cert_details, master_record)
                
                results.append({
                    'filename': pdf_file.name,
                    'certificate_number': cert_number,
                    'page_num': cert['page_num'],
                    'found_in_master': True,
                    'comparison_df': comparison_df,
                    'cert_details': cert_details,
                    'instrument': master_record.get('Instrument', 'N/A'),
                    'make': master_record.get('Make', 'N/A'),
                    'serial_no': master_record.get('Instrument / Equipment / Software (Sr. No.)', 'N/A'),
                    'cal_date': master_record.get('Cal Date', 'N/A'),
                    'due_date': due_date,
                    'expiry_status': expiry['status'],
                    'expiry_message': expiry['message'],
                    'days_left': expiry['days_left'],
                    'location': master_record.get('User Location', 'N/A'),
                    'agency': master_record.get('Calibration Agency', 'N/A'),
                    'remarks': master_record.get('Remarks', 'N/A'),
                    'error': None
                })
            else:
                results.append({
                    'filename': pdf_file.name,
                    'certificate_number': cert_number,
                    'page_num': cert['page_num'],
                    'found_in_master': False,
                    'error': f'Certificate {cert_number} not found in master database'
                })
        
        return results
    except Exception as e:
        return [{
            'filename': pdf_file.name,
            'certificate_number': None,
            'found_in_master': False,
            'error': str(e)
        }]

def generate_batch_excel_report(results: List[Dict]) -> bytes:
    report_data = []
    for r in results:
        report_data.append({
            'Filename': r.get('filename', ''),
            'Certificate Number': r.get('certificate_number', ''),
            'Page Number': r.get('page_num', ''),
            'Status': 'FOUND' if r.get('found_in_master') else 'NOT FOUND',
            'Expiry Status': r.get('expiry_status', ''),
            'Instrument': r.get('instrument', ''),
            'Make': r.get('make', ''),
            'Serial No': r.get('serial_no', ''),
            'Calibration Date': r.get('cal_date', ''),
            'Due Date': r.get('due_date', ''),
            'Days Left': r.get('days_left', ''),
            'Location': r.get('location', ''),
            'Agency': r.get('agency', ''),
            'Remarks': r.get('remarks', ''),
            'Error': r.get('error', '')
        })
    
    df = pd.DataFrame(report_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Batch Results', index=False)
        worksheet = writer.sheets['Batch Results']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    output.seek(0)
    return output.getvalue()

def generate_expiry_excel_report(master_df: pd.DataFrame, report_type: str) -> bytes:
    today = datetime.now().date()
    report_data = []
    
    cert_col = None
    for col in master_df.columns:
        if 'certificate' in col.lower():
            cert_col = col
            break
    
    for idx, row in master_df.iterrows():
        due_date = row.get('Due Date')
        if pd.notna(due_date):
            try:
                if isinstance(due_date, str):
                    try:
                        due = datetime.strptime(due_date, '%d-%b-%y').date()
                    except:
                        try:
                            due = datetime.strptime(due_date, '%Y-%m-%d').date()
                        except:
                            due = pd.to_datetime(due_date).date()
                else:
                    due = pd.to_datetime(due_date).date()
                
                days_left = (due - today).days
                
                if report_type == 'expired' and days_left < 0:
                    report_data.append({
                        'Certificate No': row.get(cert_col, '') if cert_col else '',
                        'Instrument': row.get('Instrument', ''),
                        'Make': row.get('Make', ''),
                        'Serial No': row.get('Instrument / Equipment / Software (Sr. No.)', ''),
                        'Due Date': due.strftime('%Y-%m-%d'),
                        'Days Overdue': abs(days_left),
                        'Location': row.get('User Location', '')
                    })
                elif report_type == 'expiring' and 0 <= days_left <= 60:
                    status = 'Critical' if days_left <= 7 else 'Warning' if days_left <= 30 else 'Info'
                    report_data.append({
                        'Certificate No': row.get(cert_col, '') if cert_col else '',
                        'Instrument': row.get('Instrument', ''),
                        'Make': row.get('Make', ''),
                        'Serial No': row.get('Instrument / Equipment / Software (Sr. No.)', ''),
                        'Due Date': due.strftime('%Y-%m-%d'),
                        'Days Left': days_left,
                        'Status': status,
                        'Location': row.get('User Location', '')
                    })
            except:
                pass
    
    df = pd.DataFrame(report_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=report_type, index=False)
        worksheet = writer.sheets[report_type]
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    output.seek(0)
    return output.getvalue()

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Calibration Certificate Validator</h1>
        <p>Upload PDFs → Auto-detects certificates → Full comparison with master database → Tracks expiry</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("OfficeFlowAI")
        st.image("OfficeFlow Ai-01-01.png", width=120)
        
        st.divider()
        
        st.header("📊 Master Database")
        master_file = st.file_uploader("Upload Trail master list.xlsx", type=['xlsx', 'xls'])
        
        if master_file:
            validator = CertificateValidator(GEMINI_API_KEY)
            success, msg = validator.load_master_excel(master_file)
            if success:
                st.session_state.master_df = validator.master_df
                st.session_state.validator = validator
                st.success(msg)
                
                cert_col = validator.get_certificate_column()
                if cert_col:
                    st.metric("Total Records", len(st.session_state.master_df))
                    st.metric("Unique Certificates", st.session_state.master_df[cert_col].nunique())
            else:
                st.error(msg)
    
    if st.session_state.master_df is None:
        st.info("👈 Please upload the master Excel file first")
        return
    
    if st.session_state.validator:
        st.session_state.validator.processed_certs = set()
    
    # ==================== EXPIRY REPORTS SECTION ====================
    st.header("📊 Certificate Expiry Reports")
    
    due_col = 'Due Date'
    today = datetime.now().date()
    
    expired_list = []
    critical_list = []
    warning_list = []
    info_list = []
    
    if due_col in st.session_state.master_df.columns:
        cert_col = st.session_state.validator.get_certificate_column()
        
        for idx, row in st.session_state.master_df.iterrows():
            due = row.get(due_col)
            if pd.notna(due):
                try:
                    if isinstance(due, str):
                        try:
                            due_date = datetime.strptime(due, '%d-%b-%y').date()
                        except:
                            try:
                                due_date = datetime.strptime(due, '%Y-%m-%d').date()
                            except:
                                due_date = pd.to_datetime(due).date()
                    else:
                        due_date = pd.to_datetime(due).date()
                    
                    days_left = (due_date - today).days
                    
                    record = {
                        'Certificate No': row.get(cert_col, '') if cert_col else '',
                        'Instrument': row.get('Instrument', ''),
                        'Make': row.get('Make', ''),
                        'Serial No': row.get('Instrument / Equipment / Software (Sr. No.)', ''),
                        'Due Date': due_date.strftime('%Y-%m-%d'),
                        'Days Left': days_left,
                        'Location': row.get('User Location', '')
                    }
                    
                    if days_left < 0:
                        record['Status'] = '🔴 EXPIRED'
                        expired_list.append(record)
                    elif days_left <= 7:
                        record['Status'] = '🔴 CRITICAL'
                        critical_list.append(record)
                    elif days_left <= 30:
                        record['Status'] = '🟠 WARNING'
                        warning_list.append(record)
                    elif days_left <= 60:
                        record['Status'] = '🟡 INFO'
                        info_list.append(record)
                except:
                    pass
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔴 Expired", len(expired_list))
    with col2:
        st.metric("🔴 Critical (1-7 days)", len(critical_list))
    with col3:
        st.metric("🟠 Warning (8-30 days)", len(warning_list))
    with col4:
        st.metric("🟡 Info (31-60 days)", len(info_list))
    
    col1, col2 = st.columns(2)
    with col1:
        if expired_list:
            expired_excel = generate_expiry_excel_report(st.session_state.master_df, 'expired')
            st.download_button(
                label=f"📥 Download Expired Certificates ({len(expired_list)})",
                data=expired_excel,
                file_name=f"expired_certificates_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    with col2:
        if critical_list or warning_list or info_list:
            expiring_excel = generate_expiry_excel_report(st.session_state.master_df, 'expiring')
            st.download_button(
                label=f"📥 Download Expiring Soon Certificates ({len(critical_list) + len(warning_list) + len(info_list)})",
                data=expiring_excel,
                file_name=f"expiring_certificates_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    st.divider()
    
    # ==================== EXPIRY TABLES ====================
    if expired_list:
        st.subheader("🔴 Expired Certificates")
        st.dataframe(pd.DataFrame(expired_list), use_container_width=True, hide_index=True)
    else:
        st.info("✅ No expired certificates found")
    
    if critical_list:
        st.subheader("🔴 Critical - Expiring in 1-7 Days")
        critical_df = pd.DataFrame(critical_list)
        critical_df = critical_df.sort_values('Days Left')
        st.dataframe(critical_df, use_container_width=True, hide_index=True)
    else:
        st.info("✅ No certificates expiring in 1-7 days")
    
    if warning_list:
        st.subheader("🟠 Warning - Expiring in 8-30 Days")
        warning_df = pd.DataFrame(warning_list)
        warning_df = warning_df.sort_values('Days Left')
        st.dataframe(warning_df, use_container_width=True, hide_index=True)
    else:
        st.info("✅ No certificates expiring in 8-30 days")
    
    if info_list:
        st.subheader("🟡 Info - Expiring in 31-60 Days")
        info_df = pd.DataFrame(info_list)
        info_df = info_df.sort_values('Days Left')
        st.dataframe(info_df, use_container_width=True, hide_index=True)
    else:
        st.info("✅ No certificates expiring in 31-60 days")
    
    st.divider()
    
    # ==================== UPLOAD MODE SELECTION ====================
    st.header("📄 Certificate Validation")
    
    upload_mode = st.radio("Select mode:", ["Single PDF File", "Multiple PDF Files (Bulk Upload)"], horizontal=True)
    
    # ==================== SINGLE PDF MODE ====================
    if upload_mode == "Single PDF File":
        cert_file = st.file_uploader("Choose PDF file", type=['pdf'], key="single_cert")
        
        if cert_file:
            if st.button("🚀 Extract & Compare", type="primary", use_container_width=True):
                validator = st.session_state.validator
                validator.processed_certs = set()
                
                with st.spinner("Processing PDF - detecting certificates page by page..."):
                    results = process_single_pdf(validator, cert_file)
                
                if results:
                    st.success(f"✅ Found {len(results)} certificate(s) in this PDF")
                    
                    for idx, result in enumerate(results):
                        st.markdown(f"---")
                        st.markdown(f"### Certificate {idx+1}: {result.get('certificate_number', 'Unknown')}")
                        st.info(f"📄 Found on Page: {result.get('page_num', 'Unknown')}")
                        
                        if result.get('found_in_master'):
                            expiry_color = {
                                'EXPIRED': 'error-box',
                                'CRITICAL': 'error-box',
                                'EXPIRING SOON': 'warning-box',
                                'VALID': 'success-box'
                            }.get(result.get('expiry_status', ''), 'info-box')
                            
                            st.markdown(f"""
                            <div class="{expiry_color}">
                                <strong>📅 {result.get('expiry_message', '')}</strong>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.subheader("📊 Full Comparison: Certificate vs Master Database")
                            comparison_df = result.get('comparison_df')
                            if comparison_df is not None:
                                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        else:
                            st.error(f"❌ {result.get('error', 'Certificate not found in master database')}")
                    
                    st.balloons()
    
    # ==================== BULK UPLOAD MODE ====================
    else:
        cert_files = st.file_uploader(
            "Choose multiple PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            key="bulk_certs",
            help="Upload multiple PDF files, each may contain multiple certificates"
        )
        
        if cert_files:
            st.info(f"📁 {len(cert_files)} file(s) selected")
            
            if st.button("🚀 Process All Files", type="primary", use_container_width=True):
                validator = st.session_state.validator
                validator.processed_certs = set()
                all_results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, cert_file in enumerate(cert_files):
                    status_text.text(f"Processing {cert_file.name}...")
                    results = process_single_pdf(validator, cert_file)
                    all_results.extend(results)
                    progress_bar.progress((i + 1) / len(cert_files))
                
                status_text.text("✅ Processing complete!")
                
                st.subheader("📊 Results Summary")

                results_data = []
                for r in all_results:
                    # Get days remaining
                    days_left = r.get('days_left')
                    expiry_status = r.get('expiry_status', 'N/A')
                    
                    # Format days remaining display
                    if days_left is not None:
                        if days_left < 0:
                            days_display = f"🔴 {days_left} days"
                            expiry_display = f"{expiry_status}"
                        elif days_left == 0:
                            days_display = f"🔴 0 days"
                            expiry_display = f"{expiry_status}"
                        elif days_left <= 7:
                            days_display = f"{days_left} days"
                            expiry_display = f"🔴 {expiry_status}"
                        elif days_left <= 30:
                            days_display = f"🟠 {days_left} days"
                            expiry_display = f"{expiry_status}"
                        elif days_left <= 60:
                            days_display = f"🟡 {days_left} days"
                            expiry_display = f"{expiry_status}"
                        else:
                            days_display = f"🟢 {days_left} days"
                            expiry_display = f"{expiry_status}"
                    else:
                        days_display = 'N/A'
                        expiry_display = '⚪ No Date'
                    
                    results_data.append({
                        'File': r['filename'],
                        'Certificate No': r.get('certificate_number', 'N/A'),
                        'Page': r.get('page_num', 'N/A'),
                        'Status': '✅ Found' if r.get('found_in_master') else '❌ Not Found',
                        'Expiry Status': expiry_display,
                        'Days Remaining': days_display,
                        'Instrument': r.get('instrument', ''),
                        'Due Date': r.get('due_date', '')
                    })
                                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                excel_data = generate_batch_excel_report(all_results)
                st.download_button(
                    label="📥 Download Complete Report (Excel)",
                    data=excel_data,
                    file_name=f"certificate_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
                found_results = [r for r in all_results if r.get('found_in_master')]
                if found_results:
                    st.subheader("📋 Detailed Comparison Results")
                    for r in found_results:
                        with st.expander(f"📄 Certificate {r['certificate_number']} - {r['instrument']} (Page {r.get('page_num', 'N/A')})"):
                            comparison_df = r.get('comparison_df')
                            if comparison_df is not None:
                                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()