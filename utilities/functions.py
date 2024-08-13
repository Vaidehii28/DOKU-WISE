from google.cloud import vision
from langdetect import detect
from utilities.config import device
from utilities.constants import language_map
from model import get_bert_tokenizer
import json

def extract_text_from_image(client, content):
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    extracted_text = texts[0].description if texts else ""
    return extracted_text


def create_prompt_for_14(doc_type, text):
    prompts = {
        'Aadhaar-masked': f"Extract Aadhaar number, Name, Gender, DOB, Address (if present): {text}",
        'Aadhaar-unmasked': f"Extract Aadhaar number, Name, Gender, DOB, Address (if present): {text}",
        'PAN': f"Extract Permanent Account Number, Name, Father name, Date of Birth, Date of Incorporation/Formation (if present): {text}",
        'DrivingLicense': f"Extract DL No., Name, Address, DOB, Issue date, Validity NT, Validity TR (if present): {text}",
        'Passport': f"Extract Passport No., SurName, Given Name(s), Date of Birth, Place of Birth, Issue date, Date of Expiry, Place of Issue: {text}",
        'VoterId': f"Extract Voter Id number, Name, Gender, Father name, Husband name (if present), Date of Birth, Address (if present): {text}",
        'Udyam': f"Extract Udyam Registration Number, Type of Enterprise/ Organization, Major Activity, SOCIAL CATEGORY OF ENTREPRENEUR, NAME OF UNIT(S), OFFICIAL ADDRESS OF ENTERPRISE, Flat/Door/Block No., Name of Premises/ Building, Village/Town, Block, Road/Street/Lane, City, State, District, Mobile, Email, DATE OF INCORPORATION / REGISTRATION OF ENTERPRISE, DATE OF COMMENCEMENT OF PRODUCTION/BUSINESS, NATIONAL INDUSTRY CLASSIFICATION CODE(S), DATE OF UDYAM REGISTRATION, Date of printing: {text}",
        'GST-Registration-Certificate': f"Extract Registration number, Legal Name, Trade Name, Constitution of Business, Address of Principal Place of Business, Period of Validity, Type of Registration, Date of Issue of Certificate, Details of Additional Places of Business: {text}",
        'Import-Export-Certificate': f"Extract IEC, PAN, Firm Name, Nature of Concern, Date of Issue, Registered Address, Issued From File No.: {text}",
        'Rent-Agreement': f"Extract address, first party, second party, issue date/comencement date, and expiry date or period if expiry date is not mentioned: {text}",
        'Food-DrugLicenseCertificate': f"Extract Registration number, License number, Name of Food Business Operator, Permanent address of Food Business Operator, Address of location where food business is to be conducted/premises, Kind of Business, Issued on, Valid upto: {text}",
        'IncomeTaxReturn': f"Extract Assessment Year, PAN, Name, Address, Status, Form Number, e-filing acknowledgement number, Total income, Net tax payable, Taxes paid: {text}",
        'Shop-Establishment-Act': f"Extract Registration number, Name of establishment, Address of establishment, Employer name, Employer address, Date of issue, Expiry date: {text}",
        'UtilityBills': f"Extract Name, Address: {text}"
    }
    prompt = prompts.get(doc_type, f"Extract information: {text}")
    return prompt

def generate_predictions_for_14(model, tokenizer, doc_type, text):
    if doc_type == 'Other':
        return None
    prompt = create_prompt_for_14(doc_type, text)
    print("Prompt created:", prompt) 
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output_ids = model.generate(input_ids, max_length=256)
    prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return prediction

def parse_extracted_fields(fields_json):
    parsed_fields = json.loads(fields_json)
    return [{"Key": key, "Value": value} for key, value in parsed_fields.items()]

def get_language_name(text):
    try:
        detected_language = detect(text)
        return language_map.get(detected_language, "unknown")
    except:
        return "unknown"
    
def tokenize_texts(texts):
    bert_tokenizer = get_bert_tokenizer()
    return bert_tokenizer(texts, padding=True, truncation=True, return_tensors='pt')