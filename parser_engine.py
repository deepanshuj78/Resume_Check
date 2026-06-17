import pdfplumber
import re
import spacy
from config import FLAT_SKILLS_BANK
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import os
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load("en_core_web_sm")
def extract_text_from_pdf(pdf_path):
    raw_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                raw_text += text + "\n"
    return raw_text
def clean_text_stream(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s.,+-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
def parse_candidate_metadata(clean_text):
    doc = nlp(clean_text)
    text_lower = clean_text.lower()
    extracted_skills = set()
    tokens = {token.text.lower().strip() for token in doc if not token.is_space}
    
    # Boundary-safe lookup matching
    for skill in FLAT_SKILLS_BANK:
        if " " not in skill and skill in tokens:
            extracted_skills.add(skill)
        elif " " in skill or skill == "git":
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                extracted_skills.add(skill)
    exp_matches = re.findall(r'(\d+)\+?\s*(?:years?|yrs?)(?:\s*of\s*experience)?', clean_text, re.IGNORECASE)
    experience_years = max([int(year) for year in exp_matches]) if exp_matches else 0
    name = "Unknown Candidate"
    for ent in doc.ents:
        if ent.label_ == "PERSON" and len(ent.text.strip()) > 2:
            name = ent.text.strip()
            break
            
    return name, list(extracted_skills), experience_years
