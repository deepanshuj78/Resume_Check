import pdfplumber
import re
import os
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

# 1. Spacy Linguistic Brain Loading Pipeline
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("'en_core_web_sm' not found. Installing Now...")
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load("en_core_web_sm")

# 2. Expanded Technical Taxonomy Bank
TECH_SKILLS_BANK = [
    "python", "java", "sql", "javascript", "typescript", "react", "node", "html", "css",
    "express", "mongodb", "next.js", "git", "docker", "aws", "kubernetes", "excel", 
    "tableau", "power bi", "machine learning", "data science", "nlp", "pandas", "numpy"
]

def extract_text_from_pdf(pdf_path):
    """Safely extracts raw plain-text streams out of binary PDF matrices."""
    raw_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                raw_text += text + "\n"
    return raw_text

def clean_resume_text(text):
    """Strips layout formatting noise, hyperlinks, and emails."""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'[^\w\s.,+-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_features(clean_text):
    """Performs boundary-safe skill indexing and experience calculation."""
    doc = nlp(clean_text)
    text_lower = clean_text.lower()
    
    # Boundary-Safe Skill Extraction to stop partial-word false matches
    extracted_skills = []
    tokens = {token.text.lower().strip() for token in doc if not token.is_space}
    
    for skill in TECH_SKILLS_BANK:
        if " " not in skill and skill in tokens:
            extracted_skills.append(skill)
        elif " " in skill or skill == "git":
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                extracted_skills.append(skill)

    # Advanced Experience Extraction Matrix
    experience_matches = re.findall(r'(\d+)\+?\s*(?:years?|yrs?|years of experience)\b', clean_text, re.IGNORECASE)
    experience_years = max([int(year) for year in experience_matches]) if experience_matches else 0
    
    # Isolate applicant names using NER
    name = "Unknown Candidate"
    for ent in doc.ents:
        if ent.label_ == "PERSON" and len(ent.text.strip()) > 2:
            name = ent.text.strip()
            break            
    return name, extracted_skills, experience_years

def train_and_classify_resume(target_resume_text):
    """Categorizes candidates against structural industrial domains."""
    training_data = [
        "python data science machine learning model analytics sql pandas numpy",
        "recruitment payroll onboarding human resources interview hiring employee benefits",
        "javascript react html css node frontend developer web app java design typescript angular"
    ]
    labels = ["Data Scientist", "HR Specialist", "Web Developer"]
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train = vectorizer.fit_transform(training_data)
    model = LogisticRegression()
    model.fit(X_train, labels)
    X_target = vectorizer.transform([target_resume_text])
    predicted_category = model.predict(X_target)
    return str(predicted_category)

def calculate_match_scores(job_description, resumes_df):
    """Applies a composite matching formula returning true percentage baselines."""
    if resumes_df.empty or not job_description.strip():
        resumes_df['Match_Score'] = 0.0
        return resumes_df
        
    resumes_df = resumes_df[resumes_df['Cleaned_Text'].str.strip() != ""].copy()
    if resumes_df.empty:
        return resumes_df
        
    # Semantic Vector Context Math
    corpus = [job_description] + resumes_df['Cleaned_Text'].tolist()
    spacy_stops = set(spacy.lang.en.stop_words.STOP_WORDS)
    custom_extensions = {"company", "job", "person", "team", "role", "work", "responsibilities", "resume"}
    final_stop_words = list(spacy_stops.union(custom_extensions))
    vectorizer = TfidfVectorizer(stop_words=final_stop_words)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    cosine_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    _, required_skills, _ = parse_features(job_description)
    required_skills_set = set(required_skills)
    final_scores = []
    for idx, row in enumerate(resumes_df.itertuples()):
        candidate_skills_raw = row.Skills.split(", ") if isinstance(row.Skills, str) else list(row.Skills)
        candidate_skills_set = {s.lower().strip() for s in candidate_skills_raw}
        if required_skills_set:
            matched = candidate_skills_set.intersection(required_skills_set)
            skill_score = len(matched) / len(required_skills_set)
        else:
            skill_score = 0.5
        semantic_component = float(cosine_scores[idx])
        weighted_score = (semantic_component * 0.60) + (skill_score * 0.40)
        final_scores.append(round(weighted_score * 100, 1))        
    resumes_df['Match_Score'] = final_scores
    return resumes_df.sort_values(by='Match_Score', ascending=False)
