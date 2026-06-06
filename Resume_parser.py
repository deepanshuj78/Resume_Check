import pdfplumber
import re
import os
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("'en_core_web_sm' not found. Installing Now...")
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load("en_core_web_sm")
TECH_SKILLS_BANK = [
    "python", "java", "sql", "javascript", "react", "node", "html", "css",
    "machine learning", "data science", "nlp", "pandas", "numpy", "aws", "git",
    "docker", "kubernetes", "excel", "tableau", "power bi"
]
def extract_text_from_pdf(pdf_path):
    raw_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                raw_text += text + "\n"
    return raw_text
def clean_resume_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'[^\w\s.,+-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
def parse_features(clean_text):
    doc = nlp(clean_text)
    tokens = [token.text.lower().strip() for token in doc if not token.is_space]
    extracted_skills = []
    for skill in TECH_SKILLS_BANK:
        if skill in tokens or skill in clean_text.lower():
            extracted_skills.append(skill)
    experience_matches = re.findall(r'(\d+)\+?\s*(?:years?|yrs?)(?:\s*of\s*experience)?', clean_text, re.IGNORECASE)
    experience_years = max([int(year) for year in experience_matches]) if experience_matches else 0
    name = "Unknown Candidate"
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break            
    return name, extracted_skills, experience_years
def train_and_classify_resume(target_resume_text):
    training_data = [
        "python data science machine learning model analytics sql pandas numpy",
        "recruitment payroll onboarding human resources interview hiring employee benefits",
        "javascript react html css node frontend developer web app java design"
    ]
    labels = ["Data Scientist", "HR Specialist", "Web Developer"]
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train = vectorizer.fit_transform(training_data)
    model = LogisticRegression()
    model.fit(X_train, labels)
    X_target = vectorizer.transform([target_resume_text])
    predicted_category = model.predict(X_target)
    return predicted_category[0]
def process_multiple_resumes(pdf_list):
    all_candidate_records = []
    for pdf_filename in pdf_list:
        try:
            print(f"\n Processing: {pdf_filename}")
            raw_text = extract_text_from_pdf(pdf_filename)
            cleaned_text = clean_resume_text(raw_text)
            name, skills, experience = parse_features(cleaned_text)
            category = train_and_classify_resume(cleaned_text)
            candidate_record = {
                "Name": name if name != "Unknown Candidate" else pdf_filename,
                "Category_Tag": category,
                "Skills": ", ".join(skills) if skills else "None Detected",
                "Experience": f"{experience} Years" if experience > 0 else "Not Specified",
                "Cleaned_Text": cleaned_text
            }
            all_candidate_records.append(candidate_record)
        except FileNotFoundError:
            print(f"Error: File '{pdf_filename}' not found. Skipping.")
        except Exception as e:
            print(f"Error processing '{pdf_filename}': {str(e)}")
    df = pd.DataFrame(all_candidate_records)
    return df
def calculate_match_scores(job_description, resumes_df):
    if resumes_df.empty or not job_description.strip():
        resumes_df['Match_Score'] = 0.0
        return resumes_df
    resumes_df = resumes_df[resumes_df['Cleaned_Text'].str.strip() != ""].copy()
    if resumes_df.empty:
        return resumes_df
    corpus = [job_description] + resumes_df['Cleaned_Text'].tolist()
    custom_stop_words = list(spacy.lang.en.stop_words.STOP_WORDS) + [
        "company", "job", "person", "team", "role", "work", "responsibilities", "resume"
    ]
    vectorizer = TfidfVectorizer(stop_words=custom_stop_words)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    resumes_df['Match_Score'] = [round(float(score) * 100, 1) for score in scores]
    resumes_df = resumes_df.sort_values(by='Match_Score', ascending=False)
    return resumes_df
if __name__ == "__main__":
    mock_job_description = "Looking for a Python Developer with experience in SQL databases and data science."
    resumes_to_test = ["Stockholm-Resume-Template-Simple.pdf"]    
    base_dataframe = process_multiple_resumes(resumes_to_test)    
    if not base_dataframe.empty:
        ranked_dataframe = calculate_match_scores(mock_job_description, base_dataframe)
        print("\n--- Final AI Ranked Candidate Leaderboard ---")
        display_columns = ["Name", "Category_Tag", "Match_Score", "Skills", "Experience"]
        print(ranked_dataframe[display_columns].to_string(index=False))