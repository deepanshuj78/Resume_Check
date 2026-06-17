import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Config import FINAL_STOP_WORDS
from parser_engine import parse_candidate_metadata
def compute_composite_scores(job_description, candidates_df):
    """Computes robust multi-criteria scores preventing keyword stuffing cheats."""
    if candidates_df.empty or not job_description.strip():
        candidates_df['Match_Score'] = 0.0
        return candidates_df
    candidates_df = candidates_df[candidates_df['Cleaned_Text'].str.strip() != ""].copy()
    corpus = [job_description] + candidates_df['Cleaned_Text'].tolist()
    vectorizer = TfidfVectorizer(stop_words=FINAL_STOP_WORDS)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    cosine_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    _, target_skills, _ = parse_candidate_metadata(job_description)
    target_skills_set = set(target_skills)
    final_scores = []
    for idx, row in enumerate(candidates_df.itertuples()):
        resume_skills = set(row.Skills_List)
        if target_skills_set:
            overlap = resume_skills.intersection(target_skills_set)
            skill_score = len(overlap) / len(target_skills_set)
        else:
            skill_score = 0.5 
        semantic_component = float(cosine_scores[idx])
        weighted_score = (semantic_component * 0.60) + (skill_score * 0.40)
        final_scores.append(round(weighted_score * 100, 1))
    candidates_df['Match_Score'] = final_scores
    return candidates_df.sort_values(by='Match_Score', ascending=False)