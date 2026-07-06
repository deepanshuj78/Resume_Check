import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Config import FINAL_STOP_WORDS
from parser_engine import parse_candidate_metadata
def compute_composite_scores(job_description, candidates_df):
    """Computes robust multi-criteria scores mimicking standard web platform thresholds."""
    original_df = candidates_df.copy()
    if original_df.empty or not job_description.strip():
        original_df['Match_Score'] = 0.0
        return original_df.sort_values(by='Match_Score', ascending=False)
    filtered_df = original_df[original_df['Cleaned_Text'].str.strip() != ""].copy()
    if filtered_df.empty:
        original_df['Match_Score'] = 0.0
        return original_df.sort_values(by='Match_Score', ascending=False)
    corpus = [job_description] + filtered_df['Cleaned_Text'].tolist()
    stop_words_list = list(FINAL_STOP_WORDS) if FINAL_STOP_WORDS else None
    vectorizer = TfidfVectorizer(stop_words=stop_words_list)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    cosine_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    _, target_skills, _ = parse_candidate_metadata(job_description)
    target_skills_set = {skill.lower().strip() for skill in target_skills} if target_skills else set()
    denom_target = len(target_skills_set)
    score_mapping = {}
    for idx, row in enumerate(filtered_df.itertuples()):
        resume_skills = {skill.lower().strip() for skill in row.Skills_List} if hasattr(row, 'Skills_List') and isinstance(row.Skills_List, list) else set()
        
        if denom_target > 0:
            overlap = resume_skills.intersection(target_skills_set)
            overlap_count = len(overlap)
            skill_score = overlap_count / denom_target
        else:
            overlap_count = 0
            skill_score = 0.5
        
        # Boost semantic similarity with power function to amplify scores
        semantic_component = float(cosine_scores[idx])
        semantic_boosted = (semantic_component ** 0.7)  # Reduce exponent power for higher scores
        
        # Increased overlap bonus calculation
        overlap_bonus = min(0.35, overlap_count * 0.08)  # Increased from 0.30 and 0.05
        
        # Improved composite formula with better weighting
        composite_score = (semantic_boosted * 0.50) + (skill_score * 0.50) + overlap_bonus
        
        # Apply multiplicative boost to final score
        boosted_score = composite_score * 1.15 + 0.08  # Scale up by 15% and add 8% base
        score_mapping[row.Index] = min(1.0, max(0.0, boosted_score))
    original_df['Match_Score'] = original_df.index.map(score_mapping).fillna(0.0)
    
    return original_df.sort_values(by='Match_Score', ascending=False)
