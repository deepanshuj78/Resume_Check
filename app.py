import pdfplumber
import re
import os
import pandas as pd
import streamlit as st
from Resume_parser import extract_text_from_pdf, clean_resume_text, parse_features, calculate_match_scores, train_and_classify_resume
st.set_page_config(page_title="AI Resume Screener", page_icon="🚀", layout="wide")
st.title("🚀 AI Single-Resume Screener Dashboard")
st.write("Evaluate a single candidate profile against job requirements instantly.")
col1,col2 = st.columns(2)
with col1:
    st.subheader("📋 Step 1: Define Job Requirements")
    job_desc = st.text_area(
        "Paste Job Description text here:", 
        height=200, 
        placeholder="e.g. Seeking a Python Developer with experience in SQL databases and machine learning models..."
    )
with col2:
    st.subheader("Step 2: Upload Application")
    uploaded_file = st.file_uploader(
        "Drop exactly ONE candidate PDF resume here", 
        type=["pdf"], 
        accept_multiple_files=False
    )
st.markdown("---")
if st.button("Run AI Match Ranker"):
    if not job_desc.strip():
        st.error("Error: Please paste a job description context first.")
    elif not uploaded_file:  
        st.error("Error: Please upload a resume PDF to evaluate.")
    else:
        processed_candidates = []
        temp_filename = f"temp_{uploaded_file.name}"
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            raw = extract_text_from_pdf(temp_filename)
            cleaned = clean_resume_text(raw)
            name, skills, exp = parse_features(cleaned)
            category = train_and_classify_resume(cleaned)
            processed_candidates.append({
                "Name": name if name != "Unknown Candidate" else uploaded_file.name,
                "Category_Tag": category if isinstance(category, (list, tuple)) else str(category),
                "Skills": ", ".join(skills) if skills else "None Detected",
                "Experience": f"{exp} Years" if exp > 0 else "Not Specified",
                "Cleaned_Text": cleaned
            })
        except Exception as e:
            st.error(f"System failure reading file {uploaded_file.name}: {str(e)}")
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        base_df = pd.DataFrame(processed_candidates)
        if not base_df.empty:
            with st.spinner("AI Core alignment calculator engine executing matrices..."):
                final_ranked_df = calculate_match_scores(job_desc, base_df) 
            st.success("✅ Analysis Complete! Target profile matrix generated below:")
            display_columns = ["Name", "Match_Score", "Category_Tag", "Skills", "Experience"]
            st.dataframe(
                final_ranked_df[display_columns], 
                use_container_width=True, 
                hide_index=True
            )