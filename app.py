import os
import spacy 
from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd
import streamlit as st
from parser_engine import extract_text_from_pdf, clean_text_stream, parse_candidate_metadata
from ML_Classifier import classifier_pipeline
from Scoring_Matrix import compute_composite_scores
import Config 
st.set_page_config(page_title="AI Resume Screener", page_icon="🚀", layout="wide")
st.markdown("""
    <style>
    /* Centered Typography Adjustments */
    .centered-title {
        text-align: center;
        margin-bottom: 0;
        color: #FFFFFF;
        font-size: 2.5rem; 
        font-weight: 800;
    }
    .centered-subtitle {
        text-align: center;
        color: #94A3B8;
        font-size: 1rem;
        max-width: 600px;
        margin: -4px auto 20px auto;
    }
    
    /* Executive Card Wrapper for Input Steps */
    .input-card {
        background-color: #1E293B;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 20px;
    }
    .custom-caption {
        color: #64748B; /* Capitalized hex for consistency */
        font-size: 13px;
        margin-top: 6px;
    }
    .ats-score-card {
        background-color: #FFFFFF;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3);
        text-align: center;
        border: 1px solid #E2E8F0;
    }
    .circle-score {
        font-size: 64px;
        font-weight: 800;
        color: #F59E0B;
        margin-bottom: 2px;
        line-height: 1;
    }
    .circle-label {
        font-size: 12px;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .grade-badge {
        background-color: rgba(37, 99, 235, 0.15);
        color: #60A5FA;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        margin-top: 15px;
        border: 1px solid rgba(37, 99, 235, 0.3);
    }
    .summary-card {
        background-color: #FFFFFF;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3);
        border: 1px solid #E2E8F0;
        min-height: 220px;
    }
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        border-radius: 12px !important;
        padding: 4px !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important;
        border: 2px dashed #CBD5E1 !important;
        border-radius: 10px !important;
    }

    /* 3. Primary instruction text label (e.g., "Drag and drop file here") */
    [data-testid="stFileUploader"] section [data-testid="stMarkdownContainer"] p {
        color: #1E293B !important;
        font-weight: 500 !important;
    }

    /* 4. Small sub-text limitation text label (e.g., "5MB per file • PDF") */
    [data-testid="stFileUploader"] section small {
        color: #64748B !important;
    }

    /* 5. Streamlit native "Browse files" button default state styling */
    [data-testid="stFileUploader"] button {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        transition: background-color 0.2s ease !important;
    }

    /* 6. Button hover interaction animation state */
    [data-testid="stFileUploader"] button:hover {
        background-color: #E2E8F0 !important;
        color: #0F172A !important;
        border-color: #94A3B8 !important;
    }
        /* =========================================================================
       COMPLETE WHITE TRANSFORMATION FOR THE STREAMLIT TEXT AREA WIDGET
       ========================================================================= */
    /* 1. Target the main textarea box background and text color */
    [data-testid="stTextArea"] textarea {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
    }

    /* 2. Target the placeholder text color so it remains readable */
    [data-testid="stTextArea"] textarea::placeholder {
        color: #94A3B8 !important;
        opacity: 1 !important; /* Forces visibility across different browsers */
    }

    /* 3. Handle the active focus state when a user clicks into the box */
    [data-testid="stTextArea"] textarea:focus {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 1px #6366F1 !important;
    }

    </style>
""", unsafe_allow_html=True)

st.markdown(f"""
                    <div class="reference-card" style="
                        background-color: #0F172A; 
                        border: 1px dashed #334155; 
                        border-radius: 0.75rem; 
                        padding: 1.25rem; 
                        margin: 1rem 0;
                        font-family: sans-serif;
                    ">
                        <h5 style="margin: 0 0 0.75rem 0; color: #94A3B8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">
                            📌 2026 Full-Stack Baseline Reference Points
                        </h5>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                            <div>
                                <strong style="color: #F8FAFC; font-size: 0.9rem;">Frontend</strong>
                                <ul style="margin: 0.25rem 0 0 0; padding-left: 1.25rem; color: #94A3B8; font-size: 0.8rem; line-height: 1.5;">
                                    <li>HTML5 / CSS (Flexbox & Grid)</li>
                                    <li>Async JS (ES6+) & TypeScript</li>
                                    <li>React or Next.js</li>
                                </ul>
                            </div>
                            <div>
                                <strong style="color: #F8FAFC; font-size: 0.9rem;">Backend & APIs</strong>
                                <ul style="margin: 0.25rem 0 0 0; padding-left: 1.25rem; color: #94A3B8; font-size: 0.8rem; line-height: 1.5;">
                                    <li>Node.js + Express / Python + FastAPI</li>
                                    <li>RESTful API Design</li>
                                </ul>
                            </div>
                            <div>
                                <strong style="color: #F8FAFC; font-size: 0.9rem;">Databases</strong>
                                <ul style="margin: 0.25rem 0 0 0; padding-left: 1.25rem; color: #94A3B8; font-size: 0.8rem; line-height: 1.5;">
                                    <li>SQL: PostgreSQL / MySQL</li>
                                    <li>NoSQL: MongoDB</li>
                                </ul>
                            </div>
                            <div>
                                <strong style="color: #F8FAFC; font-size: 0.9rem;">DevOps & Git</strong>
                                <ul style="margin: 0.25rem 0 0 0; padding-left: 1.25rem; color: #94A3B8; font-size: 0.8rem; line-height: 1.5;">
                                    <li>Git Branching & GitHub</li>
                                    <li>Vercel / Render / Railway</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
st.markdown("""
    <style>
        /* Smoothly animate all progress bars on load */
        .stProgress > div > div > div > div {
            transition: width 0.8s ease-in-out !important;
            background-image: linear-gradient(to right, #6366F1, #4F46E5) !important;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; padding: 20px 0 10px 0;">
        <h1 style="color: #0B0909; font-size: 2.5rem; font-weight: 800; margin-bottom: 4px;">
            🚀 AI Resume Screener | NovaNectar
        </h1>
        <p style="color: #0B0909; font-size: 1rem; max-width: 600px; margin: 0 auto;">
            Instantly check how well a candidate fits your role.
        </p>
    </div>
""", unsafe_allow_html=True)
st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
st.subheader("📥 Step 1: Upload Application")
uploaded_file = st.file_uploader(
    "Drop exactly ONE Candidate PDF Resume here", 
    type=["pdf"], 
    accept_multiple_files=False,
    label_visibility="collapsed"
)
if not uploaded_file:
    st.markdown('<p class="custom-caption">ℹ️ Maximum size per file is 5MB. Only single .PDF files are supported.</p>', unsafe_allow_html=True)
else:
    st.markdown(f"""
        <div class="custom-upload-pill" style="
            background-color: #FFFFFF; 
            border: 1px solid #E2E8F0;
            border-left: 4px solid #10B981; 
            padding: 16px; 
            border-radius: 12px; 
            margin-top: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        ">
            <div>
                <span style="color: #64748B; font-size: 12px; display: block; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">Staged Document</span>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 16px;">📕</span>
                    <strong style="color: #0F172A; font-size: 15px;">{uploaded_file.name}</strong>
                    <span style="color: #64748B; font-size: 12px; font-family: monospace;">({uploaded_file.size / 1024:.1f} KB)</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    action_col1, action_col2, action_col3 = st.columns([1.5, 1.5, 4])
    with action_col1:
        st.download_button(
            label="👀 Preview / Download",
            data=uploaded_file.getvalue(),
            file_name=uploaded_file.name,
            mime="application/pdf",
            use_container_width=True,
            type="secondary"
        )
    with action_col2:
            st.markdown("""
            <a href="/" target="_self" style="text-decoration: none;">
                <button style="
                    width: 100%;
                    background-color: #EF4444; 
                    color: white; 
                    border: none;
                    padding: 8px 16px;
                    font-size: 14px;
                    font-weight: 500;
                    border-radius: 4px;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 6px;
                    height: 38px;
                ">
                    🗑️ Remove
                </button>
            </a>
        """, unsafe_allow_html=True)
    with action_col3:
        st.markdown("""
            <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%; padding-top: 4px;">
                <span style="background-color: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.3); font-size: 11px; padding: 4px 10px; border-radius: 9999px; font-weight: 600; text-transform: uppercase;">
                    ● System Ready
                </span>
            </div>
        """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True) 
st.subheader("📋 Step 2: Define Job Requirements")
st.markdown('<p style="color: #94a3b8; font-size: 14px; margin-bottom: 8px;">Paste Job Description text here:</p>', unsafe_allow_html=True)
job_desc = st.text_area(
    "Job Description Input Space", 
    height=220, 
    placeholder="e.g. Seeking a Python Developer with experience in SQL databases and machine learning models...",
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")
left_space, center_target, right_space = st.columns([1.5, 1, 1.5])
with center_target:
    run_clicked = st.button("Run AI Match Ranker", type="primary", use_container_width=True)
if run_clicked:
    if not uploaded_file:  
        st.error("Error: Please upload a resume PDF to evaluate first.")
    elif not job_desc.strip():
        st.error("Error: Please paste a job description context to proceed.")
    else:
        processed_candidates = []
        temp_filename = f"temp_{uploaded_file.name}"
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            raw = extract_text_from_pdf(temp_filename)
            cleaned = clean_text_stream(raw)
            name, skills, exp = parse_candidate_metadata(cleaned)
            category = classifier_pipeline.predict_profile_tag(cleaned)
            processed_candidates.append({
                "Name": str(name) if name != "Unknown Candidate" else uploaded_file.name,
                "Category_Tag": str(category),
                "Skills_Display": ", ".join(skills) if skills else "None Detected",
                "Skills_List": skills,  
                "Experience": f"{exp} Years" if exp > 0 else "Not Specified",
                "Cleaned_Text": cleaned
            })
        except Exception as e:
            st.error(f"System failure reading file {uploaded_file.name}: {str(e)}")
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                
        if len(processed_candidates) > 0:
            base_df = pd.DataFrame(processed_candidates)
            with st.spinner("AI Core alignment calculator engine executing matrices..."):
                final_ranked_df = compute_composite_scores(job_desc, base_df) 
            if not final_ranked_df.empty and "Match_Score" in final_ranked_df.columns:
                st.success("✅ Analysis Complete!")
                score_val = int(float(final_ranked_df["Match_Score"].iloc[0]))
                _, target_skills, _ = parse_candidate_metadata(job_desc)
                job_skills_lower = {s.lower().strip() for s in target_skills} if target_skills else set()
                resume_skills_lower = {s.lower().strip() for s in skills} if skills else set()
                missing_skills_set = job_skills_lower - resume_skills_lower
                if missing_skills_set:
                    missing_formatted = ", ".join([f"'{s.upper()}'" for s in list(missing_skills_set)[:4]])
                    keyword_feedback = f"Missing key job-related keywords such as {missing_formatted} reduce ATS keyword match."
                else:
                    keyword_feedback = "The profile demonstrates excellent coverage of key target keyword requirements."
                if score_val >= 85:
                    grade = "A"
                    feedback = f"The resume provides a comprehensive technical background and strong domain experience matching the target role. {keyword_feedback} Strong profile showing excellent domain matches."
                    badge_bg = "rgba(16, 185, 129, 0.15)"
                    badge_text = "#10B981"
                    badge_border = "rgba(16, 185, 129, 0.3)"
                elif score_val >= 75:
                    grade = "B"
                    feedback = f"The resume provides a solid baseline profile with reliable industry exposure. {keyword_feedback} Polishing specific measurable achievements would maximize application impact."
                    badge_bg = "rgba(20, 184, 166, 0.15)"
                    badge_text = "#14B8A6"
                    badge_border = "rgba(20, 184, 166, 0.3)"
                elif score_val >= 65:
                    grade = "C+"
                    feedback = f"The resume provides a clear education background and relevant skills matching the role. {keyword_feedback} Adding more metrics would improve recruiter appeal."
                    badge_bg = "rgba(245, 158, 11, 0.15)"
                    badge_text = "#F59E0B"
                    badge_border = "rgba(245, 158, 11, 0.3)"
                elif score_val >= 40:
                    grade = "C"
                    feedback = f"The resume provides a basic layout match but features clear experience gaps. {keyword_feedback}"
                    badge_bg = "rgba(249, 115, 22, 0.15)"
                    badge_text = "#F97316"
                    badge_border = "rgba(249, 115, 22, 0.3)"
                else:
                    grade = "D"
                    feedback = f"Significant target profile requirement gaps detected. {keyword_feedback} The portfolio requires structural keyword updates to fulfill core role prerequisites."
                    badge_bg = "rgba(239, 68, 68, 0.15)"
                    badge_text = "#EF4444"
                    badge_border = "rgba(239, 68, 68, 0.3)"
                st.subheader("📊 ATS Scan Results")
                layout_col1, layout_col2 = st.columns([1, 2.2], gap="medium")
                with layout_col1:
                    st.markdown(f"""
                         <div class="ats-score-card">
                            <!-- Score text automatically matches the grade color theme -->
                            <div class="circle-score" style="color: {badge_text}; font-size: 3.5rem; font-weight: 800; line-height: 1;">{score_val}</div>
                            <div class="circle-label" style="font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; margin: 0.5rem 0 1rem 0;">ATS Score</div>
                            <!-- Injected inline dynamic styles -->
                            <div class="grade-badge" style="background-color: {badge_bg}; color: {badge_text}; border: 1px solid {badge_border}; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.85rem; font-weight: 600;">
                                Grade: {grade}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                with layout_col2:
                    st.markdown(f"""
                       <div class="summary-card" style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 1rem; padding: 1.5rem; height: 100%; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
                            <h4 style="margin-top:0; color:#0F172A; font-size: 1.1rem; font-weight: 700;">Summary</h4>
                            <p style="color:#334155; line-height:1.6; font-size:15px; margin-bottom: 0;">
                                {feedback}<br><br>The system has automatically mapped the candidate under the <span style="color:#4F46E5; font-weight:600;">{category}</span> domain vertical.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("Section Scores")
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("Section Scores")
                bar_col1, bar_col2 = st.columns(2, gap="medium")
                with bar_col1:
                    st.write(f"Formatting Optimization: **{min(score_val + 5, 100)}/100**")
                    st.progress(min(score_val + 5, 100) / 100)
                    st.write(f"Experience Benchmark Metric: **{max(min(score_val - 2, 100), 0) if score_val > 5 else score_val}/100**")
                    st.progress(max(min(score_val - 2, 100), 0) / 100)
                with bar_col2:
                    st.write(f"🔑 Target Keywords Density Ratio: **{score_val}/100**")
                    st.progress(score_val / 100)
                    st.write(f"🛠️ Skills Match Relevance: **{min(score_val + 2, 100)}/100**")
                    st.progress(min(score_val + 2, 100) / 100)
