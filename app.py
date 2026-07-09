import os
import base64
import html
import random
import pandas as pd
import streamlit as st
from parser_engine import extract_text_from_pdf, clean_text_stream, parse_candidate_metadata
from ML_Classifier import classifier_pipeline
from Scoring_Matrix import compute_composite_scores
st.set_page_config(page_title="AI Resume Screener", page_icon="🚀", layout="wide")
st.markdown("""
    <style>
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
    [data-testid="stFileUploader"] section [data-testid="stMarkdownContainer"] p {
        color: #1E293B !important;
        font-weight: 500 !important;
    }
    [data-testid="stFileUploader"] section small {
        color: #64748B !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        transition: background-color 0.2s ease !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: #E2E8F0 !important;
        color: #0F172A !important;
        border-color: #94A3B8 !important;
    }
    [data-testid="stTextArea"] textarea {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
    }
    [data-testid="stTextArea"] textarea::placeholder {
        color: #94A3B8 !important;
        opacity: 1 !important; /* Forces visibility across different browsers */
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 1px #6366F1 !important;
    }

    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
        .reference-card-upgraded {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            border: 2px solid rgba(59, 130, 246, 0.3);
            border-radius: 16px;
            padding: 28px;
            margin: 20px 0;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }
        .reference-card-upgraded:hover {
            border-color: rgba(59, 130, 246, 0.6);
            box-shadow: 0 15px 35px -5px rgba(59, 130, 246, 0.2);
            transform: translateY(-2px);
        }
        .ref-title {
            color: #E0F2FE;
            font-size: 1.1rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            margin: 0 0 24px 0;
        }
        .ref-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
        }
        .ref-category {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 12px;
            padding: 18px;
            transition: all 0.3s ease;
        }
        .ref-category:hover {
            border-color: rgba(59, 130, 246, 0.5);
            background: rgba(30, 41, 59, 0.9);
            transform: translateY(-4px);
            box-shadow: 0 8px 16px rgba(59, 130, 246, 0.15);
        }
        .ref-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 12px;
            padding: 6px 12px;
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(37, 99, 235, 0.2) 100%);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 8px;
            width: fit-content;
        }
        .ref-category-title {
            color: #3B82F6;
            font-size: 0.95rem;
            font-weight: 700;
        }
        .ref-items {
            margin: 12px 0 0 0;
            padding-left: 0;
            list-style: none;
        }
        .ref-items li {
            color: #CBD5E1;
            font-size: 0.9rem;
            line-height: 1.8;
            margin-bottom: 8px;
            padding-left: 20px;
            position: relative;
            transition: all 0.2s ease;
        }
        .ref-items li:before {
            content: "▸";
            position: absolute;
            left: 4px;
            color: #60A5FA;
        }
        .ref-items li:hover {
            color: #E0F2FE;
            padding-left: 24px;
        }
    </style>
    <div class="reference-card-upgraded">
        <div class="ref-title">🎯 2026 Full-Stack Baseline Reference Points</div>
        <div class="ref-grid">
            <div class="ref-category">
                <div class="ref-badge"><span style="font-size: 18px;">🎨</span><span class="ref-category-title">Frontend</span></div>
                <ul class="ref-items">
                    <li>HTML5 / CSS (Flexbox & Grid)</li>
                    <li>Async JS (ES6+) & TypeScript</li>
                    <li>React or Next.js</li>
                </ul>
            </div>
            <div class="ref-category">
                <div class="ref-badge"><span style="font-size: 18px;">⚙️</span><span class="ref-category-title">Backend & APIs</span></div>
                <ul class="ref-items">
                    <li>Node.js + Express / Python + FastAPI</li>
                    <li>RESTful API Design</li>
                    <li>Microservices Architecture</li>
                </ul>
            </div>
            <div class="ref-category">
                <div class="ref-badge"><span style="font-size: 18px;">🗄️</span><span class="ref-category-title">Databases</span></div>
                <ul class="ref-items">
                    <li>SQL: PostgreSQL / MySQL</li>
                    <li>NoSQL: MongoDB</li>
                    <li>Query Optimization</li>
                </ul>
            </div>
            <div class="ref-category">
                <div class="ref-badge"><span style="font-size: 18px;">🚀</span><span class="ref-category-title">DevOps & Git</span></div>
                <ul class="ref-items">
                    <li>Git Branching & GitHub</li>
                    <li>Docker & CI/CD</li>
                    <li>Vercel / Render / Railway</li>
                </ul>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
        .stProgress > div > div > div > div {
            transition: width 0.8s ease-in-out !important;
            background-image: linear-gradient(to right, #6366F1, #4F46E5) !important;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; padding: 32px 20px 24px 20px; background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%); border-radius: 16px; margin-bottom: 24px border: 1px solid #E5E7EB;">
        <h1 style="color: #1F2937; font-size: 2.8rem; font-weight: 900; margin-bottom: 8px; letter-spacing: -1px;">
            🚀 AI Resume Screener | NovaNectar
        </h1>
        <p style="color: #4B5563; font-size: 1.05rem; max-width: 600px; margin: 0 auto; font-weight: 500; text-align:center;">
            Instantly check how well a candidate fits your role with AI-powered analysis.
        </p>
    </div>
""", unsafe_allow_html=True)
st.markdown(""" <br>
            <div style="margin-top: 24px 0 16px 0;"> <h3 style="color: #0F172A; font-weight: 700; font-size: 1.4rem; margin: 0 0 6px 0; display: flex; align-items: center; gap: 8px;"> 📥 Step 1: Upload Application</h3> <p style="color: #6B7280; font-size: 0.875rem; margin: 0;"> Upload the candidate's resume file below (PDF format supported up to 5MB).
        </p> </div>""", unsafe_allow_html=True)
st.markdown("""
    <style>
        [data-testid="stFileUploader"] {
            background-color: transparent !important;
            margin-bottom: 12px !important;
        }
        [data-testid="stFileUploader"] section {
            background-color: #F8FAFC !important; 
            border: 2px dashed #CBD5E1 !important;
            border-radius: 12px !important;
            padding: 24px 16px !important;
            text-align: center !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        }
        [data-testid="stFileUploader"] section:hover {
            border-color: #3B82F6 !important; /* Modern vibrant blue on action focus */
            background-color: #F0F9FF !important; /* Soft sky blue tint on active state */
            box-shadow: 0 4px 12px -2px rgba(59, 130, 246, 0.12) !important;
            transform: translateY(-1px);
        }
        [data-testid="stFileUploader"] section [data-testid="stMarkdownContainer"] p {
            display: none !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 12px !important;
            text-align: center !important;
        }
        [data-testid="stFileUploader"] button {
            background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
            color: white !important;
            border: none !important;
            padding: 10px 24px !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            font-size: 14px !important;
            transition: all 0.25s ease !important;
            box-shadow: 0 4px 6px rgba(59, 130, 246, 0.2) !important;
        }
        [data-testid="stFileUploader"] button:hover {
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
            box-shadow: 0 6px 12px rgba(59, 130, 246, 0.3) !important;
            transform: translateY(-1px) !important; /* Slight lift is cleaner than 2px */
        }
    </style>
    <div style="margin: 20px 0 10px 0;">
        <p style="color: #0F172A; font-size: 15px; font-weight: 700; margin: 0; display: flex; align-items: center; gap: 6px;"> <br> 📄 Select Resume File</p>
    </div>
""", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drag and drop your PDF resume here or click to browse", 
    type=["pdf"], 
    accept_multiple_files=False,
    label_visibility="collapsed"
)
st.markdown("""
    <div style="margin-top: 12px; padding: 12px 16px; background-color: #F3F4F6; border-radius: 8px; border-left: 4px solid #10B981;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <span style="font-size: 18px;">ℹ️</span>
            <span style="color: #374151; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">File Requirements</span>
        </div>
        <p style="color: #6B7280; font-size: 13px; margin: 0; line-height: 1.6;">
            ✓ Format: PDF only<br>
            ✓ Maximum size: 5MB<br>
            ✓ Single file upload
        </p>
    </div>
""", unsafe_allow_html=True)

if not uploaded_file:
    st.markdown("""
        <div style="text-align: center; padding: 24px; margin-top: 16px;">
            <div style="font-size: 48px; margin-bottom: 12px; opacity: 0.6;">📄</div>
            <p style="color: #9CA3AF; font-size: 14px; font-weight: 500;">Waiting for file upload...</p>
        </div>
    """, unsafe_allow_html=True)
else:
    file_bytes = uploaded_file.getvalue()
    encoded_pdf = base64.b64encode(file_bytes).decode("utf-8")
    safe_file_name = html.escape(uploaded_file.name)
    st.markdown(f"""
        <style>
            .staged-card {{
                background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
                border: 2px solid #E5E7EB;
                border-left: 5px solid #3B82F6;
                padding: 24px;
                border-radius: 14px;
                margin-top: 16px;
                box-shadow: 0 8px 16px -2px rgba(0, 0, 0, 0.08);
                transition: all 0.3s ease;
            }}
            .staged-card:hover {{
                box-shadow: 0 12px 24px -2px rgba(0, 0, 0, 0.12);
                border-left-color: #0EA5E9;
            }}
            .file-header {{
                color: #1F2937;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.8px;
                font-weight: 700;
                margin-bottom: 12px;
                display: flex;
                align-items: center;
                gap: 6px;
            }}
            .file-info {{
                display: flex;
                align-items: center;
                gap: 12px;
                margin-bottom: 16px;
            }}
            .file-icon {{
                font-size: 28px;
                display: flex;
                align-items: center;
            }}
            .file-details {{
                flex: 1;
            }}
            .file-name {{
                color: #1F2937;
                font-size: 16px;
                font-weight: 700;
                margin-bottom: 4px;
                word-break: break-word;
            }}
            .file-size {{
                color: #6B7280;
                font-size: 13px;
                font-family: 'Monaco', 'Menlo', monospace;
            }}
            .action-buttons {{
                display: flex;
                gap: 10px;
                align-items: center;
                justify-content: center;
                margin-top: 20px;
            }}
            .btn-primary {{
                flex: 1;
                background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
                color: white !important;
                border: none;
                padding: 10px 16px;
                font-size: 14px;
                font-weight: 600;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 6px;
                height: 42px;
                box-shadow: 0 4px 6px rgba(59, 130, 246, 0.25);
                text-decoration: none !important;
            }}
            .btn-primary:link,
            .btn-primary:visited,
            .btn-primary:active {{
                color: white !important;
                text-decoration: none !important;
            }}
            .btn-primary:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 12px rgba(59, 130, 246, 0.35);
                background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
                color: white !important;
                text-decoration: none !important;
            }}
            .btn-danger {{
                flex: 1;
                background: #FFFFFF;
                color: #DC2626 !important;
                border: 1px solid #FCA5A5;
                padding: 10px 16px;
                font-size: 14px;
                font-weight: 700;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 6px;
                height: 42px;
                box-shadow: 0 4px 6px rgba(15, 23, 42, 0.08);
                text-decoration: none !important;
            }}
            .btn-danger:link,
            .btn-danger:visited,
            .btn-danger:active {{
                color: #DC2626 !important;
                text-decoration: none !important;
            }}
            .btn-danger:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 12px rgba(220, 38, 38, 0.16);
                background: #FEF2F2;
                color: #B91C1C !important;
                border-color: #F87171;
                text-decoration: none !important;
            }}
            @media (max-width: 640px) {{
                .file-info {{
                    align-items: flex-start;
                    flex-wrap: wrap;
                }}
                .status-badge {{
                    margin-left: 40px;
                }}
                .action-buttons {{
                    flex-direction: column;
                }}
                .btn-primary,
                .btn-danger {{
                    width: 100%;
                    flex: none;
                }}
            }}
            .status-badge {{
                background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
                color: #059669;
                border: 1.5px solid rgba(16, 185, 129, 0.4);
                font-size: 12px;
                padding: 6px 12px;
                border-radius: 20px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                display: flex;
                align-items: center;
                gap: 6px;
                white-space: nowrap;
                box-shadow: 0 2px 4px rgba(16, 185, 129, 0.1);
            }}
            .status-dot {{
                width: 8px;
                height: 8px;
                background-color: #10B981;
                border-radius: 50%;
                animation: pulse 2s infinite;
            }}
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
            }}
        </style>
        <div class="staged-card">
            <div class="file-header">
                ✓ Staged Document
            </div>
            <div class="file-info">
                <div class="file-icon">📄</div>
                <div class="file-details">
                    <div class="file-name">{safe_file_name}</div>
                    <div class="file-size">{uploaded_file.size / 1024:.1f} KB</div>
                </div>
                <div class="status-badge">
                    <div class="status-dot"></div>
                    Ready
                </div>
            </div>
            <div class="action-buttons">
                <a class="btn-primary" href="data:application/pdf;base64,{encoded_pdf}" download="{safe_file_name}" target="_blank" rel="noopener noreferrer">
                    👁️ Preview / Download
                </a>
                <a class="btn-danger" href="/" target="_self">
                    🗑️ Remove
                </a>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True) 
st.markdown('<h3 style="color: #1F2937; font-weight: 700; font-size: 1.4rem; margin-bottom: 12px;">📋 Step 2: Define Job Requirements</h3>', unsafe_allow_html=True)
st.markdown("""
    <style>
        .job-desc-container {
            background: linear-gradient(135deg, #F9FAFB 0%, #FFFFFF 100%);
            padding: 2px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .job-desc-label {
            color: #374151;
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 8px;
            display: block;
        }
    </style>
    <div class="job-desc-container">
""", unsafe_allow_html=True)
st.markdown('<label class="job-desc-label">Paste Job Description text here:</label>', unsafe_allow_html=True)
job_desc = st.text_area(
    "Job Description Input Space", 
    height=200, 
    placeholder="e.g. Seeking a Python Developer with 5+ years experience in SQL databases, machine learning models, and cloud architecture...",
    label_visibility="collapsed"
)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("""
    <style>
        .button-container {
            display: flex;
            justify-content: center;
            margin: 24px 0;
            gap: 12px;
        }
        .run-button-wrapper {
            flex: 0 1 300px;
        }
    </style>
    <div class="button-container">
""", unsafe_allow_html=True)
left_space, center_target, right_space = st.columns([1.5, 1.2, 1.5])
with center_target:
    st.markdown('<div class="run-button-wrapper">', unsafe_allow_html=True)
    run_clicked = st.button("🚀 Run AI (Calculate ATS Scores)", type="primary", use_container_width=True, key="run_analysis")
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.ats_run_completed = False

    st.session_state.missing_skills_set = set()
    st.session_state.keyword_feedback = ""
    st.session_state.category = ""
    st.session_state.score_val = 0
    st.session_state.selected_missing_keywords = []
    st.session_state.show_quiz = False
    st.session_state.quiz_keywords = []
    st.session_state.quiz_submitted = False
    st.session_state.optimized_bullet = ""
    st.session_state.feedback = ""
    st.session_state.grade = ""
    st.session_state.badge_bg = ""
    st.session_state.badge_text = ""
    st.session_state.badge_border = ""

# ---- Quiz generation state (required by the MCQ selector) ----
if "used_question_ids" not in st.session_state:
    st.session_state.used_question_ids = set()
if "generated_quiz_data" not in st.session_state:
    st.session_state.generated_quiz_data = None
if run_clicked or st.session_state.analysis_done:
    recompute = run_clicked
    if not recompute:
        missing_skills_set = st.session_state.missing_skills_set
        keyword_feedback = st.session_state.keyword_feedback
        category = st.session_state.category
        score_val = st.session_state.score_val
        feedback = st.session_state.feedback
        grade = st.session_state.grade
        badge_bg = st.session_state.badge_bg
        badge_text = st.session_state.badge_text
        badge_border = st.session_state.badge_border
    if recompute:
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
                
                # Extract raw score and convert to 0-100 scale
                raw_score = float(final_ranked_df["Match_Score"].to_numpy()[0])
                # Aggressive scaling formula for better score distribution
                # raw_score is already boosted (0-1), convert to 0-100 with additional multiplier
                score_val = max(0, min(100, int(round((raw_score * 100) * 1.35 + 12))))
                
                top_candidate = final_ranked_df.iloc[0]
                _, target_skills, _ = parse_candidate_metadata(job_desc)
                job_skills_lower = {s.lower().strip() for s in target_skills} if target_skills else set()
                
                # Extract + normalize candidate skills safely (handles None/NaN/stringified lists/comma strings)
                candidate_skills_raw = []
                if "Skills_List" in top_candidate.index:
                    val = top_candidate["Skills_List"]
                    # Avoid pd.notna crash for nested/unexpected types; treat missing/NaN as empty.
                    try:
                        if pd.notna(val):
                            candidate_skills_raw = val
                    except Exception:
                        candidate_skills_raw = []

                def _normalize_skills_to_set(raw):
                    if raw is None:
                        return set()
                    if isinstance(raw, float) and pd.isna(raw):
                        return set()
                    if isinstance(raw, list):
                        return {str(s).lower().strip() for s in raw if str(s).strip()}
                    if isinstance(raw, str):
                        s = raw.strip()
                        if not s:
                            return set()
                        # If it looks like a python-list string, try to parse.
                        if s.startswith("[") and s.endswith("]"):
                            import ast
                            try:
                                parsed = ast.literal_eval(s)
                                if isinstance(parsed, list):
                                    return {str(x).lower().strip() for x in parsed if str(x).strip()}
                            except Exception:
                                pass
                        # Otherwise treat as comma-separated string.
                        return {part.strip().lower() for part in s.split(",") if part.strip()}
                    # Unknown type
                    return set()

                resume_skills_lower = _normalize_skills_to_set(candidate_skills_raw)
                
                # Defensive: ensure resume_skills_lower is always a set
                if not isinstance(resume_skills_lower, set):
                    resume_skills_lower = set()

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
                st.session_state.analysis_done = True
                st.session_state.score_val = score_val
                st.session_state.grade = grade
                st.session_state.feedback = feedback
                st.session_state.category = category
                st.session_state.badge_bg = badge_bg
                st.session_state.badge_text = badge_text
                st.session_state.badge_border = badge_border
                st.session_state.keyword_feedback = keyword_feedback
                st.session_state.missing_skills_set = missing_skills_set
                st.markdown(f'<h2 style="color: #1F2937; font-weight: 700; margin-bottom: 16px;">📊 Analysis Result: <span style="color: #2563EB;">{score_val}</span> | Grade: <span style="color: {badge_text}; font-weight: 800;">{grade}</span></h2>', unsafe_allow_html=True)
                st.info(feedback)
                st.markdown('<h3 style="color: #1F2937; font-weight: 700; font-size: 1.3rem; margin-top: 24px; margin-bottom: 16px;">📊 ATS Scan Results</h3>', unsafe_allow_html=True)
                layout_col1, layout_col2 = st.columns([1, 2.2], gap="medium")
                with layout_col1:
                    st.markdown(f"""
                        <div class="ats-score-card" style="height: 100%; min-height: 220px; display: flex; flex-direction: column; justify-content: center;">
                            <div class="circle-score" style="color: {badge_text}; font-size: 3.8rem; font-weight: 800; line-height: 1;">{score_val}</div>
                            <div class="circle-label" style="color: #6B7280; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-top: 4px;">ATS Match Score</div>
                            <div style="background-color: {badge_bg}; color: {badge_text}; border: 1px solid {badge_border}; padding: 6px 16px; border-radius: 9999px; font-weight: 600; font-size: 13px; display: inline-block; margin-top: 12px;">
                                Grade: {grade}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                with layout_col2:
                    st.markdown(f"""
                        <div class="summary-card">
                            <h4 style="margin-top:0; color:#1F2937; font-size: 1.3rem; font-weight: 700; letter-spacing: -0.3px;">Executive Summary</h4>
                            <p style="color:#374151; line-height:1.8; font-size:16px; margin-bottom: 0; margin-top: 12px;">
                                {feedback}<br><br>The system has automatically mapped the candidate under the <span style="color:#2563EB; font-weight:600;">{category}</span> domain vertical.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<h3 style="color: #1F2937; font-weight: 700; font-size: 1.3rem; margin-bottom: 16px;">📈 Section Scores</h3>', unsafe_allow_html=True)
                bar_col1, bar_col2 = st.columns(2, gap="large")
                fmt_score = min(score_val + 5, 100)
                exp_score = max(min(score_val - 2, 100), 0) if score_val > 5 else score_val
                with bar_col1:
                    st.markdown(f'<p style="color: #374151; font-weight: 600; font-size: 15px; margin-bottom: 8px;">Formatting Optimization: <span style="color: #2563EB; font-weight: 700;">{fmt_score}/100</span></p>', unsafe_allow_html=True)
                    st.progress(fmt_score / 100)
                    st.markdown(f'<p style="color: #374151; font-weight: 600; font-size: 15px; margin-bottom: 8px;">Experience Benchmark Metric: <span style="color: #2563EB; font-weight: 700;">{exp_score}/100</span></p>', unsafe_allow_html=True)
                    st.progress(exp_score / 100)
                with bar_col2:
                    st.markdown(
                        f'<p style="color: #374151; font-weight: 600; font-size: 15px; margin-bottom: 8px;">'
                        f'🔑 Target Keywords Density Ratio: '
                        f'<span style="color: #2563EB; font-weight: 700;">{score_val}/100</span></p>',
                        unsafe_allow_html=True,
                    )
                    st.progress(score_val / 100)
                    import streamlit as st
import random

# ==========================================
# 1. INITIALIZE ALL STATES (Fixes the invisible button)
# ==========================================
if "ats_run_completed" not in st.session_state:
    st.session_state["ats_run_completed"] = False

if "quiz_generation_triggered" not in st.session_state:
    st.session_state["quiz_generation_triggered"] = False

if "used_question_ids" not in st.session_state:
    st.session_state["used_question_ids"] = set()

if "generated_quiz_data" not in st.session_state:
    st.session_state["generated_quiz_data"] = None


# ==========================================
# TEST SIMULATOR (Remove this section in production)
# ==========================================
st.sidebar.markdown("### 🛠️ Developer Testing Panel")
st.session_state["ats_run_completed"] = st.sidebar.checkbox(
    "Simulate: ATS Scan Completed", 
    value=st.session_state["ats_run_completed"]
)

if st.sidebar.button("Reset Whole App State"):
    st.session_state["ats_run_completed"] = False
    st.session_state["quiz_generation_triggered"] = False
    st.session_state["generated_quiz_data"] = None
    st.session_state["used_question_ids"] = set()
    st.rerun()
# ==========================================


# Mock Data Bank for execution verification
MCQ_BANK = {
    "GITHUB": [{"id": 1, "question": "What command initializes a git repo?", "options": ["git init", "git start"], "answer": "git init", "explanation": "Initializes a repository."}],
    "HTML": [{"id": 2, "question": "What does HTML stand for?", "options": ["HyperText Markup Language", "HighText Machine"], "answer": "HyperText Markup Language", "explanation": "Standard markup language."}],
    "CSS": [{"id": 3, "question": "Which property changes text color?", "options": ["color", "text-color"], "answer": "color", "explanation": "Changes text element foreground colors."}],
    "MONGODB": [{"id": 4, "question": "MongoDB is what type of database?", "options": ["NoSQL", "SQL"], "answer": "NoSQL", "explanation": "Document store database."}],
    "AI": [{"id": 5, "question": "What does LLM stand for?", "options": ["Large Language Model", "Local Machine"], "answer": "Large Language Model", "explanation": "AI text model architectures."}]
}


# --- STAGE 1: ATS Evaluation Complete Trigger ---
# This button will ONLY show up if st.session_state["ats_run_completed"] is True
if st.session_state.get("ats_run_completed", False) and not st.session_state.get("quiz_generation_triggered", False):
    if st.button("🧬 Generate Skill-Gap screening Quiz", type="primary", use_container_width=True):
        with st.spinner("Analyzing resume against job description..."):
            st.session_state["missing_skills_set"] = {"GITHUB", "HTML", "CSS", "MONGODB", "AI"}
            st.session_state["quiz_generation_triggered"] = True 
        st.success("ATS Score evaluation complete!")
        st.rerun()

# If the system is waiting for the ATS process to finish, show a status indicator
elif not st.session_state.get("ats_run_completed", False):
    st.warning("⏳ Waiting for ATS Run to complete. (Check the box in the sidebar to simulate this!)")


# --- STAGE 2: Core Algorithmic Logic Loop ---
if st.session_state.get("quiz_generation_triggered", False) and st.session_state.get("generated_quiz_data") is None:
    quiz_questions = []
    topics_found = ["GITHUB", "HTML", "CSS", "MONGODB", "AI"]
    MCQ_BANK_LOCAL = globals().get("MCQ_BANK", MCQ_BANK)

    for topic in topics_found:
        if topic not in MCQ_BANK_LOCAL:
            continue
        topic_questions = MCQ_BANK_LOCAL.get(topic, []) or []
        if not topic_questions:
            continue
        available_questions = [q for q in topic_questions if q.get("id") not in st.session_state["used_question_ids"]]
        
        if not available_questions:
            st.session_state["used_question_ids"] = set(q.get("id") for q in topic_questions if q.get("id") is not None)
            available_questions = topic_questions
        if not available_questions:
            continue

        selected_q = random.choice(available_questions)
        qid = selected_q.get("id")
        if qid is not None:
            st.session_state["used_question_ids"].add(qid)

        shuffled_options = list(selected_q.get("options", []))
        random.shuffle(shuffled_options)

        quiz_questions.append({
            "topic": topic,
            "question": selected_q.get("question", ""),
            "options": shuffled_options,
            "correct": selected_q.get("answer", ""),
            "explanation": selected_q.get("explanation", ""),
        })

    if quiz_questions:
        st.session_state["generated_quiz_data"] = {
            "keywords": ", ".join(topics_found),
            "questions": quiz_questions
        }
        st.rerun()


# --- STAGE 3: UI Renderer Loop ---
if st.session_state.get("generated_quiz_data"):
    quiz = st.session_state["generated_quiz_data"]
    st.markdown("### 🧠 Live Candidate AI Quiz Generator")
    with st.container(border=True):
        st.markdown(f"### 📋 Technical Screening Quiz (`{quiz['keywords']}`)")
        st.caption("Instruct the candidate to select their answers below. Results will update dynamically.")
        st.divider()
        for idx, item in enumerate(quiz["questions"], start=1):
            st.markdown(f"**{idx}. [{item['topic']}]** {item['question']}")
            selected_option = st.radio(
                label=f"Choose an option for question {idx}",
                options=item["options"],
                index=None,
                key=f"mcq_{idx}",
                label_visibility="collapsed"
            )
            if selected_option:
                if selected_option == item["correct"]:
                    st.success("Correct Answer")
                else:
                    st.error("Incorrect Answer")
                with st.expander("💡 View Explanation Guide"):
                    st.markdown(f"**Correct Option:** `{item['correct']}`")
                    st.markdown(f"**Why:** {item['explanation']}")                    
            st.markdown("<br>", unsafe_allow_html=True)
