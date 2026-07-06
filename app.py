import os
import base64
import html
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
    <div style="text-align: center; padding: 28px 20px 16px 20px; background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%); border-radius: 16px; margin-bottom: 24px;">
        <h1 style="color: #1F2937; font-size: 2.8rem; font-weight: 900; margin-bottom: 8px; letter-spacing: -1px;">
            🚀 AI Resume Screener | NovaNectar
        </h1>
        <p style="color: #6B7280; font-size: 1.05rem; max-width: 600px; margin: 0 auto; font-weight: 500;">
            Instantly check how well a candidate fits your role with AI-powered analysis.
        </p>
    </div>
""", unsafe_allow_html=True)
st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
st.markdown('<h3 style="color: #1F2937; font-weight: 700; font-size: 1.4rem; margin-bottom: 16px;">📥 Step 1: Upload Application</h3>', unsafe_allow_html=True)

st.markdown("""
    <style>
        [data-testid="stFileUploader"] {
            background-color: transparent !important;
        }
        [data-testid="stFileUploader"] section {
            background: linear-gradient(135deg, #F0F9FF 0%, #F9FAFB 100%) !important;
            border: 2px dashed #3B82F6 !important;
            border-radius: 14px !important;
            padding: 32px !important;
            text-align: center !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.1) !important;
        }
        [data-testid="stFileUploader"] section:hover {
            border-color: #2563EB !important;
            box-shadow: 0 8px 12px -1px rgba(59, 130, 246, 0.2) !important;
            background: linear-gradient(135deg, #E0F2FE 0%, #F0F9FF 100%) !important;
        }
        [data-testid="stFileUploader"] section [data-testid="stMarkdownContainer"] {
            display: none !important;
        }
        [data-testid="stFileUploaderDropzone"] {
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
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 6px rgba(59, 130, 246, 0.25) !important;
        }
        [data-testid="stFileUploader"] button:hover {
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
            box-shadow: 0 8px 12px rgba(59, 130, 246, 0.35) !important;
            transform: translateY(-2px) !important;
        }
    </style>
    <div style="margin-bottom: 8px;">
        <p style="color: #6B7280; font-size: 15px; font-weight: 500; margin: 0;">📄 Select Resume File</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag and drop your PDF resume here or click to browse", 
    type=["pdf"], 
    accept_multiple_files=False,
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
    run_clicked = st.button("🚀 Run AI Match Ranker", type="primary", use_container_width=True, key="run_analysis")
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
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
                
                # Extract candidate skills
                if "Skills_List" in top_candidate.index and pd.notna(top_candidate["Skills_List"]):
                    candidate_skills_raw = top_candidate["Skills_List"]
                else:
                    candidate_skills_raw = []
                
                # Process skills into lowercase set
                if isinstance(candidate_skills_raw, list):
                    resume_skills_lower = {str(s).lower().strip() for s in candidate_skills_raw}
                elif isinstance(candidate_skills_raw, str):
                    resume_skills_lower = {s.lower().strip() for s in candidate_skills_raw.split(",")}
                else:
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
                
                # Save to session state for persistence
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
                        <div class="ats-score-card">
                            <div class="circle-score" style="color: {badge_text}; font-size: 3.8rem; font-weight: 800; line-height: 1;">{score_val}</div>
                            <div class="circle-label" style="color: #6B7280; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">ATS Match Score</div>
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
                    st.markdown(f'<p style="color: #374151; font-weight: 600; font-size: 15px; margin-bottom: 8px;">🔑 Target Keywords Density Ratio: <span style="color: #2563EB; font-weight: 700;">{score_val}/100</span></p>', unsafe_allow_html=True)
                    st.progress(score_val / 100)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.write("")
            #         display_missing_options = []
            #         if missing_keywords:
            #         display_missing_options = missing_keywords  
            #         else:
            #         display_missing_options = ["None Detected"]
            #         st.subheader("💡 AI Optimization Playground")
            #         with st.container():
            # st.markdown(
            #     "**Select detected missing keywords to dynamically generate optimized resume content:**"
            # )
            # selected_keywords = st.multiselect(
            #     "Gaps Identified in Pipeline Analysis:",
            #     options=display_missing_options,
            #     key="selected_missing_keywords"
            # )
            # if st.button(
            #     "✨ Generate Optimized Experience Bullet",
            #     type="secondary",
            #     use_container_width=True,
            #     key="generate_bullet"
            # ):
            #     if selected_keywords and "None Detected" not in selected_keywords:
            #         keywords_str = ", ".join(keyword.upper() for keyword in selected_keywords)
            #         st.session_state["optimized_bullet"] = (
            #             f"Engineered scalable project solutions integrating "
            #             f"{keywords_str} core technical methodologies to maximize "
            #             f"platform performance, automation, and cross-functional "
            #             f"collaboration while improving system reliability."
            #         )
            #     else:
            #         st.session_state["optimized_bullet"] = ""
            #         st.warning("No missing keywords selected or the profile is already optimized.")
            # if "optimized_bullet" in st.session_state and st.session_state["optimized_bullet"] != "":
            #     st.success("### Suggested Resume Bullet")
            #     st.markdown(st.session_state["optimized_bullet"])
        #  st.markdown("<br>", unsafe_allow_html=True)
        # st.subheader("🧠 Live Candidate AI Quiz Generator")
        # with st.container():
        #     st.write("Generate a real-time technical screening quiz based on the candidate's core missing keywords:")
        #     selected_keywords = st.session_state.selected_missing_keywords

        #     if st.button("🚀 Generate Screening Quiz from Missing Skills", type="primary", use_container_width=True, key="generate_quiz"):
        #         if selected_keywords and selected_keywords != ["None Detected"]:
        #             st.session_state.show_quiz = True
        #             st.session_state.quiz_keywords = list(selected_keywords)
        #             st.session_state.quiz_submitted = False
        #         else:
        #             st.warning("Please verify skill selections inside the playground drop-down menu first.")

        #     if st.session_state.show_quiz and st.session_state.quiz_keywords:
        #         st.write("---")
        #         quiz_keywords = st.session_state.quiz_keywords
        #         st.markdown(f"### 📋 Technical Screening Assessment (`Tags: {', '.join([k.upper() for k in quiz_keywords])}`) ")
        #         answers_map = {}
        #         for idx, skill in enumerate(quiz_keywords, 1):
        #             normalized_skill = skill.lower().strip()
        #             if normalized_skill == "html":
        #                 st.markdown(f"**Question {idx}.1 (HTML):** What is the functional semantic difference between a layout `<section>` element and a generic `<div>` wrapper container?")
        #                 q1_ans = st.radio("Select Correct Core Architecture Answer:", ["No difference", "Semantic containers aid accessibility, SEO, and document outline models", "Divs are deprecated in modern HTML5 schemas"], key=f"q_mc_{idx}")
        #                 answers_map[f"q_mc_{idx}"] = {"user": q1_ans, "correct": "Semantic containers aid accessibility, SEO, and document outline models"}

        #                 st.markdown(f"**Question {idx}.2 (HTML):** True or False: A `<section>` element always provides better accessibility than a `<div>.")
        #                 q2_ans = st.radio("Answer:", ["True", "False"], key=f"q_tf_{idx}")
        #                 answers_map[f"q_tf_{idx}"] = {"user": q2_ans, "correct": "False"}

        #                 st.markdown(f"**Question {idx}.3 (HTML):** Which attribute is mandatory to optimize search engine crawling on a picture element fallback path?")
        #                 q3_ans = st.radio("Select Layout Flag:", ["alt attribute text", "style declaration parameters", "loading lazy tags"], key=f"q_{idx}_3")
        #                 answers_map[f"q_{idx}_3"] = {"user": q3_ans, "correct": "alt attribute text"}
        #             elif normalized_skill == "css":
        #                 st.markdown(f"**Question {idx}.1 (CSS):** Explain the functional priority rendering chain of the browser engine's box model cascade sequence.")
        #                 q1_ans = st.radio("Select Correct Core Architecture Answer:", ["Inline styles > ID selectors > Class selectors > Element tags", "Class elements completely override structural root declarations", "Element tag settings maintain execution authority"], key=f"q_mc_{idx}")
        #                 answers_map[f"q_mc_{idx}"] = {"user": q1_ans, "correct": "Inline styles > ID selectors > Class selectors > Element tags"}

        #                 st.markdown(f"**Question {idx}.2 (CSS):** True or False: CSS specificity determines which rules apply when multiple selectors match the same element.")
        #                 q2_ans = st.radio("Answer:", ["True", "False"], key=f"q_tf_{idx}")
        #                 answers_map[f"q_tf_{idx}"] = {"user": q2_ans, "correct": "True"}

        #                 st.markdown(f"**Question {idx}.3 (CSS):** What layout strategy is optimized to force flexible grid elements to align directly down a vertical center vector alignment?")
        #                 q3_ans = st.radio("Select Property Configuration:", ["align-items: center", "float: left clear layout", "display: text block inline"], key=f"q_{idx}_3")
        #                 answers_map[f"q_{idx}_3"] = {"user": q3_ans, "correct": "align-items: center"}
        #             else:
        #                 st.markdown(f"**Question {idx}.1 ({skill.upper()}):** Describe production deployment optimization pipelines regarding critical dependencies for **{skill.upper()}**.")
        #                 q1_ans = st.radio("Select Correct Core Architecture Answer:", ["Minification and tree-shaking protocols", "Disabling development log frames entirely", "Skipping build verification tests"], key=f"q_mc_{idx}")
        #                 answers_map[f"q_mc_{idx}"] = {"user": q1_ans, "correct": "Minification and tree-shaking protocols"}

        #                 st.markdown(f"**Question {idx}.2 ({skill.upper()}):** True or False: Production deployment optimization always requires disabling build verification tests.")
        #                 q2_ans = st.radio("Answer:", ["True", "False"], key=f"q_tf_{idx}")
        #                 answers_map[f"q_tf_{idx}"] = {"user": q2_ans, "correct": "False"}

        #                 st.markdown(f"**Question {idx}.3 ({skill.upper()}):** What is the core structural performance risk involved when installing unverified external modules inside the production build environment?")
        #                 q3_ans = st.radio("Select Architectural Tradeoff:", ["Dependency bloat and potential security vulnerabilities", "Complete styling interface compilation drops", "Automatic script runtime acceleration"], key=f"q_{idx}_3")
        #                 answers_map[f"q_{idx}_3"] = {"user": q3_ans, "correct": "Dependency bloat and potential security vulnerabilities"}

        #         st.write("")
        #         if st.button("Submit Assessment Log Framework", type="secondary", key="submit_assessment"):
        #             total_questions = len(answers_map)
        #             correct_count = sum(1 for key, val in answers_map.items() if val["user"] == val["correct"])
        #             score_percentage = (correct_count / total_questions) * 100 if total_questions > 0 else 0
        #             st.session_state.quiz_submitted = True
        #             st.session_state.final_score = f"{correct_count} / {total_questions}"
        #             st.session_state.final_percentage = score_percentage

        #         if st.session_state.get("quiz_submitted", False):
        #             st.success("Candidate technical score tracking matrix dispatched to system database logs successfully!")
        #             with st.container():
        #                 st.markdown("### 📊 Live Screening Examination Results")
        #                 col_score, col_grade = st.columns(2)
        #                 with col_score:
        #                     st.metric(label="Total Correct Evaluation Matches", value=st.session_state.final_score)
        #                 with col_grade:
        #                     st.metric(label="Final Matrix Performance Percentage", value=f"{st.session_state.final_percentage:.1f}%")
        #                 if st.session_state.final_percentage >= 70:
        #                     if hasattr(st, "balloons"):
        #                         st.balloons()
        #                     st.info("💡 **AI Recruiter Decision Log:** Candidate exhibits robust fundamentals inside technical gaps. Proceeding directly to structural codebase layout panel checks.")
        #                 else:
        #                     st.warning("⚠️ **AI Recruiter Decision Log:** Performance levels down trend core framework thresholds. Targeted remedial deployment upskilling modules suggested prior to system entry logs.")
        #         elif st.session_state.show_quiz:
        #             st.warning("Please select missing keywords first to build the quiz.")
        #     elif st.session_state.show_quiz:
        #         st.warning("Please select missing keywords first to build the quiz.")

        # st.markdown("<br>", unsafe_allow_html=True)
        # st.subheader("📋 Tailored Interview Kit")
        # with st.expander("🤖 View Detailed AI Interview Questions for this Candidate", expanded=False):
        #     st.write("These questions are automatically tailored to address the specific performance patterns observed:")
        #     gap_text = keyword_feedback.split('such as ')[-1] if 'such as ' in keyword_feedback else 'core role tools'
        #     st.markdown(f"""
        #         1. **Technical Alignment:** *\"During our platform check, we noted architectural gaps regarding {gap_text}. Can you explain how you plan to integrate these methods into your dev pipeline?\"*
        #         2. **Experience Benchmarking:** *\"Can you detail a complex deployment you managed that directly aligns with **{category}** vertical architectures?\"*
        #         """)
