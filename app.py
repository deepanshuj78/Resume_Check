# ==========================================
# 3. HIGH-CONTRAST IMPROVED TESTING PANEL
# ==========================================
with st.sidebar:
    # Stylized clean header using injected CSS class
    st.markdown('<div class="dev-header">🛠️ Developer Testing Panel</div>', unsafe_allow_html=True)
    
    # High-contrast checkbox trigger
    st.session_state["ats_run_completed"] = st.checkbox(
        "Simulate: ATS Scan Completed", 
        value=st.session_state["ats_run_completed"]
    )
    
    st.markdown("---") # Visual separation line
    
    # Styled state clear mechanism button
    if st.button("🔄 Reset Whole App State", use_container_width=True):
        st.session_state["ats_run_completed"] = False
        st.session_state["quiz_generation_triggered"] = False
        st.session_state["generated_quiz_data"] = None
        st.session_state["used_question_ids"] = set()
        st.rerun()


# ==========================================
# 4. CONTENT LOGIC RUNNERS
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
if st.session_state.get("ats_run_completed", False) and not st.session_state.get("quiz_generation_triggered", False):
    st.info("💡 Simulation Active: The screening triggers are now live below.")
    if st.button("🧬 Generate Skill-Gap screening Quiz", type="primary", use_container_width=True):
        with st.spinner("Analyzing resume against job description..."):
            st.session_state["missing_skills_set"] = {"GITHUB", "HTML", "CSS", "MONGODB", "AI"}
            st.session_state["quiz_generation_triggered"] = True 
        st.success("ATS Score evaluation complete!")
        st.rerun()

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
