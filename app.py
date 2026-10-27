
import io
import streamlit as st
from pypdf import PdfReader
from docx import Document
from main import analyze_career_gap, groq_client


# --------------------------------------------------
# 1. PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="AI Career Gap Analyzer",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 AI Career Gap Analyzer")
st.write(
    "Upload your resume, choose your target role, "
    "and discover your career skill gaps."
)

st.divider()


# --------------------------------------------------
# 2. USER INPUTS
# --------------------------------------------------

target_role = st.text_input(
    "Which job role are you targeting?",
    placeholder="e.g. AI Engineer, Java Developer"
)

resume_file = st.file_uploader(
    "Upload your resume",
    type=[
        "txt", "py", "java", "js",
        "csv", "md", "pdf", "docx"
    ]
)


# --------------------------------------------------
# 3. ANALYZE BUTTON
# --------------------------------------------------

if st.button("Analyze Career Gap", type="primary"):

    if not target_role.strip():
        st.warning("Please enter your target job role.")

    elif resume_file is None:
        st.warning("Please upload your resume.")

    else:
        try:
            file_name = resume_file.name.lower()

            # A. EXTRACT TEXT FROM PDF
            if file_name.endswith(".pdf"):
                pdf_reader = PdfReader(
                    io.BytesIO(resume_file.getvalue())
                )

                extracted_pages = []

                for page in pdf_reader.pages:
                    page_text = page.extract_text()

                    if page_text:
                        extracted_pages.append(page_text)

                resume_text = "\n".join(extracted_pages)

            # B. EXTRACT TEXT FROM WORD DOCUMENT
            elif file_name.endswith(".docx"):
                document = Document(
                    io.BytesIO(resume_file.getvalue())
                )

                resume_text = "\n".join(
                    paragraph.text
                    for paragraph in document.paragraphs
                )

            # C. EXTRACT TEXT FROM OTHER FILES
            else:
                resume_text = resume_file.getvalue().decode(
                    "utf-8",
                    errors="replace"
                )

            # D. VALIDATE EXTRACTED TEXT
            if not resume_text.strip():
                st.warning(
                    "No readable text was extracted. "
                    "Scanned PDFs require OCR support."
                )

            else:
                st.success("Resume uploaded successfully!")

                st.write("**Target Role:**", target_role)

                with st.expander("Preview extracted resume"):
                    st.text(resume_text[:2000])

                # E. RUN CAREER GAP ANALYSIS
                with st.spinner(
                    "Searching jobs and analyzing your skills..."
                ):
                    result = analyze_career_gap(
                        target_role,
                        resume_text
                    )

                # SAVE CAREER CONTEXT FOR ASK AI
                st.session_state["career_context"] = {
                    "target_role": target_role,
                    "resume_text": resume_text,
                    "report": result["report"],
                    "matched_jobs": result.get("matched_jobs", []),
                    "candidate_skills": result.get(
                        "candidate_skills", []
                    )
                }

                # F. DISPLAY AI REPORT
                st.divider()
                st.header("📊 Your Career Gap Report")

                st.markdown(result["report"])

                # G. DISPLAY MATCHED JOBS
                st.divider()
                st.header("🔎 Matched Jobs")

                matched_jobs = result.get("matched_jobs", [])

                if matched_jobs:
                    for index, job in enumerate(
                        matched_jobs,
                        start=1
                    ):
                        title = job.get(
                            "job_title",
                            "Untitled Job"
                        )

                        with st.expander(f"{index}. {title}"):
                            st.write(
                                "**Experience level:**",
                                job.get("experience_level", "Unknown")
                            )

                            st.write(
                                "**Matching skills:**",
                                ", ".join(
                                    job.get("matching_skills", [])
                                ) or "None detected"
                            )

                            st.write(
                                "**Missing skills:**",
                                ", ".join(
                                    job.get("missing_skills", [])
                                ) or "None detected"
                            )

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric(
                                    "Skills matched",
                                    f'{job.get("match_percentage", 0)}%'
                                )

                            with col2:
                                st.metric(
                                    "Career gap",
                                    f'{job.get("career_gap", 0)}%'
                                )

                            with col3:
                                st.metric(
                                    "ATS-Style Match Score",
                                    f'{job.get("ats_score", 0)}%'
                                )

                else:
                    st.info("No matching job details were returned.")

                # H. DISPLAY SKILLS DETECTED
                st.divider()
                st.subheader("🛠️ Skills Detected in Your Resume")

                candidate_skills = result.get(
                    "candidate_skills",
                    []
                )

                if candidate_skills:
                    st.write(", ".join(candidate_skills))
                else:
                    st.write(
                        "No matching skills were detected "
                        "from the retrieved job listings."
                    )

        except Exception as e:
            st.error("The analysis could not be completed.")
            st.exception(e)


# --------------------------------------------------
# 4. ASK AI CHAT INTERFACE
# --------------------------------------------------

st.divider()
st.header("💬 Ask AI")

st.write(
    "Ask questions about your career, skills, "
    "learning roadmap, or target job role."
)


# --------------------------------------------------
# 5. CHAT HISTORY
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get a new question from the user
user_question = st.chat_input(
    "Ask something about your career..."
)


# --------------------------------------------------
# 6. GENERATE CONTEXT-AWARE AI ANSWER
# --------------------------------------------------

if user_question:

    # Save the user's question
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_question
    })

    # Display the user's question
    with st.chat_message("user"):
        st.markdown(user_question)

    # Get the saved career analysis
    career_context = st.session_state.get(
        "career_context",
        None
    )

    if career_context is None:
        answer = (
            "Please upload your resume and click "
            "'Analyze Career Gap' first. Then I can "
            "answer questions using your career analysis."
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                try:
                    response = groq_client.chat.completions.create(
                        model="openai/gpt-oss-20b",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a career-focused AI assistant. "
                                    "Use the supplied career context to "
                                    "personalize your answers. Answer "
                                    "questions about career development, "
                                    "technical skills, learning roadmaps, "
                                    "projects, interviews, and target roles. "
                                    "For complex questions, give detailed, "
                                    "structured and practical guidance. "
                                    "For simple questions, answer concisely. "
                                    "Prioritize role-specific technical "
                                    "skills over generic soft skills when "
                                    "recommending what to learn. For "
                                    "unrelated topics, politely redirect "
                                    "the user to career-related questions. "
                                    "Never claim to have reviewed information "
                                    "that is not in the supplied context."
                                )
                            },
                            {
                                "role": "user",
                                "content": (
                                    "Here is my career analysis context:\n"
                                    + str(career_context)
                                    + "\n\nMy question is:\n"
                                    + user_question
                                )
                            }
                        ]
                    )

                    answer = (
                        response.choices[0].message.content
                        or "Sorry, I couldn't generate an answer."
                    )

                    st.markdown(answer)

                except Exception:
                    answer = (
                        "Sorry, I couldn't generate a response. "
                        "Please try again."
                    )
                    st.error(answer)

    # Save the AI's answer
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })