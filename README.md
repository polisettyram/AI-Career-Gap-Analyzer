# 🎯 AI Career Gap Analyzer

An AI-powered career guidance application that analyzes a user's resume, finds relevant job opportunities, calculates an ATS-style match score, identifies skill gaps, and generates a personalized career development roadmap.

## 🚀 Live Demo

**Try the application:** https://ats-career-analyzer.streamlit.app/

> The live demo may be temporarily unavailable while deployment issues are being resolved.

## ✨ Features

- **Resume Upload:** Upload resumes in TXT, PDF, DOCX, Python, Java, JavaScript, CSV, and Markdown formats.
- **Job Matching:** Find relevant job listings based on the target role and resume content.
- **Semantic Search:** Use Sentence Transformers and vector search to identify jobs based on meaning and context.
- **Keyword Search:** Use SQLite FTS5 and BM25 ranking to retrieve jobs matching important keywords.
- **Hybrid Search:** Combine semantic and keyword search results using Reciprocal Rank Fusion (RRF).
- **ATS-Style Scoring:** Calculate a project-specific score using skill matching, semantic relevance, and keyword ranking.
- **Skill Gap Analysis:** Identify matching and missing skills for each recommended job.
- **AI Career Report:** Generate personalized feedback, learning roadmaps, and project recommendations.
- **Ask AI:** Chat with an AI career assistant about your analysis, skills, and career goals.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Web interface |
| SQLite | Job data storage |
| SQLite FTS5 | Keyword search and BM25 ranking |
| ChromaDB | Vector storage and semantic retrieval |
| Sentence Transformers | Text embeddings |
| Scikit-learn | Cosine similarity and scoring |
| Groq API | AI-generated career guidance |
| Pandas | Job dataset processing |

## 🧠 How It Works

1. **Upload a resume:** The application extracts readable text from the uploaded file.
2. **Enter a target role:** For example, Java Developer or AI Engineer.
3. **Search job listings:** Semantic search and keyword search retrieve relevant opportunities.
4. **Rank results:** Reciprocal Rank Fusion combines the search rankings.
5. **Analyze skills:** The application compares detected resume skills against each job's required skills.
6. **Calculate the ATS-style score:** The project combines three components:

   - Skill match: **50%**
   - Semantic relevance: **35%**
   - Keyword-search ranking: **15%**

   These weights are a project-specific formula, not an official employer ATS score.

7. **Generate career guidance:** The Groq-powered AI produces a career report and answers follow-up questions.

## 📂 Project Structure

```text
AI-Career-Gap-Analyzer/
│
├── app.py                 # Streamlit frontend
├── main.py                # Search, scoring and AI logic
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .gitignore             # Files excluded from Git
│
├── data/
│   └── jobs.csv           # Job listings dataset
│
├── database/
│   └── jobs.db            # SQLite database
│
└── chroma_db/             # Persistent vector database
```

*The exact structure may vary depending on how the database and vector index are initialized.*

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Career-Gap-Analyzer.git
cd AI-Career-Gap-Analyzer
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Get an API key from [Groq Console](https://console.groq.com/).

Never commit your `.env` file or expose your API key publicly.

### 5. Start the application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL, usually `http://localhost:8501`.

## 🔐 Deployment

The application can be deployed using Streamlit Community Cloud.

1. Push the project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect the GitHub repository.
4. Select `app.py` as the entry point.
5. Configure `GROQ_API_KEY` in the app's Secrets settings.
6. Deploy the application.

Ensure all required dependencies, job data, and vector indexes are available or initialized during deployment.

## 📊 ATS-Style Score Formula

The project's scoring formula is:

```text
ATS Score =
    (0.50 × Skill Match Score)
  + (0.35 × Semantic Similarity Score)
  + (0.15 × Keyword Ranking Score)
```

Each component is represented on a 0–100 scale. The result is a heuristic to help users compare job relevance; it does not predict the exact score used by an employer's applicant tracking system.

## ⚠️ Limitations

- Resume analysis depends on the quality and readability of extracted text.
- Scanned PDFs may require OCR support.
- Job recommendations depend on the available dataset and search results.
- Skills not explicitly mentioned in a resume may be incorrectly treated as missing.
- AI-generated recommendations should be reviewed and verified.
- Semantic and ATS-style scores are estimates, not guarantees of selection or employment.

## 🔮 Future Improvements

- Support image-based resumes using OCR.
- Expand and regularly update the job dataset.
- Improve skill extraction and job-specific scoring.
- Add job filters for experience, location, and technology.
- Enable users to download their career analysis.
- Add authentication and saved analysis history.

## 👨‍💻 Author

**Your Name**

B.Tech Computer Science Engineering Student

GitHub: https://github.com/YOUR_USERNAME

---

*Built to help job seekers understand their skill gaps and prepare more effectively for their target roles.*
