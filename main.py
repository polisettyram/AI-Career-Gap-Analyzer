import os
import re
import sqlite3

import chromadb
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==================================================
# 1. SETUP
# ==================================================

load_dotenv()

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="jobs"
)


# ==================================================
# 2. DATABASE AND SEARCH INDEX SETUP
# ==================================================

def setup_database():
    """Prepare SQLite, ChromaDB and the FTS5 search index."""

    jobs = pd.read_csv("data/jobs.csv")

    connection = sqlite3.connect("database/jobs.db")

    try:
        jobs.to_sql(
            "jobs",
            connection,
            if_exists="replace",
            index=False
        )

        jobs["job_document"] = (
            "Job Title: "
            + jobs["Title"].fillna("").astype(str)
            + ". Experience Level: "
            + jobs["ExperienceLevel"].fillna("").astype(str)
            + ". Years of Experience: "
            + jobs["YearsOfExperience"].fillna("").astype(str)
            + ". Skills: "
            + jobs["Skills"].fillna("").astype(str)
            + ". Responsibilities: "
            + jobs["Responsibilities"].fillna("").astype(str)
            + ". Keywords: "
            + jobs["Keywords"].fillna("").astype(str)
        )

        connection.execute(
            "DROP TABLE IF EXISTS jobs_fts"
        )

        connection.execute("""
            CREATE VIRTUAL TABLE jobs_fts
            USING fts5(
                job_id UNINDEXED,
                Title,
                Skills,
                Responsibilities,
                Keywords
            )
        """)

        connection.executemany("""
            INSERT INTO jobs_fts
                (job_id, Title, Skills, Responsibilities, Keywords)
            VALUES (?, ?, ?, ?, ?)
        """, [
            (
                f"job_{i}_{row['JobID']}",
                str(row["Title"]) if pd.notna(row["Title"]) else "",
                str(row["Skills"]) if pd.notna(row["Skills"]) else "",
                str(row["Responsibilities"])
                if pd.notna(row["Responsibilities"]) else "",
                str(row["Keywords"])
                if pd.notna(row["Keywords"]) else ""
            )
            for i, row in jobs.iterrows()
        ])

        connection.commit()

    finally:
        connection.close()

    # Generate and store embeddings for the job descriptions.
    # This setup function is called only when explicitly requested.
    embeddings = model.encode(
        jobs["job_document"].tolist(),
        show_progress_bar=True
    )

    collection.upsert(
        ids=[
            f"job_{i}_{job_id}"
            for i, job_id in enumerate(jobs["JobID"])
        ],
        embeddings=embeddings.tolist(),
        documents=jobs["job_document"].tolist(),
        metadatas=[
            {
                "title": str(row["Title"]),
                "experience_level": str(row["ExperienceLevel"]),
                "years_of_experience": str(row["YearsOfExperience"])
            }
            for _, row in jobs.iterrows()
        ]
    )

    print("Database and search index prepared.")
    print("Jobs loaded:", len(jobs))
    print("Vectors stored:", collection.count())


# ==================================================
# 3. EXPERIENCE CLASSIFICATION
# ==================================================

def classify_experience(level):
    level = str(level).strip().lower()

    if level in ["fresher", "entry-level", "junior"]:
        return "Beginner"

    if level in [
        "experienced",
        "mid-level",
        "mid-senior",
        "mid-senior level"
    ]:
        return "Intermediate"

    if level in ["senior", "senior-level", "lead"]:
        return "Advanced"

    return "Other"


# ==================================================
# 4. MAIN CAREER GAP ANALYSIS FUNCTION
# ==================================================

def analyze_career_gap(target_role, resume_text):

    if not target_role.strip():
        raise ValueError("Please enter a target job role.")

    if not resume_text.strip():
        raise ValueError("No readable resume text was found.")

    # --------------------------------------------------
    # A. SEMANTIC SEARCH
    # --------------------------------------------------

    query = (
        f"Target role: {target_role}\n"
        f"Resume:\n{resume_text[:6000]}"
    )

    query_embedding = model.encode([query]).tolist()
    query_vector = model.encode(
        query,
        normalize_embeddings=True
    )

    semantic_results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(5, collection.count())
    )

    # --------------------------------------------------
    # B. KEYWORD SEARCH WITH FTS5 AND BM25
    # --------------------------------------------------

    terms = re.findall(r"\w+", target_role)

    if not terms:
        raise ValueError("Please enter a valid target role.")

    fts_query = " OR ".join(
        f'"{term}"' for term in terms
    )

    connection = sqlite3.connect("database/jobs.db")

    try:
        keyword_results = connection.execute("""
            SELECT job_id, Title, Skills,
                   bm25(jobs_fts) AS score
            FROM jobs_fts
            WHERE jobs_fts MATCH ?
            ORDER BY score ASC
            LIMIT 5
        """, (fts_query,)).fetchall()

        # Convert keyword search ranking into a 0-100 score
        keyword_rank_scores = {}
        total_keyword_results = len(keyword_results)

        for rank, result in enumerate(keyword_results, start=1):
            if total_keyword_results == 1:
                score = 100.0
            elif total_keyword_results > 1:
                score = 100.0 * (
                    (total_keyword_results - rank)
                    / (total_keyword_results - 1)
                )
            else:
                score = 0.0

            keyword_rank_scores[result[0]] = round(score, 2)

        # --------------------------------------------------
        # C. RECIPROCAL RANK FUSION
        # --------------------------------------------------

        rrf_scores = {}
        rrf_k = 60

        for rank, job_id in enumerate(
            semantic_results["ids"][0],
            start=1
        ):
            rrf_scores[job_id] = (
                rrf_scores.get(job_id, 0)
                + 1 / (rrf_k + rank)
            )

        for rank, job in enumerate(
            keyword_results,
            start=1
        ):
            job_id = job[0]

            rrf_scores[job_id] = (
                rrf_scores.get(job_id, 0)
                + 1 / (rrf_k + rank)
            )

        ranked_jobs = sorted(
            rrf_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        # --------------------------------------------------
        # D. RETRIEVE JOB DETAILS
        # --------------------------------------------------

        ranked_job_details = []

        for job_id, score in ranked_jobs:

            job = connection.execute("""
                SELECT Title, Skills
                FROM jobs_fts
                WHERE job_id = ?
                LIMIT 1
            """, (job_id,)).fetchone()

            row_number = int(job_id.split("_")[1]) + 1

            experience_row = connection.execute("""
                SELECT ExperienceLevel
                FROM jobs
                WHERE rowid = ?
            """, (row_number,)).fetchone()

            if job is None:
                continue

            experience_level = (
                experience_row[0]
                if experience_row
                else "Unknown"
            )

            ranked_job_details.append({
                "job_id": job_id,
                "title": job[0],
                "required_skills": job[1] or "",
                "experience_level": experience_level,
                "experience_category": classify_experience(
                    experience_level
                ),
                "rrf_score": round(score, 6)
            })

    finally:
        connection.close()

    if not ranked_job_details:
        raise ValueError(
            "No matching jobs found. Try another target role."
        )

    # Rank jobs by relevance, not by experience level.
    ranked_job_details.sort(
        key=lambda job: job["rrf_score"],
        reverse=True
    )

    # --------------------------------------------------
    # REMOVE DUPLICATE JOB TITLES
    # --------------------------------------------------

    unique_jobs = []
    seen_titles = set()

    for job in ranked_job_details:
        title = " ".join(
            job["title"].strip().lower().split()
        )

        if title not in seen_titles:
            seen_titles.add(title)
            unique_jobs.append(job)

    ranked_job_details = unique_jobs

    # Calculate semantic similarity for retrieved jobs
    query_vector = model.encode(
        query,
        normalize_embeddings=True
    )

    semantic_scores = {}

    job_vectors = collection.get(
        ids=[job["job_id"] for job in ranked_job_details],
        include=["embeddings"]
    )

    for job_id, embedding in zip(
        job_vectors["ids"],
        job_vectors["embeddings"]
    ):
        similarity = cosine_similarity(
            query_vector.reshape(1, -1),
            [embedding]
        )[0][0]

        semantic_scores[job_id] = round(
            max(0.0, min(1.0, float(similarity))) * 100,
            2
        )

    # --------------------------------------------------
    # E. DETECT TECHNICAL SKILLS IN THE UPLOADED RESUME
    # --------------------------------------------------

    # Ignore generic soft skills
    ignored_skills = {
        "communication",
        "communication skills",
        "teamwork",
        "team work",
        "leadership",
        "problem solving",
        "problem-solving",
        "critical thinking",
        "time management",
        "interpersonal skills",
        "adaptability",
        "creativity",
        "attention to detail",
        "organizational skills",
        "presentation skills",
        "collaboration",
        "multitasking",
        "work ethic"
    }

    known_skills = set()

    for job in ranked_job_details:
        for skill in re.split(
            r"[;,|]",
            job["required_skills"]
        ):
            skill = skill.strip()
            skill_lower = skill.lower()

            if skill and skill_lower not in ignored_skills:
                known_skills.add(skill)

    resume_lower = resume_text.lower()
    candidate_skills = set()

    for skill in known_skills:
        skill_lower = skill.lower().strip()

        pattern = (
            r"(?<!\w)"
            + re.escape(skill_lower)
            + r"(?!\w)"
        )

        if re.search(pattern, resume_lower):
            candidate_skills.add(skill_lower)

    if "java" in candidate_skills:
        candidate_skills.add("core java")


    # --------------------------------------------------
    # F. CALCULATE SKILL GAPS
    # --------------------------------------------------

    career_gap_reports = []

    for job in ranked_job_details:
        required_skills = {
            skill.strip().lower()
            for skill in re.split(
                r"[;,|]",
                job["required_skills"]
            )
            if skill.strip()
        }

        # Remove generic soft skills before calculating percentages
        required_skills = {
            skill for skill in required_skills
            if skill not in ignored_skills
        }

        matching_skills = (
            required_skills & candidate_skills
        )

        if (
            "core java" in required_skills
            and "java" in candidate_skills
        ):
            matching_skills.add("core java")

        missing_skills = (
            required_skills - matching_skills
        )

        total_skills = len(required_skills)

        gap_percentage = (
            len(missing_skills) / total_skills * 100
            if total_skills
            else 0
        )
        match_percentage = 100 - gap_percentage

        # ATS-style match score
        semantic_score = semantic_scores.get(
            job["job_id"], 0.0
        )

        keyword_score = keyword_rank_scores.get(
            job["job_id"], 0.0
        )

        ats_score = (
            0.50 * match_percentage
            + 0.35 * semantic_score
            + 0.15 * keyword_score
        )

        career_gap_reports.append({
            "job_title": job["title"],
            "experience_level": job["experience_level"],
            "matching_skills": sorted(matching_skills),
            "missing_skills": sorted(missing_skills),
            "career_gap": round(gap_percentage, 2),
            "match_percentage": round(match_percentage, 2),
            "semantic_score": round(semantic_score, 2),
            "keyword_score": round(keyword_score, 2),
            "ats_score": round(ats_score, 2),
            "rrf_score": job["rrf_score"]
        })

    # --------------------------------------------------
    # G. GENERATE THE GROQ REPORT
    # --------------------------------------------------

    prompt = f"""
You are an AI career mentor.

Target role:
{target_role}

Uploaded resume:
{resume_text[:6000]}

Skills detected in the resume:
{sorted(candidate_skills)}

Matched job listings and calculated skill gaps:
{career_gap_reports[:5]}

Provide:
1. Overall assessment for the target role.
2. Existing strengths supported by the resume.
3. Missing skills based on the matched job listings.
4. A prioritized learning roadmap.
5. Practical projects to build.
6. Recommended next steps for job preparation.
7. The most suitable matched jobs.

Rules:
- Base your analysis on the supplied resume and job results.
- Do not invent candidate skills or experience.
- A skill missing from the resume may simply be undocumented.
- Keep recommendations realistic and beginner-friendly.
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    ai_report = response.choices[0].message.content

    return {
        "report": ai_report,
        "matched_jobs": career_gap_reports[:5],
        "candidate_skills": sorted(candidate_skills)
    }


# ==================================================
# 5. OPTIONAL COMMAND-LINE MODE
# ==================================================

if __name__ == "__main__":
    print("Preparing the job database and search index...")
    setup_database()

    target_role = input(
        "\nEnter your target job role: "
    ).strip()

    resume_path = input(
        "Enter the path to your resume text file: "
    ).strip()

    with open(resume_path, "r", encoding="utf-8") as file:
        resume_text = file.read()

    result = analyze_career_gap(
        target_role,
        resume_text
    )

    print("\n========== AI CAREER GAP REPORT ==========")
    print(result["report"])

    print("\n========== MATCHED JOBS ==========")

    for job in result["matched_jobs"]:
        print("\nJob:", job["job_title"])
        print("Matching skills:", job["matching_skills"])
        print("Missing skills:", job["missing_skills"])
        print("Career gap:", job["career_gap"], "%")
        print("Skills matched:", job["match_percentage"], "%")
