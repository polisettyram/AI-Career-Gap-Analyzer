import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def analyze_career_gap(target_role, resume_text):
    prompt = f"""
    You are an AI Career Gap Analyzer.

    Target job role: {target_role}

    Resume:
    {resume_text}

    Analyze the candidate and provide:
    1. Overall career readiness
    2. Existing skills and strengths
    3. Missing skills for the target role
    4. A step-by-step learning roadmap
    5. Practical projects to build
    6. Recommended next steps

    Only claim skills and experience supported by the resume.
    Keep the report beginner-friendly.
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


