from datetime import datetime
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load env
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# === MongoDB Connection ===
client = MongoClient(MONGO_URI)
db = client["user_profiles"]
finance_profiles_col = db["finance_profiles"]
suggestions_col = db["daily_suggestions"]

# === Gemini LLM via LangChain ===
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_output_tokens=60
)

# === Prompt Template ===
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a personal financial assistant."),
    ("user", """Based on this user's financial analysis:
{analysis}

Create ONE short, actionable suggestion for this user.
Keep it under 20 words, simple, and non-repetitive.""")
])

def generate_daily_suggestions():
    """Generate one-liner suggestions from founded_pattern and store them uniquely in daily_suggestions."""
    users = list(finance_profiles_col.find({}, {"user_id": 1, "founded_pattern": 1, "_id": 0}))

    for user in users:
        user_id = user.get("user_id")
        founded_pattern = user.get("founded_pattern")

        if not founded_pattern:
            print(f"⚠️ Skipping {user_id} — no founded_pattern available.")
            continue

        # Check last suggestion to avoid duplicates
        last_entry = suggestions_col.find_one(
            {"user_id": user_id},
            sort=[("created_at", -1)]
        )

        # Build LangChain prompt
        chain = prompt_template | llm
        suggestion = chain.invoke({"analysis": founded_pattern}).content.strip()

        if last_entry and last_entry.get("suggestion") == suggestion:
            print(f"⏭️ Skipping {user_id} — duplicate suggestion.")
            continue

        # Insert new suggestion
        suggestions_col.insert_one({
            "user_id": user_id,
            "suggestion": suggestion,
            "created_at": datetime.utcnow()
        })

        print(f"✅ Suggestion stored for {user_id}: {suggestion}")
