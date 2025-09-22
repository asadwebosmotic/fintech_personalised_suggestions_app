from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import json

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
    google_api_key=os.getenv('GEMINI_API_KEY'),
    temperature=0.1,
    max_output_tokens=150,
)

# === Prompt Template with repetition guard ===
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a financial assistant."),
    ("user", """From this financial analysis:
    {analysis}

    Already given suggestions in last 30 days:
    {past_suggestions}

    Task:
    - Generate ONE new actionable financial suggestion.
    - Must be unique (not repeating above).
    - Strictly one sentence, under 20 words.
    - Return JSON only: {{"suggestion": "<your one-liner>"}}""")
])

def generate_daily_suggestions():
    """Generate unique one-liner suggestions and append them per user."""
    users = list(finance_profiles_col.find({}, {"user_id": 1, "founded_pattern": 1, "_id": 0}))

    for user in users:
        user_id = user.get("user_id")
        founded_pattern = user.get("founded_pattern")

        if not founded_pattern:
            print(f"⚠️ Skipping {user_id} — no founded_pattern available.")
            continue

        # Fetch past 30 days suggestions
        past_entry = suggestions_col.find_one({"user_id": user_id})
        past_suggestions = []
        if past_entry:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
            past_suggestions = [
                s["text"] for s in past_entry.get("suggestions", [])
                if s.get("created_at") >= cutoff_date
            ]

        # Format past suggestions for prompt
        past_suggestions_str = json.dumps(past_suggestions, ensure_ascii=False)

        # Run LLM with prompt
        chain = LLMChain(
            llm = llm,
            prompt = prompt_template,
        )
        
        raw_output = chain.invoke({
            "analysis": founded_pattern,
            "past_suggestions": past_suggestions_str
        })

        # Depending on LangChain version, raw_output may be dict-like
        output_text = raw_output.get("text", str(raw_output)).strip()

        # Extract JSON suggestion safely
        try:
            suggestion_json = json.loads(output_text)
            suggestion = suggestion_json.get("suggestion", "").strip()
        except Exception:
            print(f"⚠️ Parsing failed for {user_id}, raw output:", output_text)
            continue

        if not suggestion:
            print(f"⚠️ No suggestion generated for {user_id}")
            continue

        # Append to MongoDB suggestions array
        suggestions_col.update_one(
            {"user_id": user_id},
            {"$push": {
                "suggestions": {
                    "text": suggestion,
                    "created_at": datetime.now(timezone.utc)
                }
            }},
            upsert=True
        )

        print(f"✅ New suggestion for {user_id}: {suggestion}")

if __name__ == '__main__':
    generate_daily_suggestions()
