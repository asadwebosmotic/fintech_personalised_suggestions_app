'''two slms according to the daigram'''
from pymongo import MongoClient
from groq import Groq
import os, json
from dotenv import load_dotenv

load_dotenv()

# === Connect to MongoDB ===
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["user_profiles"]
raw_profiles_col = db["sample_data"]           # raw user profiles
finance_profiles_col = db["finance_profiles"]  # aggregated schema-validated profiles

# === Groq Client ===
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- SLM: Analyze finance profiles -> Spending Patterns ---
def analyze_with_slm(force: bool = False) -> dict:
    """Analyze raw user profiles one by one and append insights into finance_profiles collection."""
    results = {}
    raw_users = list(raw_profiles_col.find({}))

    for user in raw_users:
        user_id = user.get("user_id")
        print(f"🔎 Analyzing spending pattern for user_id: {user_id}")

        # Skip if already analyzed (and not forcing)
        existing = finance_profiles_col.find_one({"user_id": user_id}, {"founded_pattern": 1})
        if existing and existing.get("founded_pattern") and not force:
            print(f"⏭️ Skipping {user_id} — founded_pattern already exists.")
            results[user_id] = existing.get("founded_pattern")
            continue

        # Convert full raw profile to JSON for SLM input
        profile_text = json.dumps(user, indent=2, default=str)

        prompt = f"""
        You are a financial insights assistant.
        Given the following validated user finance profiles (with last 2-3 months spending info):
        {profile_text}

        Task:
        1. Identify spending patterns, unusual expenses, and savings behavior individually.
        2. Summarize insights across individual users.
        3. Keep the response clear and actionable.

        Respond in natural language (JSON).
        """
        try:
            completion = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",  # SLM2 (stronger reasoning than SLM1)
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=600
            )

            analysis  = completion.choices[0].message.content.strip()
            
            if analysis:
                finance_profiles_col.update_one(
                        {"user_id": user_id},
                        {"$set": {"founded_pattern": analysis}},
                        upsert=True
                )
                print(f"✅ Analysis stored for {user_id}")
                results[user_id] = analysis
            else:
                    print(f"⚠️ Empty analysis for {user_id}")
                    # preserve previous value if any, else store empty string in results
                    results[user_id] = existing.get("founded_pattern") if existing else ""

        except Exception as e:
            print(f"⚠️ Failed to analyze {user_id}: {e}")
            results[user_id] = None

    return results

# --- Run main ---
if __name__ == "__main__":
    insights = analyze_with_slm()
    print("\n📊 Individual Spending Pattern Insights:")
    for uid, analysis in insights.items():
        print(f"\n--- {uid} ---\n{analysis or 'No analysis available'}")
