'''two slms according to the daigram'''
from pymongo import MongoClient
from groq import Groq
from pydantic import ValidationError
import os, json
from typing import List
from dotenv import load_dotenv
# === Import FinanceProfile Schema ===
from static_templates import FinanceProfile  # put your class in finance_profile_schema.py

load_dotenv()

# === Connect to MongoDB ===
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["user_profiles"]
raw_profiles_col = db["sample_data"]           # raw user profiles
finance_profiles_col = db["finance_profiles"]  # aggregated schema-validated profiles

# === Groq Client ===
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- SLM1: Convert raw profile -> FinanceProfile ---
def process_with_slm1(raw_profile: dict) -> dict:
    # 🔹 Reduce document size (keep only relevant fields + last N transactions)
    raw_minimal = {
        "user_id": raw_profile.get("user_id"),
        "name": raw_profile.get("name"),
        "dob": raw_profile.get("dob"),
        "employment": raw_profile.get("employment"),
        "accounts": [
            {
                "account_number": acc.get("account_number"),
                "transactions": acc.get("transactions", [])  # only last 5 txns
            }
            for acc in raw_profile.get("accounts", [])
        ]
    }

    schema_str = json.dumps(FinanceProfile.model_json_schema(), indent=2)


    prompt = f"""
    You are a data extraction assistant.

    Task:
    - Map the simplified user profile into the FinanceProfile schema below.
    - Fill ONLY the fields that have direct matches in the input data.
    - If a field has no value, completely omit it from the JSON (do not include null/empty).
    - Output must be valid JSON only, with no explanations.

    FinanceProfile Schema (for reference):
    {schema_str}

    Simplified profile:
    {json.dumps(raw_minimal, indent=2, default=str)}

    Return ONLY valid JSON matching the schema.
    """

    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1200  # cap output size
    )


    try:
        parsed = json.loads(completion.choices[0].message.content)
        # Validate with Pydantic schema
        profile = FinanceProfile.model_validate(parsed)
        return profile.model_dump(exclude_none=True)
    except (json.JSONDecodeError, ValidationError) as e:
        print("⚠️ Validation/Parsing failed:", e)
        print("Raw response was:", completion.choices[0].message.content)
        return None

# --- SLM2: Analyze finance profiles -> Spending Patterns ---
def analyze_with_slm2(force: bool = False) -> dict:
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

        Respond in natural language (not JSON).
        """
        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",  # SLM2 (stronger reasoning than SLM1)
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

from decimal import Decimal

def clean_for_mongo(doc: dict) -> dict:
    """Recursively convert Decimal to float so MongoDB can store it."""
    if isinstance(doc, dict):
        return {k: clean_for_mongo(v) for k, v in doc.items()}
    elif isinstance(doc, list):
        return [clean_for_mongo(v) for v in doc]
    elif isinstance(doc, Decimal):
        return float(doc)
    return doc

# --- Pipeline Runner ---
def run_pipeline():
    raw_users = list(raw_profiles_col.find({}))
    aggregated_profiles = []

    for user in raw_users:
        user_id = user.get("user_id")
        print(f"🔎 Processing user_id: {user_id}")

        # ✅ Skip if already exists in finance_profiles
        if finance_profiles_col.find_one({"user_id": user_id}):
            print(f"⏭️ Skipping {user_id}, already processed by SLM1.")
            continue
        processed = process_with_slm1(user)
        if processed:
            cleaned_doc = clean_for_mongo(processed)
            finance_profiles_col.update_one(
                {"user_id": user["user_id"]},
                {"$set": cleaned_doc},
                upsert=True
            )
            aggregated_profiles.append(cleaned_doc)

    print("✅ All profiles processed & stored in finance_profiles.")

    # Run SLM2 across all aggregated profiles
        # If no new profiles, load all existing ones
    if not aggregated_profiles:
        aggregated_profiles = list(finance_profiles_col.find({}, {"_id": 0}))  # exclude _id for cleanliness

    if aggregated_profiles:
        insights = analyze_with_slm2(aggregated_profiles)
        print("\n📊 Individual Spending Pattern Insights:")
        for uid, analysis in insights.items():
            print(f"\n--- {uid} ---\n{analysis or 'No analysis available'}")
    else:
        print("⚠️ No finance profiles available for analysis.")

# --- Run main ---
if __name__ == "__main__":
    run_pipeline()
