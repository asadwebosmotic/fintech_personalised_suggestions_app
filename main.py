from pymongo import MongoClient
from groq import Groq
import os
import json
from dotenv import load_dotenv

load_dotenv()

# --- MongoDB Connection ---
MONGO_URI = os.getenv("MONGO_URI")  # change if Atlas or remote
client = MongoClient(MONGO_URI)
db = client["sample_mflix"]
collection = db["comments"]

# --- Groq Client ---
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- Function: Query Mongo + Send to Groq SLM ---
def query_and_summarize(query: dict, limit: int = 5):
    # Fetch matching docs
    docs = list(collection.find(query).limit(limit))
    
    if not docs:
        return "No matching documents found."
    
    # Convert to string for LLM input
    docs_text = json.dumps(docs, indent=2, default=str)
    
    # Send to Groq's small language model (fast + accurate)
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",   # ✅ Groq’s fastest SLM
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes MongoDB query results."},
            {"role": "user", "content": f"Here are some documents:\n{docs_text}\n\nSummarize or extract useful info as per query."}
        ],
        temperature=0.5,
        max_tokens=300
    )
    
    return completion.choices[0].message.content

# --- Example Run ---
if __name__ == "__main__":
    # Dynamic MongoDB query (change to test)
    mongo_query = {"name": "Mercedes Tyler"}
    response = query_and_summarize(mongo_query, limit=3)
    print("🔎 Groq SLM Output:\n", response)
