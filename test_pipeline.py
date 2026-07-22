"""Quick end-to-end test of the full RAG pipeline."""
import os, json, numpy as np, faiss
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from router import keyword_router, fetch_api_data

load_dotenv()

index = faiss.read_index("faiss_store/index.faiss")
with open("faiss_store/metadata.json", encoding="utf-8") as f:
    metadata = json.load(f)

model  = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def search_faiss(q, k=3):
    v = np.array(model.encode([q]), dtype=np.float32)
    _, idxs = index.search(v, k)
    return ["[" + metadata[i]["source"] + "] " + metadata[i]["content"]
            for i in idxs[0] if i != -1]

def ask(question):
    print("=" * 60)
    print("Q:", question)
    routing = keyword_router(question)
    print("Router:", routing["source"].upper())
    chunks = []
    if routing["source"] in ("faiss", "both"):
        chunks += search_faiss(question)
    if routing["source"] in ("api", "both"):
        for r in fetch_api_data(routing["api_endpoints"]):
            if "data" in r:
                chunks.append("[Live API] " + json.dumps(r["data"]))
    ctx = "\n".join(chunks)
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful ShopEase store assistant. Answer ONLY using this context:\n" + ctx},
            {"role": "user",   "content": question}
        ],
        temperature=0.3,
        max_tokens=150,
    )
    print("A:", resp.choices[0].message.content.strip())
    print()

ask("What is the return policy?")
ask("Are there any deals today?")
ask("Who is the CEO?")
