"""
chat.py — ShopEase RAG Chatbot
================================
Main entry point. Ties together:
  1. FAISS vector search  (static knowledge: docs, products, company info)
  2. Live REST APIs       (dynamic data: weather, deals, stock)
  3. Groq LLM             (generates a final natural-language answer)

Run:
    # Terminal 1 — start the dummy API server
    uvicorn apis.university_apis:app --port 8001

    # Terminal 2 — start the chatbot
    python chat.py

Try asking:
    "What is your return policy?"          -> FAISS
    "Are there any deals today?"           -> API
    "Is the webcam in stock?"              -> API
    "Will rain delay my delivery?"         -> API
    "Who is the CEO of ShopEase?"          -> FAISS
    "What headphones do you sell?"         -> FAISS
"""

import os
import json
import numpy as np
import faiss
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from router import keyword_router, fetch_api_data
# To use LLM routing instead, swap the import above to:
# from router import llm_router as keyword_router, fetch_api_data

load_dotenv()

# ── Load FAISS index + metadata ────────────────────────────────────────────────
# These files were created by ingest.py. We load them once at startup.
FAISS_DIR     = "faiss_store"
INDEX_PATH    = os.path.join(FAISS_DIR, "index.faiss")
METADATA_PATH = os.path.join(FAISS_DIR, "metadata.json")

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

with open(METADATA_PATH, encoding="utf-8") as f:
    metadata = json.load(f)

print(f"Index loaded. {index.ntotal} vectors ready.\n")

# ── Load embedding model ───────────────────────────────────────────────────────
# Same model used during ingestion — MUST match, otherwise vectors are incompatible.
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model ready.\n")

# ── Groq client ────────────────────────────────────────────────────────────────
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── FAISS search function ──────────────────────────────────────────────────────
def search_faiss(question: str, top_k: int = 3) -> list[str]:
    """
    Converts 'question' into a vector, then finds the top_k closest
    vectors in the FAISS index. Returns the matching text chunks.

    top_k=3 means: return the 3 most semantically similar passages.
    """
    # Embed the question the same way we embedded the documents
    query_vector = model.encode([question])                      # shape: (1, 384)
    query_vector = np.array(query_vector, dtype=np.float32)

    # index.search returns:
    #   distances → how far each result is (smaller = more similar for L2)
    #   indices   → position in the metadata list
    distances, indices = index.search(query_vector, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:  # -1 means no result found
            chunk = metadata[idx]
            results.append(
                f"[Source: {chunk['source']}] {chunk['content']}"
            )
    return results


# ── Build the final answer using Groq ─────────────────────────────────────────
def generate_answer(question: str, context_chunks: list[str]) -> str:
    """
    Sends the retrieved context + user question to Groq's LLM.
    The system prompt instructs the model to ONLY use the provided context
    — this prevents hallucination and keeps answers grounded in your data.
    """
    context_text = "\n\n".join(context_chunks) if context_chunks else "No context found."

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant for ShopEase, an online tech store. "
                "Answer the user's question using ONLY the context provided below. "
                "If the context doesn't contain enough information, say so clearly. "
                "Be concise and friendly.\n\n"
                f"CONTEXT:\n{context_text}"
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.3,   # Lower = more factual, less creative
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


# ── Main chat loop ─────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  ShopEase RAG Chatbot")
    print("  Type 'quit' to exit")
    print("=" * 50)
    print()

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Step 1: Router decides where to look
        routing = keyword_router(question)
        source  = routing["source"]
        print(f"\n[Router] Source: {source.upper()}")

        context_chunks = []

        # Step 2a: If FAISS is needed, search the vector store
        if source in ("faiss", "both"):
            faiss_results = search_faiss(question, top_k=3)
            context_chunks.extend(faiss_results)
            print(f"[FAISS]  Found {len(faiss_results)} relevant chunk(s)")

        # Step 2b: If API is needed, call the live endpoint(s)
        if source in ("api", "both"):
            api_results = fetch_api_data(routing["api_endpoints"])
            for result in api_results:
                if "data" in result:
                    # Convert the JSON response into a readable string for the LLM
                    api_text = f"[Live API: {result['url']}] {json.dumps(result['data'])}"
                    context_chunks.append(api_text)
                    print(f"[API]    Called: {result['url']}")
                else:
                    print(f"[API]    Error calling {result['url']}: {result.get('error')}")

        # Step 3: Generate the final answer using Groq
        print("[Groq]   Generating answer...\n")
        answer = generate_answer(question, context_chunks)

        print(f"Bot: {answer}")
        print()


if __name__ == "__main__":
    main()
