import os
import json
import numpy as np
import faiss
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from router import keyword_router, fetch_api_data

load_dotenv()

INDEX_PATH = "faiss_store/index.faiss"
METADATA_PATH = "faiss_store/metadata.json"

index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, encoding="utf-8") as f:
    metadata = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def search_faiss(question: str, top_k: int = 3) -> list[str]:
    query_vector = np.array(model.encode([question]), dtype=np.float32)
    _, indices = index.search(query_vector, top_k)
    return [f"[{metadata[i]['source']}] {metadata[i]['content']}" for i in indices[0] if i != -1]

def generate_answer(question: str, context_chunks: list[str]) -> str:
    context_text = "\n".join(context_chunks) if context_chunks else "No context found."
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": f"Answer using ONLY this context. Be concise.\n\nCONTEXT:\n{context_text}"
            },
            {"role": "user", "content": question}
        ],
        temperature=0.3,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def main():
    print("🤖 RAG Chatbot ready. Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            break

        routing = keyword_router(question)
        source = routing["source"]
        print(f"[Router] Target: {source.upper()}")

        context = []
        if source in ("faiss", "both"):
            faiss_res = search_faiss(question)
            context.extend(faiss_res)

        if source in ("api", "both"):
            api_res = fetch_api_data(routing["api_endpoints"])
            for res in api_res:
                if "data" in res:
                    context.append(f"[Live API] {json.dumps(res['data'])}")

        answer = generate_answer(question, context)
        print(f"Bot: {answer}\n")

if __name__ == "__main__":
    main()
