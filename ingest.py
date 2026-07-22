"""
ingest.py — FAISS Ingestion Pipeline
======================================
Loads data from 3 source types → embeds each chunk → saves a FAISS index.

Run this ONCE before starting the chatbot:
    python ingest.py
"""

import os
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR      = "data"
FAISS_DIR     = "faiss_store"
INDEX_PATH    = os.path.join(FAISS_DIR, "index.faiss")
METADATA_PATH = os.path.join(FAISS_DIR, "metadata.json")

os.makedirs(FAISS_DIR, exist_ok=True)

# ── Load embedding model ───────────────────────────────────────────────────────
# all-MiniLM-L6-v2: a small (80 MB) model that converts any sentence into
# a 384-dimensional vector. Downloads automatically on first run.
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model ready.\n")

chunks   = []   # raw text strings — one per chunk
metadata = []   # parallel list of dicts: where did each chunk come from?

# ── Source 1: docs.txt (plain text) ───────────────────────────────────────────
# Each line is already a self-contained sentence → treat each line as one chunk.
print("Loading docs.txt...")
with open(os.path.join(DATA_DIR, "docs.txt"), encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]

for line in lines:
    chunks.append(line)
    metadata.append({"source": "docs.txt", "type": "text", "content": line})

print(f"  -> {len(lines)} chunks\n")

# ── Source 2: products.csv ─────────────────────────────────────────────────────
# Turn each row into a readable sentence — embedding models understand language,
# not raw CSV values, so we convert before embedding.
print("Loading products.csv...")
df = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))

for _, row in df.iterrows():
    text = f"{row['name']} costs ${row['price']}. {row['description']}"
    chunks.append(text)
    metadata.append({"source": "products.csv", "type": "csv", "content": text})

print(f"  -> {len(df)} chunks\n")

# ── Source 3: data.json ────────────────────────────────────────────────────────
# Flatten the JSON object into one readable sentence per key-value pair.
# This way each fact is independently searchable.
print("Loading data.json...")
with open(os.path.join(DATA_DIR, "data.json"), encoding="utf-8") as f:
    company = json.load(f)

for key, value in company.items():
    text = f"The company {key} is: {value}."
    chunks.append(text)
    metadata.append({"source": "data.json", "type": "json", "content": text})

print(f"  -> {len(company)} chunks\n")

# ── Embed all chunks ───────────────────────────────────────────────────────────
# model.encode() runs each chunk through the neural network.
# Output: a 2D numpy array of shape (num_chunks, 384)
print(f"Embedding {len(chunks)} chunks total...")
embeddings = model.encode(chunks, show_progress_bar=True)
print(f"Embedding shape: {embeddings.shape}\n")

# ── Build FAISS IndexFlatL2 ────────────────────────────────────────────────────
# IndexFlatL2:
#   - "Flat"  → stores every vector as-is, no compression
#   - "L2"    → measures similarity using Euclidean distance
#   - Brute-force: checks query against ALL stored vectors → 100% accurate
#   - Perfect for small datasets (< a few thousand chunks); no trade-offs needed
dimension = embeddings.shape[1]          # 384
index     = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings, dtype=np.float32))

print(f"FAISS index built. Vectors stored: {index.ntotal}")

# ── Save to disk ───────────────────────────────────────────────────────────────
# index.faiss   → binary vector store, loaded by FAISS at query time
# metadata.json → maps vector position (0, 1, 2 …) back to its source text
faiss.write_index(index, INDEX_PATH)
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"Saved: {INDEX_PATH}")
print(f"Saved: {METADATA_PATH}")
print(f"\nTotal chunks indexed: {index.ntotal}")
print("Run the chatbot next:  python chat.py")
