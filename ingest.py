import os
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
FAISS_DIR = "faiss_store"
os.makedirs(FAISS_DIR, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")
chunks, metadata = [], []

# 1. Plain Text
with open(os.path.join(DATA_DIR, "docs.txt"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            chunks.append(line)
            metadata.append({"source": "docs.txt", "content": line})

# 2. CSV
df = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
for _, row in df.iterrows():
    text = f"{row['name']} costs ${row['price']}. {row['description']}"
    chunks.append(text)
    metadata.append({"source": "products.csv", "content": text})

# 3. JSON
with open(os.path.join(DATA_DIR, "data.json"), encoding="utf-8") as f:
    company = json.load(f)
for k, v in company.items():
    text = f"The company {k} is: {v}."
    chunks.append(text)
    metadata.append({"source": "data.json", "content": text})

# Embed & Index
embeddings = model.encode(chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings, dtype=np.float32))

faiss.write_index(index, os.path.join(FAISS_DIR, "index.faiss"))
with open(os.path.join(FAISS_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Ingestion complete. Indexed {len(chunks)} chunks.")
