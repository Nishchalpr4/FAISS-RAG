import os
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# Folders for input data and output FAISS index
DATA_DIR = "data"
FAISS_DIR = "faiss_store"
os.makedirs(FAISS_DIR, exist_ok=True)

# Load lightweight AI model that turns sentences into numbers (embeddings)
model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = []    # Holds plain text lines to convert into vectors
metadata = []  # Remembers which file each text came from

# 1. READ TEXT FILE (docs.txt)
# We read each line as a separate piece of information.
with open(os.path.join(DATA_DIR, "docs.txt"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            chunks.append(line)
            metadata.append({"source": "docs.txt", "content": line})

# 2. READ CSV FILE (products.csv)
# We turn each product row into a natural English sentence.
df = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
for _, row in df.iterrows():
    text = f"{row['name']} costs ${row['price']}. {row['description']}"
    chunks.append(text)
    metadata.append({"source": "products.csv", "content": text})

# 3. READ JSON FILE (data.json)
# We turn each company fact into a simple sentence.
with open(os.path.join(DATA_DIR, "data.json"), encoding="utf-8") as f:
    company = json.load(f)
for key, value in company.items():
    text = f"The company {key} is: {value}."
    chunks.append(text)
    metadata.append({"source": "data.json", "content": text})

# 4. CONVERT TEXT TO NUMBERS & SAVE TO FAISS VECTOR INDEX
# model.encode() converts all sentences into lists of numbers (384 numbers per sentence)
embeddings = model.encode(chunks)

# IndexFlatL2 creates a simple search index using straight-line math distance
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings, dtype=np.float32))

# Save the FAISS vector index and text lookup map to disk
faiss.write_index(index, os.path.join(FAISS_DIR, "index.faiss"))
with open(os.path.join(FAISS_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"Done! Successfully converted {len(chunks)} items into FAISS vector database.")
