"""
DATA INGESTION & VECTOR DATABASE BUILDER (ingest.py)

What this script does in simple terms:
1. Reads information from 3 data sources: text file (docs.txt), CSV spreadsheet (products.csv), and JSON file (data.json).
2. Converts each piece of text into a sentence.
3. Uses an AI model (SentenceTransformer) to turn sentences into lists of numbers called "embeddings".
   Embeddings let the computer understand the *meaning* of words, not just match exact letters.
4. Stores these embedding numbers into a fast search index called FAISS (Facebook AI Similarity Search).
5. Saves the FAISS index and a metadata map to disk so the chatbot (chat.py) can search through them later.
"""

import os     # For creating folders and working with file paths
import sys    # For exiting the script if errors happen
import json   # For reading data.json and saving metadata.json
import numpy as np  # For handling numerical arrays required by FAISS
import pandas as pd # For reading CSV files easily like spreadsheets
import faiss        # High-performance vector database to quickly search sentence embeddings
from sentence_transformers import SentenceTransformer  # AI model that converts text to number vectors

# Define folder locations:
# DATA_DIR: Where input files (docs.txt, products.csv, data.json) live
# FAISS_DIR: Where output index files (index.faiss, metadata.json) will be saved
DATA_DIR   = "data"
FAISS_DIR  = "faiss_store"
os.makedirs(FAISS_DIR, exist_ok=True)  # Create 'faiss_store' folder if it doesn't exist yet

# ── Safety Check: Ensure data directory exists ──────────────────────────────
if not os.path.isdir(DATA_DIR):
    print(f"[ERROR] Data directory '{DATA_DIR}' not found. Create it and add docs.txt, products.csv, data.json.")
    sys.exit(1)

# Load a small, super fast AI model that converts sentences into 384 numbers (vector embedding)
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Lists to collect our knowledge data:
chunks   = []   # Stores raw sentences/text (e.g. "Wireless Headphones cost $99.99.")
metadata = []   # Stores details about where each sentence came from (file name, type, original text)

# ── STEP 1: READ TEXT FILE (docs.txt) ────────────────────────────────────────
# Read general knowledge documents line by line.
docs_path = os.path.join(DATA_DIR, "docs.txt")
if not os.path.isfile(docs_path):
    print(f"[WARNING] '{docs_path}' not found – skipping.")
else:
    with open(docs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # Remove extra spaces or newlines
            if line:             # Ignore empty lines
                chunks.append(line)
                metadata.append({"source": "docs.txt", "type": "text", "content": line})
    print(f"  [docs.txt]      {sum(1 for m in metadata if m['source'] == 'docs.txt')} chunks loaded.")

# ── STEP 2: READ CSV FILE (products.csv) ─────────────────────────────────────
# Read product catalog table and convert each row into a full English sentence.
csv_path = os.path.join(DATA_DIR, "products.csv")
if not os.path.isfile(csv_path):
    print(f"[WARNING] '{csv_path}' not found – skipping.")
else:
    df = pd.read_csv(csv_path)
    required_cols = {"name", "price", "description"}
    # Verify the CSV contains the required column headers
    missing_cols  = required_cols - set(df.columns.str.lower())
    if missing_cols:
        print(f"[WARNING] products.csv is missing columns: {missing_cols} – skipping.")
    else:
        df.columns = df.columns.str.lower()          # Standardise column names to lowercase
        before = len(chunks)
        for _, row in df.iterrows():
            # Combine columns into a human-readable sentence for better AI comprehension
            text = f"{row['name']} costs ${row['price']}. {row['description']}"
            chunks.append(text)
            metadata.append({"source": "products.csv", "type": "csv", "content": text})
        print(f"  [products.csv]  {len(chunks) - before} chunks loaded.")

# ── STEP 3: READ JSON FILE (data.json) ───────────────────────────────────────
# Read key-value pairs of company facts and convert them into plain sentences.
json_path = os.path.join(DATA_DIR, "data.json")
if not os.path.isfile(json_path):
    print(f"[WARNING] '{json_path}' not found – skipping.")
else:
    with open(json_path, encoding="utf-8") as f:
        company = json.load(f)
    before = len(chunks)
    for key, value in company.items():
        # Format key-value pairs into clear statements (e.g. "The support email is: support@example.com.")
        text = f"The {key} is: {value}."
        chunks.append(text)
        metadata.append({"source": "data.json", "type": "json", "content": text})
    print(f"  [data.json]     {len(chunks) - before} chunks loaded.")

# ── Safety Check: Ensure we loaded at least one piece of text ─────────────────
if not chunks:
    print("[ERROR] No data loaded. Nothing to index. Aborting.")
    sys.exit(1)

# ── STEP 4: CONVERT TEXT TO VECTOR NUMBERS (EMBEDDINGS) ──────────────────────
# model.encode() translates each sentence into 384 floating-point numbers representing its meaning.
print(f"\nEncoding {len(chunks)} chunks into embeddings...")
embeddings = model.encode(chunks, show_progress_bar=True)

# ── STEP 5: CREATE & POPULATE FAISS VECTOR INDEX ─────────────────────────────
# IndexFlatL2 creates a vector search space using Euclidean distance (distance between numbers).
# embeddings.shape[1] is 384 (the dimension size of our embedding model).
index = faiss.IndexFlatL2(embeddings.shape[1])
# Add all sentence vectors into FAISS database (must convert to float32 NumPy array)
index.add(np.array(embeddings, dtype=np.float32))

# ── STEP 6: SAVE VECTOR INDEX AND METADATA TO DISK ────────────────────────────
# Save FAISS index binary file
faiss.write_index(index, os.path.join(FAISS_DIR, "index.faiss"))
# Save metadata list as JSON so we can map vector search result position (0, 1, 2...) back to original text
with open(os.path.join(FAISS_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"\n[OK] Done! Successfully indexed {len(chunks)} items into FAISS vector database.")
print(f"   -> {os.path.join(FAISS_DIR, 'index.faiss')}")
print(f"   -> {os.path.join(FAISS_DIR, 'metadata.json')}")

