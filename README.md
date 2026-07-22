# ⚡ ShopEase RAG Chatbot

An end-to-end **Retrieval-Augmented Generation (RAG)** system built with **FAISS**, **FastAPI**, **Sentence-Transformers**, and **Groq (Llama-3.1-8B)**.

Designed to dynamically route user queries between a static vector store (FAISS) and live dynamic microservices (FastAPI REST endpoints).

---

## 🗺️ System Architecture

```
                       +-------------------------+
                       |    User Query Input     |
                       +------------+------------+
                                    |
                                    v
                       +------------+------------+
                       |      Query Router       |
                       |  (Keyword / Function)   |
                       +----+---------------+----+
                            |               |
             +--------------+               +--------------+
             |                                             |
             v                                             v
+------------+------------+                   +------------+------------+
|   FAISS Vector Store    |                   |   FastAPI Live REST APIs   |
| (Static Knowledge Base) |                   |  (Dynamic & Live Data)     |
+------------+------------+                   +------------+------------+
  - Plain Text (docs.txt)                       - /api/weather
  - CSV (products.csv)                          - /api/deals
  - JSON (data.json)                            - /api/stock
             |                                             |
             +--------------+---------------+--------------+
                            |
                            v
               +------------+------------+
               |  Context Injection Engine |
               +------------+------------+
                            |
                            v
               +------------+------------+
               |   Groq LLM (Llama 3.1)  |
               +------------+------------+
                            |
                            v
               +------------+------------+
               |    Grounded Answer     |
               +-------------------------+
```

---

## ✨ Features

- **Multi-Source Data Ingestion**: Ingests plain text, CSV, and JSON files, converting structured and unstructured text into 384-dimensional vector embeddings (`all-MiniLM-L6-v2`).
- **FAISS Vector Search**: Uses `IndexFlatL2` for 100% exact semantic similarity matching without trade-offs.
- **Dynamic REST APIs**: Simulates live microservices (`/api/weather`, `/api/deals`, `/api/stock`) via FastAPI for temporal data.
- **Dual Query Routing**:
  - **Keyword Router**: Lightweight, zero-latency, deterministic routing.
  - **LLM Function-Calling Router**: Contextual tool-use selection via Groq API.
- **Grounded Answer Generation**: Strictly constrains the Groq LLM to retrieved context, preventing hallucinations.

---

## 📂 Project Structure

```
FAISS-RAG/
├── data/
│   ├── docs.txt         # Store policies & FAQs
│   ├── products.csv     # Product catalog & pricing
│   └── data.json        # Company background & facts
├── faiss_store/
│   ├── index.faiss      # Binary vector index
│   └── metadata.json    # Vector-to-chunk metadata mapping
├── apis/
│   └── university_apis.py  # FastAPI live data endpoints
├── ingest.py            # FAISS vector ingestion pipeline
├── router.py            # Keyword & LLM function-calling routing logic
├── chat.py              # Main interactive CLI chatbot interface
├── test_pipeline.py     # End-to-end integration test script
├── .env                 # API Key Configuration
├── README.md            # Comprehensive Documentation
└── requirements.txt     # Python Dependencies
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Nishchalpr4/FAISS-RAG.git
cd FAISS-RAG
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Build Vector Index
Ingest dataset into FAISS vector store:
```bash
python ingest.py
```

---

## 💻 Running the System

### Step 1: Start the REST API Microservice
```bash
python -m uvicorn apis.university_apis:app --port 8001
```

### Step 2: Start the Chatbot CLI
```bash
python chat.py
```

---

## 🎯 Example Query Routing

| User Question | Source Routed | Reason |
|---|---|---|
| *"What is your return policy?"* | **FAISS** | Matches static store policy |
| *"What headphones do you sell?"* | **FAISS** | Matches product catalog |
| *"Who is the CEO of ShopEase?"* | **FAISS** | Matches corporate metadata |
| *"Are there any deals today?"* | **API** | Hits dynamic `/api/deals` endpoint |
| *"Will rain delay my delivery?"* | **API** | Hits live `/api/weather` endpoint |
| *"Is the webcam in stock?"* | **API** | Hits warehouse `/api/stock` endpoint |

---

## 📊 Technical Architecture & Trade-Offs

### 1. Vector Index Choice: FAISS `IndexFlatL2`
- **Choice**: Exact Euclidean distance comparison over all stored vectors.
- **Trade-off**: `IndexFlatL2` operates in $O(N)$ query time. While approximate indexes like **IVF** or **HNSW** offer $O(\log N)$ scalability for millions of vectors, `IndexFlatL2` guarantees 100% search accuracy with zero index tuning overhead for small-to-medium scale datasets.

### 2. Embeddings Model: `all-MiniLM-L6-v2`
- **Choice**: Lightweight local model producing 384-dimensional dense vectors.
- **Trade-off**: Operates locally with zero latency or API costs. Offers high semantic accuracy without reliance on third-party cloud embedding APIs.

### 3. Routing Strategy: Keyword vs LLM Function-Calling
- **Keyword Router**: Executes in $<1\text{ms}$ with zero API cost. Best for deterministic trigger matching.
- **LLM Function Calling**: Better handles paraphrased input, but incurs network latency (~300-500ms) and token overhead per call.

---

## 📜 License
MIT License
