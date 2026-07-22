# ⚡ FAISS-RAG: Multi-Source Retrieval-Augmented Generation with Hybrid REST Router

A production-grade, local **Retrieval-Augmented Generation (RAG)** architecture featuring a **FAISS** vector search engine, **FastAPI** dynamic microservices, and **Groq (Llama-3.1-8B)** answer synthesis. 

Designed to demonstrate semantic document search over structured/unstructured datasets alongside dynamic REST API routing.

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
| (Static Knowledge Base) |                   |  (Dynamic & Temporal Data) |
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

## ✨ Core Features

- **Multi-Source Heterogeneous Ingestion**: Ingests plain text, CSV tabular data, and JSON key-value records, mapping them into a unified 384-dimensional embedding space using `sentence-transformers` (`all-MiniLM-L6-v2`).
- **FAISS Vector Engine**: Implements `IndexFlatL2` Euclidean distance matching for 100% search accuracy without approximate indexing trade-offs.
- **Dynamic REST Service Layer**: Mocked microservices built on **FastAPI** (`/api/weather`, `/api/deals`, `/api/stock`) representing temporal, non-vectorizable data.
- **Dual Query Routing Pipeline**:
  - **Deterministic Keyword Router**: $<1\text{ms}$ execution time with zero token overhead.
  - **LLM Function-Calling Router**: Semantic tool selection using Groq's open function-calling specification.
- **Grounded LLM Generation**: Constrains answer generation strictly to retrieved context buffers to eliminate model hallucination.

---

## 📂 Repository Structure

```
FAISS-RAG/
├── data/
│   ├── docs.txt         # Plain text policies & documentation
│   ├── products.csv     # Structured tabular catalog
│   └── data.json        # Key-value corporate metadata
├── faiss_store/
│   ├── index.faiss      # Serialized FAISS binary index
│   └── metadata.json    # Vector ID to document chunk mapping
├── apis/
│   └── university_apis.py  # FastAPI dynamic REST endpoints
├── ingest.py            # Heterogeneous data parser & FAISS builder
├── router.py            # Keyword & LLM tool-calling routing logic
├── chat.py              # Interactive CLI RAG execution loop
├── test_pipeline.py     # End-to-end integration test runner
├── .env                 # API configuration (git-ignored)
├── README.md            # Architecture & technical documentation
└── requirements.txt     # Dependency definitions
```

---

## 🚀 Quickstart

### 1. Clone & Install
```bash
git clone https://github.com/Nishchalpr4/FAISS-RAG.git
cd FAISS-RAG
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Ingest Data & Build FAISS Index
```bash
python ingest.py
```

---

## 💻 Execution Workflow

### 1. Launch REST Microservices
```bash
python -m uvicorn apis.university_apis:app --port 8001
```

### 2. Launch RAG Pipeline
```bash
python chat.py
```

---

## 🎯 Query Routing Execution Matrix

| User Query Pattern | Targeted System | Architectural Rationale |
|---|---|---|
| *"What is your return policy?"* | **FAISS Store** | Semantic retrieval from static unstructured text (`docs.txt`) |
| *"What headphones do you sell?"* | **FAISS Store** | Semantic retrieval from tabular product dataset (`products.csv`) |
| *"Who is the CEO?"* | **FAISS Store** | Key-value attribute matching from metadata (`data.json`) |
| *"Are there any deals today?"* | **REST API** | Dynamic lookup against dynamic `/api/deals` endpoint |
| *"Will rain delay my delivery?"* | **REST API** | Real-time query against temporal `/api/weather` endpoint |
| *"Is the webcam in stock?"* | **REST API** | Inventory state check against `/api/stock` endpoint |

---

## 📊 Deep-Dive Architectural Trade-Offs

### 1. FAISS `IndexFlatL2` Indexing
- **Mechanism**: Calculates exact $L_2$ (Euclidean) distance across all dense vectors:
  $$D(y, x) = \sum_{i=1}^{d} (y_i - x_i)^2$$
- **Engineering Trade-off**: Operates in linear $O(N \cdot d)$ space/time complexity. Unlike approximate nearest neighbor algorithms (**IVF**, **HNSW**), `IndexFlatL2` guarantees zero recall loss, making it optimal for precision-critical datasets under $10^5$ vectors.

### 2. Local Dense Embeddings vs API-based Models
- **Mechanism**: Utilizes `all-MiniLM-L6-v2` generating 384-dimensional vector spaces.
- **Engineering Trade-off**: Runs fully locally with zero network I/O overhead and zero cost per query, avoiding third-party rate limits while retaining high semantic retrieval performance.

### 3. Deterministic Routing vs Function-Calling LLM Routing
- **Rule-based Keyword Router**: Sub-millisecond execution, deterministic outcome, zero cost.
- **LLM Function Calling**: Understands rich semantic intent (e.g., *"Is it pouring outside?"* $\rightarrow$ `/api/weather`), but introduces $\sim 300\text{--}500\text{ms}$ latency and LLM token overhead.

---

## 📜 License
MIT License
