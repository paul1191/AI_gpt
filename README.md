# ⚖️ RegAnalyst — On-Premise Multi-Agent Regulatory AI

A fully local, privacy-preserving regulatory document analyser and Q&A assistant.
**Zero cloud dependencies. Zero API costs. Everything runs on your machine.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     React Frontend (Vite)                    │
│  Query Bar │ Agent Trace │ SHAP Analysis │ Document Manager  │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP
┌────────────────────────▼────────────────────────────────────┐
│                   FastAPI Backend                            │
│                                                              │
│  ┌─────────────────── Multi-Agent Pipeline ───────────────┐  │
│  │  1. QueryAnalyser  → Intent + Entity Extraction        │  │
│  │  2. Retriever      → ChromaDB Semantic Search          │  │
│  │  3. Synthesiser    → Grounded Answer Generation        │  │
│  │  4. Critic         → Confidence Scoring + Validation   │  │
│  │  5. Formatter      → Structured Output + Citations     │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  SHAP Explainability: TF-IDF + LinearSVC + SHAP        │ │
│  │  → Per-chunk attribution + top feature importance       │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────┬─────────────────────────┬───────────────────┘
               │                         │
┌──────────────▼──────┐    ┌─────────────▼──────────────────┐
│  Ollama (Llama 3)   │    │  ChromaDB (Vector Store)        │
│  localhost:11434    │    │  all-MiniLM-L6-v2 embeddings    │
│  Fully local LLM    │    │  Persistent on ./chroma_db/     │
└─────────────────────┘    └─────────────────────────────────┘
```

---

## Prerequisites

| Tool    | Install                          | Notes                    |
|---------|----------------------------------|--------------------------|
| Python  | python.org (3.10+)               | Already installed?       |
| Node.js | nodejs.org (v18+)                | For React frontend       |
| Ollama  | https://ollama.com/download      | Local LLM runtime        |

---

## Setup (One-Time)

### 1. Install & Start Ollama + Pull Llama 3

```bash
# After installing Ollama:
ollama serve                    # starts Ollama server on :11434
ollama pull llama3              # downloads ~4.7GB model (one time only)
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv

# Activate:
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

---

## Running the App

### Terminal 1 — Ollama (if not already running)
```bash
ollama serve
```

### Terminal 2 — Backend
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

### Terminal 3 — Frontend
```bash
cd frontend
npm run dev
```

Open: **http://localhost:5173**

---

## Usage

1. **Upload Documents**: Click "Upload Regulatory Document" in the sidebar
   - Supports PDF, DOCX, TXT, MD
   - Documents are chunked and embedded automatically into ChromaDB

2. **Ask Questions**: Type a regulatory question in the query bar
   - e.g. *"What are the capital adequacy requirements under Basel III?"*
   - e.g. *"Summarise the data retention obligations under GDPR Article 5"*
   - Press `Ctrl+Enter` or click `Analyse`

3. **Review Results**:
   - **Answer tab**: Grounded answer with source citations [1] [2]
   - **Agent Trace tab**: Full audit trail of each agent's reasoning
   - **SHAP Analysis tab**: Feature importance — what words drove the answer
   - **References tab**: Source chunks with similarity scores

---

## Explainability Design

### Agent Trace
Every response exposes the full pipeline:
- **QueryAnalyser** — what intent was detected, what entities were extracted
- **Retriever** — which documents were searched, similarity scores
- **Synthesiser** — how context was used to build the answer
- **Critic** — confidence score, groundedness check, detected issues
- **Formatter** — how references were structured

### SHAP Analysis
- **Method**: TF-IDF vectorisation → LinearSVC classifier → SHAP LinearExplainer
- **Per-chunk**: SHAP importance score for each retrieved chunk
- **Per-feature**: Top TF-IDF tokens that most influenced relevance
- **Positive SHAP**: Token pushed chunk toward being relevant
- **Negative SHAP**: Token pushed chunk away from relevance

### Document References
Every answer includes:
- Source filename
- Chunk index within document
- Semantic similarity score (%)
- Full chunk text (expandable)

---

## Configuration

Edit `backend/.env` (create if needed):

```env
CHROMA_PATH=./chroma_db
EMBED_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=800
CHUNK_OVERLAP=150
OLLAMA_BASE=http://localhost:11434
```

### Switch LLM Model

In `backend/agents/orchestrator.py`, change:
```python
MODEL = "llama3"          # default
# MODEL = "mistral"       # smaller, faster
# MODEL = "llama3:70b"    # larger, more capable
```
Then: `ollama pull <model-name>`

---

## File Structure

```
reg-analyser/
├── backend/
│   ├── main.py                        # FastAPI app, routes
│   ├── requirements.txt
│   ├── agents/
│   │   └── orchestrator.py            # 5-agent pipeline
│   ├── rag/
│   │   └── document_store.py          # ChromaDB ingestion & retrieval
│   └── explainability/
│       └── shap_explainer.py          # SHAP analysis
├── frontend/
│   ├── src/
│   │   ├── App.jsx                    # Full React UI
│   │   └── main.jsx                   # Entry point
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
└── README.md
```

---

## API Endpoints

| Method | Path                    | Description                          |
|--------|-------------------------|--------------------------------------|
| GET    | /health                 | Check Ollama + ChromaDB status       |
| POST   | /documents/upload       | Upload & index a document            |
| GET    | /documents              | List all indexed documents           |
| DELETE | /documents/{doc_id}     | Remove a document                    |
| POST   | /query                  | Full analysis query                  |
| GET    | /query/stream           | Streaming token response             |
| GET    | /docs                   | Auto-generated Swagger UI            |

---

## Cost: $0 — Everything Is Local

| Component          | Tool                     | Cost  |
|--------------------|--------------------------|-------|
| LLM                | Ollama + Llama 3         | Free  |
| Embeddings         | SentenceTransformers     | Free  |
| Vector DB          | ChromaDB                 | Free  |
| Backend            | FastAPI + Python         | Free  |
| Frontend           | React + Vite             | Free  |
| Hosting            | Your own machine         | Free  |
| Explainability     | SHAP + scikit-learn      | Free  |

---

## Minimum Hardware

| Spec   | Minimum             | Recommended           |
|--------|---------------------|-----------------------|
| RAM    | 8GB                 | 16GB+                 |
| GPU    | Not required        | Any GPU speeds Ollama |
| Disk   | 10GB (model + DB)   | 20GB+                 |
| CPU    | 4 cores             | 8+ cores              |
