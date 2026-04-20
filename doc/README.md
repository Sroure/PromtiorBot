# Promtior RAG Chatbot — Architecture

## Overview

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Promtior AI. It combines information scraped from the Promtior website with a local PDF document, stores it in a vector database, and uses GPT-4o-mini to generate answers grounded in that knowledge.

---

## Architecture

### Two-phase design

**Phase 1 — Ingestion** (`ingest.py`)

Runs once at build time to populate the knowledge base:

1. **Web scraping** — `WebBaseLoader` fetches and cleans text from three Promtior pages
2. **PDF loading** — `PyPDFLoader` extracts text from the local company PDF
3. **Chunking** — `RecursiveCharacterTextSplitter` divides all text into 1000-char overlapping chunks
4. **Embedding** — `OpenAIEmbeddings` (text-embedding-3-small) converts each chunk into a semantic vector
5. **Storage** — All vectors are persisted to ChromaDB on disk at `./chroma_db`

**Phase 2 — Serving** (`main.py`)

Runs continuously, handling incoming questions:

1. **Load** the existing ChromaDB vector store
2. **Retrieve** the 5 most semantically similar chunks for each query
3. **Augment** a prompt template with retrieved context + user question
4. **Generate** an answer via GPT-4o-mini
5. **Serve** the chain over HTTP via LangServe/FastAPI

---

## Tech Stack

| Component | Library |
|---|---|
| Web scraping | `langchain-community` WebBaseLoader |
| PDF loading | `langchain-community` PyPDFLoader |
| Text splitting | `langchain-text-splitters` |
| Embeddings | `langchain-openai` (text-embedding-3-small) |
| Vector store | `chromadb` + `langchain-chroma` |
| LLM | `langchain-openai` ChatOpenAI (gpt-4o-mini) |
| Chain | LangChain LCEL (pipe operator) |
| API server | `fastapi` + `langserve` |
| Runtime | `uvicorn` |

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/chat/invoke` | POST | Single question → answer |
| `/chat/stream` | POST | Streaming answer |
| `/chat/playground` | GET | Interactive web UI |

### Example request

```bash
curl -X POST http://localhost:8000/chat/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": "What services does Promtior offer?"}'
```

---

## Running locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env: add your OPENAI_API_KEY and make sure the PDF is at ./data/company.pdf

# 3. Build the vector store
python ingest.py

# 4. Start the server
python main.py
# → Server running at http://localhost:8000
```

---

## Deploying on Railway

```bash
# Build the Docker image (PDF must be in ./data/ at build time)
docker build --build-arg OPENAI_API_KEY=sk-... -t promtiorbot .

# Run locally to verify
docker run -p 8000:8000 promtiorbot
```

On Railway:
1. Push your repo (with the PDF included in `./data/`)
2. Add `OPENAI_API_KEY` as an environment variable in Railway settings
3. Railway auto-detects the `Dockerfile` and deploys
