# Project Overview — Promtior RAG Chatbot

## Approach

The goal of this challenge was to build a chatbot capable of answering questions about Promtior AI using real, up-to-date information from the company's website and an internal PDF document.

My approach was based on **Retrieval-Augmented Generation (RAG)**: instead of relying on a general-purpose LLM that knows nothing about Promtior, I built a pipeline that retrieves relevant information from a curated knowledge base and feeds it to the model as context before generating a response. This way, the chatbot only answers with facts that are grounded in real sources.

---

## Implementation Logic

The system is divided into two phases:

### Phase 1 — Knowledge Ingestion (`ingest.py`)
The first step was to build the knowledge base. I used **LangChain's WebBaseLoader** to scrape three pages from the Promtior website (`/`, `/service`, `/use-cases`) and **PyPDFLoader** to extract content from the provided PDF document. All this text was then split into overlapping chunks of 1000 characters using `RecursiveCharacterTextSplitter` — the overlap ensures that information at the boundary between two chunks is not lost.

Each chunk was then converted into a numerical vector (embedding) using **OpenAI's text-embedding-3-small model**, which captures the semantic meaning of the text. These vectors were stored in **ChromaDB**, a local vector database persisted to disk.

### Phase 2 — Serving the Chain (`main.py`)
For the serving layer, I built a RAG chain using **LangChain Expression Language (LCEL)**. When a user sends a question, the chain:
1. Converts the question into a vector and retrieves the 5 most semantically similar chunks from ChromaDB
2. Injects those chunks as context into a prompt template alongside the original question
3. Sends the prompt to **GPT-4o-mini** to generate a grounded answer
4. Returns the plain text response

The chain is exposed as a REST API using **LangServe** on top of FastAPI, automatically providing a `POST /chat/invoke` endpoint and an interactive playground at `/chat/playground`.

---

## Main Challenges and How I Overcame Them

**1. The founding date was not on the website.**
After scraping all pages of promtior.ai, the founding date was nowhere to be found. This confirmed the importance of the PDF as a secondary source — the date was extracted from there and correctly answered by the chatbot ("Promtior was founded in May 2023").

**2. PDF path configuration for Docker deployment.**
When deploying to Railway, the Docker build failed because the PDF path in the code defaulted to `./data/company.pdf`, but the file was located at `./doc/data.pdf`. I fixed this by updating the default path in `ingest.py` and including the PDF in the repository so it is available at image build time.

**3. Running ingest at build time vs. runtime.**
I decided to run `python ingest.py` during the Docker build step (`RUN python ingest.py` in the Dockerfile) rather than at server startup. This means the ChromaDB vector store is baked into the Docker image, so the container starts instantly without any scraping or embedding delay on every deploy.

---

## Tech Stack

| Component | Technology |
|---|---|
| Web scraping | LangChain WebBaseLoader |
| PDF loading | LangChain PyPDFLoader |
| Text splitting | RecursiveCharacterTextSplitter |
| Embeddings | OpenAI text-embedding-3-small |
| Vector store | ChromaDB |
| LLM | OpenAI GPT-4o-mini |
| Chain | LangChain LCEL |
| API | FastAPI + LangServe |
| Deployment | Docker + Railway |

---

## Architecture Diagram

See [diagram.md](diagram.md) for the full component diagram in Mermaid format, covering both the ingestion pipeline and the query flow.
