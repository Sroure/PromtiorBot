"""
main.py — FastAPI + LangServe server exposing the RAG chain at POST /chat/invoke

Start the server:
    python main.py
"""

import os
from dotenv import load_dotenv

from fastapi import FastAPI
from langserve import add_routes

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

# ── Vector store ───────────────────────────────────────────────────────────────

def load_vectorstore():
    """Load the existing ChromaDB. Run ingest.py first if it doesn't exist."""
    if not os.path.exists(CHROMA_DB_PATH):
        print("ChromaDB not found — running ingest first...")
        from ingest import ingest
        ingest()

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
    )

# ── RAG chain (LCEL) ───────────────────────────────────────────────────────────

def build_chain(vectorstore):
    """
    Build the RAG chain using LangChain Expression Language (LCEL).

    Flow:
        user question
            └─► retriever  →  top-5 relevant chunks (context)
            └─► passthrough →  original question
        both go into the prompt template
            └─► ChatOpenAI (gpt-4o-mini)
                └─► StrOutputParser  →  plain text answer
    """
    # Retriever: given a query, finds the k most similar chunks in ChromaDB
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    def format_docs(docs):
        """Join retrieved chunks into a single context string."""
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant for Promtior, a Generative AI consulting company.
Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say you don't have that information.

Context:
{context}

Question: {question}

Answer:""")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # LCEL chain — reads left to right:
    # 1. The dict runs both branches in parallel:
    #    - "context": retriever receives the question, fetches chunks, format_docs joins them
    #    - "question": RunnablePassthrough just forwards the original question unchanged
    # 2. The combined dict is passed into the prompt template
    # 3. The filled prompt goes to the LLM
    # 4. StrOutputParser extracts the plain text from the LLM response object
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Promtior RAG Chatbot",
    description="RAG-based chatbot answering questions about Promtior AI",
    version="1.0.0",
)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Promtior RAG chatbot is running"}

# Wire up the chain to LangServe.
# add_routes automatically creates:
#   POST /chat/invoke   — single call, returns the answer
#   POST /chat/stream   — streaming response
#   POST /chat/batch    — multiple questions at once
#   GET  /chat/playground — interactive UI (disable in prod if needed)
vectorstore = load_vectorstore()
chain = build_chain(vectorstore)
add_routes(app, chain, path="/chat")

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
