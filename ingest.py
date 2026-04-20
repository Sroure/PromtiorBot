"""
ingest.py — Builds the ChromaDB vector store from web pages and a local PDF.

Run this once before starting the server:
    python ingest.py
"""

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

URLS = [
    "https://www.promtior.ai/",
    "https://www.promtior.ai/service",
    "https://www.promtior.ai/use-cases",
]

PDF_PATH = os.getenv("PDF_PATH", "./doc/data.pdf")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_web_documents():
    """Scrape the Promtior website pages."""
    print("Loading web pages...")
    loader = WebBaseLoader(web_paths=URLS)
    docs = loader.load()
    print(f"  Loaded {len(docs)} web page(s)")
    return docs


def load_pdf_documents():
    """Load the local PDF file."""
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"PDF not found at '{PDF_PATH}'. "
            "Place the file there or set the PDF_PATH environment variable."
        )
    print(f"Loading PDF from {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    print(f"  Loaded {len(docs)} page(s) from PDF")
    return docs

# ── Main ingest pipeline ───────────────────────────────────────────────────────

def ingest():
    # 1. Load all documents
    documents = load_web_documents() + load_pdf_documents()

    # 2. Split into chunks
    # chunk_size=1000: each chunk is ~1000 characters
    # chunk_overlap=200: chunks share 200 chars with the next one so we don't
    #                    lose context at the boundaries
    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"  Created {len(chunks)} chunks")

    # 3. Embed and store in ChromaDB
    # OpenAIEmbeddings converts each chunk into a vector (list of numbers)
    # that captures its semantic meaning.
    print("Embedding chunks and saving to ChromaDB...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )
    print(f"  Done! Vector store saved to '{CHROMA_DB_PATH}'")
    return vectorstore


if __name__ == "__main__":
    ingest()
