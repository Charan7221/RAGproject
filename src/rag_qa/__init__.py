"""
Data-Driven Document QA Using RAG and Vector Embeddings

A comprehensive Retrieval-Augmented Generation (RAG) system using:
- Hugging Face transformers for embeddings
- PostgreSQL + pgvector for vector storage
- LangChain for RAG pipeline orchestration
- OpenAI for answer generation
- Support for PDF, DOCX, TXT, and Markdown documents
"""

__version__ = "2.0.0"
__author__ = "RAG QA Team"

from .core.rag_openai import RAGWithOpenAI
from .core.pg_retriever import PgVectorRetriever, PgHybridRetriever
from .core.document_processor import DocumentProcessor
from .core.database import init_db, check_connection

__all__ = [
    "RAGWithOpenAI",
    "PgVectorRetriever",
    "PgHybridRetriever",
    "DocumentProcessor",
    "init_db",
    "check_connection"
]