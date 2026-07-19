"""
Advanced PDF to PostgreSQL Ingestion with Smart Chunking

Features:
- UnstructuredFileIOLoader for robust PDF parsing
- Regex-based text cleaning (removes artifacts)
- Tiktoken-based chunking (accurate token counting)
- Metadata tracking
- 768-dim BGE embeddings
- PostgreSQL + pgvector vector storage

Usage:
    python generate_embeddings.py
    python generate_embeddings.py --file /path/to/document.pdf
    python generate_embeddings.py --session my-session-id
"""

import os
import sys
import uuid
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import RAG components
from src.rag_qa.core.document_processor import FileIoDataLoader
from src.rag_qa.core.database import (
    init_db, check_connection, get_db_session,
    insert_chunks, DBSession, DBDocument
)
from src.rag_qa.utils.config_loader import (
    get_text_splitter_config,
    get_embeddings_config,
    get_database_config
)

# --- CONFIGURATION FROM config/config.yml ---
embeddings_config = get_embeddings_config()
text_splitter_config = get_text_splitter_config()

# OpenAI Embeddings (using text-embedding-3-small)
# EMBEDDING_MODEL_NAME = "text-embedding-3-small"
# EMBEDDING_DIMENSIONS = 1536

# Ollama Embeddings
EMBEDDING_MODEL_NAME = "nomic-embed-text"
EMBEDDING_DIMENSIONS = 768

# Text Splitter Configuration (from config)
CHUNK_SIZE = text_splitter_config['chunk_size']
CHUNK_OVERLAP = text_splitter_config['chunk_overlap']
SEPARATORS = text_splitter_config['separators']


def ingest_document_to_postgresql(file_path: str, session_id: str = None):
    """
    Orchestrates the RAG ingestion pipeline:
    - Load document (PDF, DOCX, TXT, MD)
    - Regex-based text cleaning  
    - Tiktoken-based chunking
    - Metadata tracking
    - Embed (768-dim BGE)
    - Store in PostgreSQL + pgvector
    """
    
    # 1. Initialize Database
    print("1. Initializing PostgreSQL + pgvector...")
    init_db()
    if not check_connection():
        print("❌ Cannot connect to PostgreSQL. Start it with: docker-compose up -d")
        return
    print("✅ Connected to PostgreSQL + pgvector")

    # 2. Create or use session
    if not session_id:
        session_id = str(uuid.uuid4())
    
    db_session = get_db_session()
    try:
        existing = db_session.query(DBSession).filter(DBSession.id == session_id).first()
        if not existing:
            new_session = DBSession(id=session_id)
            db_session.add(new_session)
            db_session.commit()
            print(f"   Created session: {session_id}")
        else:
            print(f"   Using existing session: {session_id}")
    except Exception as e:
        db_session.rollback()
        print(f"❌ Error creating session: {e}")
        return
    finally:
        db_session.close()

    # 3. Load and Process Document
    print(f"\n2. Loading document from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")
    
    print(f"3. Processing with advanced chunking (Tiktoken-based)...")
    print(f"   Config: {CHUNK_SIZE} tokens, {CHUNK_OVERLAP} overlap")
    file_loader = FileIoDataLoader(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS
    )
    
    with open(file_path, 'rb') as file:
        filename = os.path.basename(file_path)
        
        chunks = file_loader.scrap_and_create_documents_for_file_data(
            file_path=file_path,
            file_name=filename,
        )
    
    print(f"✅ Created {len(chunks)} clean chunks with metadata")

    # 4. Generate Embeddings
    print(f"4. Generating Ollama embeddings ({EMBEDDING_DIMENSIONS} dimensions)...")
    from langchain_ollama import OllamaEmbeddings
    bge_embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL_NAME
    )
    
    chunk_contents = [chunk.page_content for chunk in chunks]
    chunk_metadatas = [chunk.metadata for chunk in chunks]
    
    print(f"   Embedding {len(chunk_contents)} chunks...")
    chunk_vectors = bge_embeddings.embed_documents(chunk_contents)
    print(f"✅ Generated {len(chunk_vectors)} embeddings")
    
    # 5. Store in PostgreSQL + pgvector
    print(f"5. Inserting {len(chunks)} chunks into PostgreSQL...")
    
    inserted = insert_chunks(
        session_id=session_id,
        contents=chunk_contents,
        embeddings=chunk_vectors,
        metadatas=chunk_metadatas
    )
    
    # Store document metadata
    doc_id = str(uuid.uuid4())
    db_session = get_db_session()
    try:
        new_doc = DBDocument(
            id=doc_id,
            session_id=session_id,
            filename=filename,
            size=os.path.getsize(file_path),
            chunk_count=len(chunks)
        )
        db_session.add(new_doc)
        db_session.commit()
    except Exception as e:
        db_session.rollback()
        print(f"⚠️ Warning: Could not save document metadata: {e}")
    finally:
        db_session.close()

    print("-" * 60)
    print("✅ Ingestion complete!")
    print(f"📊 Statistics:")
    print(f"   • Session ID: {session_id}")
    print(f"   • Total chunks: {inserted}")
    print(f"   • Chunk size: {CHUNK_SIZE} tokens (tiktoken-based)")
    print(f"   • Chunk overlap: {CHUNK_OVERLAP} tokens")
    print(f"   • Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"   • Vector dimensions: {EMBEDDING_DIMENSIONS}")
    print(f"   • Storage: PostgreSQL + pgvector")
    print(f"\n💡 Each chunk includes:")
    print(f"   ✓ Cleaned text (regex-based)")
    print(f"   ✓ Metadata (content_type, file_name, timestamp, source)")
    print(f"   ✓ 768-dim vector embedding")
    print(f"   ✓ HNSW index for fast ANN search")
    print("-" * 60)
    
    # Show sample metadata
    if chunks:
        print(f"\n📝 Sample chunk metadata:")
        sample_metadata = chunks[0].metadata
        for key, value in sample_metadata.items():
            print(f"   {key}: {value}")
        print()
    
    print(f"\n🚀 Use this session ID to query: {session_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into PostgreSQL + pgvector")
    parser.add_argument("--file", "-f", required=True, help="Path to the document file")
    parser.add_argument("--session", "-s", default=None, help="Session ID (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    try:
        ingest_document_to_postgresql(args.file, args.session)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
    except Exception as e:
        print(f"A fatal error occurred: {e}")