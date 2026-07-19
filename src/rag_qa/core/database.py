"""
Database Module - PostgreSQL + pgvector

Provides SQLAlchemy models and database initialization for:
- Document chunks with vector embeddings (pgvector)
- Session management (persistent)
- Chat history (conversation memory)
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Text, DateTime,
    JSON, ForeignKey, Index, text as sql_text
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from sqlalchemy.pool import QueuePool
from pgvector.sqlalchemy import Vector

from ..utils.config_loader import get_database_config

logger = logging.getLogger(__name__)

Base = declarative_base()

# Embedding dimensions for Gemini embedding model
# Gemini models/gemini-embedding-001 natively produces 3072-dimensional vectors,
# but supports Matryoshka truncation to 1536, 768, or 256 dimensions.
# We use 768 dimensions for compatibility with pgvector indexes (≤2000 limit).
EMBEDDING_DIMENSIONS = 768


class DBSession(Base):
    """User session table — persists across server restarts."""
    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    documents = relationship("DBDocument", back_populates="session", cascade="all, delete-orphan")
    chunks = relationship("DBDocumentChunk", back_populates="session", cascade="all, delete-orphan")
    chat_messages = relationship("DBChatMessage", back_populates="session", cascade="all, delete-orphan")


class DBDocument(Base):
    """Uploaded document metadata."""
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True)
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(255), nullable=False)
    size = Column(Integer, nullable=False)
    chunk_count = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    session = relationship("DBSession", back_populates="documents")
    chunks = relationship("DBDocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DBDocumentChunk(Base):
    """Document chunk with vector embedding — core of the RAG retrieval."""
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=True, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIMENSIONS), nullable=False)
    metadata_ = Column("metadata", JSON, default={})
    file_name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    session = relationship("DBSession", back_populates="chunks")
    document = relationship("DBDocument", back_populates="chunks")

    # Indexes for fast retrieval
    __table_args__ = (
        Index("idx_chunks_session", "session_id"),
    )


class DBChatMessage(Base):
    """Chat history for conversation memory."""
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    sources = Column(JSON, default=None)
    token_usage = Column(JSON, default=None)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    session = relationship("DBSession", back_populates="chat_messages")


# ─────────────────────────────────────────────
# Database Engine & Session Factory
# ─────────────────────────────────────────────

_engine = None
_SessionFactory = None


def get_connection_url() -> str:
    """Build PostgreSQL connection URL from config."""
    db_config = get_database_config()
    return (
        f"postgresql://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )


def get_engine():
    """Get or create the SQLAlchemy engine (singleton)."""
    global _engine
    if _engine is None:
        db_config = get_database_config()
        _engine = create_engine(
            get_connection_url(),
            pool_size=db_config.get('pool_size', 5),
            max_overflow=db_config.get('max_overflow', 10),
            poolclass=QueuePool,
            echo=False
        )
    return _engine


def get_session_factory():
    """Get or create the session factory (singleton)."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory


def get_db_session() -> Session:
    """Create a new database session."""
    factory = get_session_factory()
    return factory()


def init_db():
    """
    Initialize the database:
    - Enable pgvector extension
    - Create all tables
    - Create vector similarity index (IVFFlat for >2000 dimensions, HNSW for ≤2000)
    """
    engine = get_engine()

    # Enable pgvector extension
    with engine.connect() as conn:
        conn.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
        logger.info("pgvector extension enabled")

    # Create all tables
    Base.metadata.create_all(engine)
    logger.info("Database tables created successfully")

    # Create appropriate index based on embedding dimensions
    # HNSW is faster but only supports ≤2000 dimensions
    # IVFFlat supports higher dimensions and is still very efficient
    with engine.connect() as conn:
        # Drop existing indexes if they exist (in case of dimension change)
        conn.execute(sql_text("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw"))
        conn.execute(sql_text("DROP INDEX IF EXISTS idx_chunks_embedding_ivfflat"))
        
        if EMBEDDING_DIMENSIONS <= 2000:
            # Use HNSW for lower dimensions (faster, more accurate)
            conn.execute(sql_text(
                "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw "
                "ON document_chunks USING hnsw (embedding vector_cosine_ops)"
            ))
            logger.info(f"Created HNSW index for {EMBEDDING_DIMENSIONS}-dimensional vectors")
        else:
            # Use IVFFlat for higher dimensions (Gemini embeddings = 3072)
            # lists parameter: sqrt(rows) is a good heuristic, start with 100
            conn.execute(sql_text(
                "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivfflat "
                "ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
            ))
            logger.info(f"Created IVFFlat index for {EMBEDDING_DIMENSIONS}-dimensional vectors")
        
        # Add document_id column for existing databases (idempotent migration)
        conn.execute(sql_text(
            "ALTER TABLE document_chunks "
            "ADD COLUMN IF NOT EXISTS document_id VARCHAR(36) "
            "REFERENCES documents(id) ON DELETE CASCADE"
        ))
        conn.commit()
        logger.info("Vector index ensured")

    logger.info("Database initialization complete")


def check_connection() -> bool:
    """Verify database connection is working."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(sql_text("SELECT 1"))
        logger.info("Database connection verified")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def insert_chunks(
    session_id: str,
    contents: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict[str, Any]],
    document_id: Optional[str] = None,
    db_session: Optional[Session] = None
) -> int:
    """
    Batch insert document chunks with embeddings.
    
    Args:
        session_id: Session UUID
        contents: List of text contents
        embeddings: List of embedding vectors
        metadatas: List of metadata dicts
        db_session: Optional existing DB session
        
    Returns:
        Number of chunks inserted
    """
    own_session = db_session is None
    if own_session:
        db_session = get_db_session()

    try:
        chunks = []
        for content, embedding, metadata in zip(contents, embeddings, metadatas):
            chunk = DBDocumentChunk(
                session_id=session_id,
                document_id=document_id,
                content=content,
                embedding=embedding,
                metadata_=metadata,
                file_name=metadata.get('file_name', 'Unknown')
            )
            chunks.append(chunk)

        db_session.add_all(chunks)
        db_session.commit()
        logger.info(f"Inserted {len(chunks)} chunks for session {session_id}")
        return len(chunks)
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error inserting chunks: {e}")
        raise
    finally:
        if own_session:
            db_session.close()


def get_chat_history(session_id: str, max_turns: int = 5) -> List[Dict[str, str]]:
    """
    Retrieve recent chat history for a session.
    
    Args:
        session_id: Session UUID
        max_turns: Maximum number of Q&A turns to retrieve
        
    Returns:
        List of {'role': ..., 'content': ...} dicts
    """
    db_session = get_db_session()
    try:
        messages = (
            db_session.query(DBChatMessage)
            .filter(
                DBChatMessage.session_id == session_id,
                DBChatMessage.role.in_(['user', 'assistant'])
            )
            .order_by(DBChatMessage.created_at.desc())
            .limit(max_turns * 2)  # Each turn has user + assistant
            .all()
        )
        # Reverse to chronological order
        messages.reverse()
        return [{'role': msg.role, 'content': msg.content} for msg in messages]
    finally:
        db_session.close()


def save_chat_message(
    session_id: str,
    role: str,
    content: str,
    sources: Optional[List[Dict]] = None,
    token_usage: Optional[Dict] = None
):
    """Save a chat message to history."""
    db_session = get_db_session()
    try:
        message = DBChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            sources=sources,
            token_usage=token_usage
        )
        db_session.add(message)
        db_session.commit()
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error saving chat message: {e}")
    finally:
        db_session.close()
