"""
FastAPI Backend for RAG Document QA System
Supports dynamic document upload, querying, streaming, and chat history.
Uses PostgreSQL + pgvector for vector storage.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import sys
import uuid
import json
import tempfile
import asyncio
import re
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rag_qa import RAGWithOpenAI, PgVectorRetriever, PgHybridRetriever
from src.rag_qa.core.document_processor import FileIoDataLoader
from src.rag_qa.core.database import (
    init_db, check_connection, get_db_session,
    DBSession, DBDocument, DBDocumentChunk, insert_chunks,
    get_chat_history, save_chat_message
)
from src.rag_qa.utils.config_loader import (
    get_database_config,
    get_embeddings_config,
    get_text_splitter_config,
    get_retrieval_config,
    get_conversation_config,
    get_llm_config
)
# from langchain_community.embeddings import HuggingFaceEmbeddings

# Import logging configuration
from backend.logging_config import get_logger, setup_logging
from backend.middleware import RequestLoggingMiddleware

# Setup logging
logger = setup_logging(
    log_file="logs/rag_api.log",
    log_level="DEBUG",
    console_output=True
)

# Initialize FastAPI
app = FastAPI(
    title="RAG Document QA API",
    description="Dynamic document upload and intelligent Q&A system with PostgreSQL + pgvector",
    version="2.0.0"
)

# Add Request Logging Middleware (MUST be before CORS)
app.add_middleware(RequestLoggingMiddleware)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("=" * 80)
logger.info("RAG API Server Starting (v2.0 - PostgreSQL + pgvector)...")
logger.info("=" * 80)

# Global shared resources
_embeddings = None
_rag_instances = {}  # session_id -> RAGWithOpenAI instance
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_RAG_CACHE_SIZE = 50


# Request/Response Models
class QueryRequest(BaseModel):
    question: str
    session_id: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    token_usage: Optional[dict] = None


class DocumentInfo(BaseModel):
    id: str
    filename: str
    size: int
    chunks: int
    uploaded_at: str


class SessionResponse(BaseModel):
    session_id: str
    documents: List[DocumentInfo]
    total_chunks: int


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning(
            "GOOGLE_API_KEY / GEMINI_API_KEY not set — "
            "embeddings and LLM calls will fail at runtime"
        )

    logger.info("Initializing database...")
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        raise


# Helper Functions
def get_embeddings():
    """Get or initialize shared embeddings model."""
    global _embeddings
    if _embeddings is None:
        embeddings_config = get_embeddings_config()
        model_name = embeddings_config.get('sbert_model', 'models/gemini-embedding-001')
        from src.rag_qa.core.database import EMBEDDING_DIMENSIONS
        
        # Check embedding model type
        if model_name.startswith('text-embedding'):
            # OpenAI embeddings
            logger.info(f"Loading OpenAI embeddings model: {model_name} ({EMBEDDING_DIMENSIONS} dimensions)")
            from langchain_openai import OpenAIEmbeddings
            import os
            _embeddings = OpenAIEmbeddings(
                model=model_name,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif 'nomic' in model_name or 'embed' in model_name:
            # Ollama embeddings
            logger.info(f"Loading Ollama embeddings model: {model_name}")
            from langchain_ollama import OllamaEmbeddings
            _embeddings = OllamaEmbeddings(
                model=model_name,
                base_url="http://localhost:11434"
            )
        else:
            # Gemini embeddings
            logger.info(f"Loading Google Gemini embeddings model: {model_name} ({EMBEDDING_DIMENSIONS} dimensions)")
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            _embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                task_type="retrieval_document",
                output_dimensionality=EMBEDDING_DIMENSIONS
            )
    return _embeddings


def get_rag_for_session(session_id: str) -> RAGWithOpenAI:
    """Get or create RAG instance for a session."""
    if session_id not in _rag_instances:
        # Evict oldest cached instance if cache is full
        if len(_rag_instances) >= MAX_RAG_CACHE_SIZE:
            oldest_key = next(iter(_rag_instances))
            del _rag_instances[oldest_key]
            logger.debug(f"Evicted RAG cache entry for session {oldest_key}")

        retrieval_config = get_retrieval_config()
        hybrid_config = retrieval_config.get('hybrid_search', {})
        
        # Use hybrid retriever if enabled, otherwise semantic-only
        if hybrid_config.get('enabled', True):
            retriever = PgHybridRetriever(
                embedding_function=get_embeddings(),
                session_id=session_id,
                top_k=retrieval_config.get('top_k', 5),
                top_k_initial=retrieval_config.get('top_k_initial', 20),
                alpha=hybrid_config.get('alpha', 0.5),
                rrf_k=hybrid_config.get('rrf_k', 60),
                use_rrf=hybrid_config.get('use_rrf', True)
            )
        else:
            retriever = PgVectorRetriever(
                embedding_function=get_embeddings(),
                session_id=session_id,
                top_k=retrieval_config.get('top_k', 5)
            )
        
        _rag_instances[session_id] = RAGWithOpenAI(retriever=retriever)
    
    return _rag_instances[session_id]


# API Endpoints

@app.get("/api/config")
async def get_config():
    """Return public configuration for the frontend."""
    llm_config = get_llm_config()
    retrieval_config = get_retrieval_config()
    embeddings_config = get_embeddings_config()
    return {
        "llm_model": llm_config.get("model", "gemini-2.0-flash"),
        "embedding_model": embeddings_config.get("default_model"),
        "top_k": retrieval_config.get("top_k", 5),
        "hybrid_search_enabled": retrieval_config.get("hybrid_search", {}).get("enabled", True),
        "max_upload_size_mb": MAX_UPLOAD_SIZE_BYTES // (1024 * 1024),
        "supported_formats": [".pdf", ".txt", ".docx", ".md"]
    }


@app.get("/")
async def root():
    """Health check endpoint."""
    db_ok = check_connection()
    db_session = get_db_session()
    try:
        session_count = db_session.query(DBSession).count()
    except Exception:
        session_count = 0
    finally:
        db_session.close()
    
    return {
        "status": "online",
        "service": "RAG Document QA API",
        "version": "2.0.0",
        "database": "connected" if db_ok else "disconnected",
        "active_sessions": session_count
    }


@app.post("/api/session/create")
async def create_session():
    """Create a new session (persisted in PostgreSQL)."""
    session_id = str(uuid.uuid4())
    
    db_session = get_db_session()
    try:
        new_session = DBSession(id=session_id)
        db_session.add(new_session)
        db_session.commit()
        logger.info(f"New session created: {session_id}")
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating session: {e}")
    finally:
        db_session.close()
    
    return {
        "session_id": session_id,
        "message": "Session created successfully"
    }


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = None
):
    """Upload and process a document."""
    
    # Create session if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
        db_session = get_db_session()
        try:
            new_session = DBSession(id=session_id)
            db_session.add(new_session)
            db_session.commit()
        except Exception:
            db_session.rollback()
        finally:
            db_session.close()
    
    logger.info(f"Document upload started: {file.filename}")
    
    # Validate file type
    allowed_extensions = ['.pdf', '.txt', '.docx', '.md']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )

    tmp_file_path = None
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()

            if len(content) > MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB"
                )

            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        logger.info(f"File saved to temp: {tmp_file_path} ({len(content)} bytes)")

        # Create document record early so chunks can reference it
        doc_id = str(uuid.uuid4())
        db_session = get_db_session()
        try:
            new_doc = DBDocument(
                id=doc_id,
                session_id=session_id,
                filename=file.filename,
                size=len(content),
                chunk_count=0
            )
            db_session.add(new_doc)
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()
        
        # Process document
        text_splitter_config = get_text_splitter_config()
        file_loader = FileIoDataLoader(
            chunk_size=text_splitter_config['chunk_size'],
            chunk_overlap=text_splitter_config['chunk_overlap'],
            separators=text_splitter_config['separators']
        )
        
        chunks = file_loader.scrap_and_create_documents_for_file_data(
            file_path=tmp_file_path,
            file_name=file.filename
        )
        
        logger.info(f"Document processed into {len(chunks)} chunks")
        
        # Generate embeddings in batches for large documents
        embeddings_model = get_embeddings()
        chunk_contents = [chunk.page_content for chunk in chunks]
        chunk_metadatas = [chunk.metadata for chunk in chunks]
        
        BATCH_SIZE = 5  # Smaller batches to avoid rate limits
        total_chunks = len(chunk_contents)
        total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE
        
        logger.info(f"Starting embedding generation: {total_chunks} chunks in {total_batches} batches")
        
        for batch_idx in range(total_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, total_chunks)
            
            batch_contents = chunk_contents[start:end]
            batch_metadatas = chunk_metadatas[start:end]
            
            # Generate embeddings for this batch with retry logic
            MAX_RETRIES = 15
            batch_embeddings = None
            for attempt in range(MAX_RETRIES):
                try:
                    batch_embeddings = embeddings_model.embed_documents(batch_contents)
                    break
                except Exception as e:
                    error_msg = str(e)
                    if '429' in error_msg or 'RESOURCE_EXHAUSTED' in error_msg:
                        # Try to parse the required retry delay
                        retry_match = re.search(r'Please retry in ([\d\.]+)s', error_msg)
                        if retry_match:
                            delay = float(retry_match.group(1)) + 3.0  # Add 3s buffer
                        else:
                            delay = (2 ** attempt) * 8  # Exponential backoff: 8, 16, 32, 64...
                        
                        logger.warning(f"Rate limit hit on batch {batch_idx+1}/{total_batches}. Retrying in {delay:.1f}s (Attempt {attempt+1}/{MAX_RETRIES}).")
                        await asyncio.sleep(delay)
                    else:
                        raise
            
            if batch_embeddings is None:
                raise Exception(f"Failed to generate embeddings after {MAX_RETRIES} attempts due to rate limits.")
            
            logger.info(f"Embedding batch {batch_idx + 1}/{total_batches} done ({end}/{total_chunks} chunks)")
            
            # Insert this batch into PostgreSQL immediately
            insert_chunks(
                session_id=session_id,
                contents=batch_contents,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                document_id=doc_id
            )
            
            # Mandatory cooldown between batches to avoid hitting rate limits
            if batch_idx < total_batches - 1:
                await asyncio.sleep(15)
        
        logger.info(f"All {total_chunks} embeddings generated and stored")
        
        # Update document chunk count
        db_session = get_db_session()
        try:
            doc = db_session.query(DBDocument).filter(DBDocument.id == doc_id).first()
            if doc:
                doc.chunk_count = len(chunks)
                db_session.commit()
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()
        
        # Invalidate cached RAG instance (new docs = new retrieval scope)
        if session_id in _rag_instances:
            del _rag_instances[session_id]
        
        doc_info = {
            "id": doc_id,
            "filename": file.filename,
            "size": len(content),
            "chunks": len(chunks),
            "uploaded_at": datetime.now().isoformat()
        }
        
        # Get total chunk count for session
        db_session = get_db_session()
        try:
            total_chunks = db_session.query(DBDocumentChunk).filter(
                DBDocumentChunk.session_id == session_id
            ).count()
            total_docs = db_session.query(DBDocument).filter(
                DBDocument.session_id == session_id
            ).count()
        finally:
            db_session.close()
        
        logger.info(f"Document upload completed: {file.filename}")
        
        return {
            "session_id": session_id,
            "document": doc_info,
            "message": f"Successfully processed {file.filename}",
            "total_documents": total_docs,
            "total_chunks": total_chunks
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing document. Check server logs for details.")
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the uploaded documents."""
    
    logger.info(f"Query received: '{request.question[:100]}...' (session: {request.session_id})")
    
    # Verify session exists
    db_session = get_db_session()
    try:
        session = db_session.query(DBSession).filter(DBSession.id == request.session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Check if session has documents
        chunk_count = db_session.query(DBDocumentChunk).filter(
            DBDocumentChunk.session_id == request.session_id
        ).count()
        if chunk_count == 0:
            raise HTTPException(status_code=400, detail="No documents uploaded yet")
    finally:
        db_session.close()
    
    try:
        rag = get_rag_for_session(request.session_id)
        
        # Load chat history for conversation memory
        conv_config = get_conversation_config()
        chat_history = get_chat_history(
            request.session_id,
            max_turns=conv_config.get('max_history_turns', 5)
        )
        
        # Save user message to history
        save_chat_message(request.session_id, 'user', request.question)
        
        # Get answer with chat history context
        result = rag.answer_question(request.question, chat_history=chat_history)
        
        # Save assistant response to history
        save_chat_message(
            request.session_id, 'assistant', result['answer'],
            sources=result.get('source_documents'),
            token_usage=result.get('token_usage')
        )
        
        logger.info(
            f"Query processed | Answer: {len(result['answer'])} chars | "
            f"Sources: {len(result['source_documents'])}"
        )
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["source_documents"],
            token_usage=result.get("token_usage")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/api/query/stream")
async def query_documents_stream(request: QueryRequest):
    """Stream the answer token-by-token using Server-Sent Events (SSE)."""
    
    logger.info(f"Stream query received: '{request.question[:100]}...' (session: {request.session_id})")
    
    # Verify session exists and has documents
    db_session = get_db_session()
    try:
        session = db_session.query(DBSession).filter(DBSession.id == request.session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        chunk_count = db_session.query(DBDocumentChunk).filter(
            DBDocumentChunk.session_id == request.session_id
        ).count()
        if chunk_count == 0:
            raise HTTPException(status_code=400, detail="No documents uploaded yet")
    finally:
        db_session.close()
    
    async def event_generator():
        """Generate SSE events from RAG streaming."""
        try:
            rag = get_rag_for_session(request.session_id)
            
            # Load chat history
            conv_config = get_conversation_config()
            chat_history = get_chat_history(
                request.session_id,
                max_turns=conv_config.get('max_history_turns', 5)
            )
            
            # Save user message
            save_chat_message(request.session_id, 'user', request.question)
            
            full_answer = ""
            sources = []
            
            async for event in rag.answer_question_stream(request.question, chat_history=chat_history):
                if event["type"] == "token":
                    full_answer += event["content"]
                    yield f"data: {json.dumps(event)}\n\n"
                elif event["type"] == "sources":
                    sources = event.get("sources", [])
                    yield f"data: {json.dumps(event)}\n\n"
                elif event["type"] == "done":
                    yield f"data: {json.dumps(event)}\n\n"
                elif event["type"] == "error":
                    yield f"data: {json.dumps(event)}\n\n"
            
            # Save assistant response to history
            if full_answer:
                save_chat_message(
                    request.session_id, 'assistant', full_answer,
                    sources=sources
                )
            
        except Exception as e:
            logger.error(f"Error in stream: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/session/{session_id}", response_model=SessionResponse)
async def get_session_info(session_id: str):
    """Get session information."""
    
    db_session = get_db_session()
    try:
        session = db_session.query(DBSession).filter(DBSession.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        documents = db_session.query(DBDocument).filter(
            DBDocument.session_id == session_id
        ).all()
        
        total_chunks = db_session.query(DBDocumentChunk).filter(
            DBDocumentChunk.session_id == session_id
        ).count()
        
        doc_list = [
            DocumentInfo(
                id=doc.id,
                filename=doc.filename,
                size=doc.size,
                chunks=doc.chunk_count,
                uploaded_at=doc.uploaded_at.isoformat()
            )
            for doc in documents
        ]
        
        return SessionResponse(
            session_id=session_id,
            documents=doc_list,
            total_chunks=total_chunks
        )
    finally:
        db_session.close()


@app.get("/api/session/{session_id}/history")
async def get_session_history(session_id: str, limit: int = 50):
    """Get chat history for a session."""
    
    db_session = get_db_session()
    try:
        session = db_session.query(DBSession).filter(DBSession.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        history = get_chat_history(session_id, max_turns=limit)
        return {"session_id": session_id, "history": history}
    finally:
        db_session.close()


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all associated data (cascading delete)."""
    
    logger.info(f"Session deletion requested: {session_id}")
    
    db_session = get_db_session()
    try:
        session = db_session.query(DBSession).filter(DBSession.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Cascading delete removes documents, chunks, and chat history
        db_session.delete(session)
        db_session.commit()
        
        # Clean up cached RAG instance
        if session_id in _rag_instances:
            del _rag_instances[session_id]
        
        logger.info(f"Session deleted successfully: {session_id}")
        return {"message": "Session deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting session: {e}")
    finally:
        db_session.close()


@app.delete("/api/session/{session_id}/documents/{document_id}")
async def delete_document(session_id: str, document_id: str):
    """Delete a specific document from session."""
    
    logger.info(f"Document deletion | Session: {session_id} | Document: {document_id}")
    
    db_session = get_db_session()
    try:
        doc = db_session.query(DBDocument).filter(
            DBDocument.id == document_id,
            DBDocument.session_id == session_id
        ).first()
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        filename = doc.filename
        
        # Delete associated chunks by document_id (falls back to filename for legacy rows)
        deleted = db_session.query(DBDocumentChunk).filter(
            DBDocumentChunk.document_id == document_id
        ).delete()
        if deleted == 0:
            db_session.query(DBDocumentChunk).filter(
                DBDocumentChunk.session_id == session_id,
                DBDocumentChunk.file_name == filename
            ).delete()
        
        # Delete document record
        db_session.delete(doc)
        db_session.commit()
        
        # Invalidate cached RAG instance
        if session_id in _rag_instances:
            del _rag_instances[session_id]
        
        logger.info(f"Document removed: {filename}")
        return {"message": f"Document {filename} removed"}
    except HTTPException:
        raise
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {e}")
    finally:
        db_session.close()


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Uvicorn server on http://0.0.0.0:8000")
    logger.info("Press CTRL+C to quit")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
