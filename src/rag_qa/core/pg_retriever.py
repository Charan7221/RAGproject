"""
PostgreSQL + pgvector Retriever

Replaces Elasticsearch retrievers with PostgreSQL-based retrieval:
- PgVectorRetriever: Semantic search using pgvector cosine similarity
- PgHybridRetriever: Combines pgvector semantic + PostgreSQL full-text (tsvector) search
  with Reciprocal Rank Fusion (RRF)
"""

import logging
from typing import List, Any, Optional, Dict, Tuple

from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from sqlalchemy import text as sql_text

from .database import get_db_session

logger = logging.getLogger(__name__)


class PgVectorRetriever(BaseRetriever):
    """
    Retriever using pgvector cosine similarity search.
    
    Uses the HNSW index for fast approximate nearest neighbor search.
    
    Attributes:
        embedding_function: Function to generate query embeddings
        session_id: Session ID to scope the search
        top_k: Number of results to return
        metadata_filter: Optional JSONB filter (e.g., {"file_name": "report.pdf"})
    """
    
    embedding_function: Any
    session_id: str
    top_k: int = 5
    metadata_filter: Optional[Dict[str, Any]] = None
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """
        Retrieve documents using pgvector cosine similarity.
        
        Args:
            query: Search query
            
        Returns:
            List of relevant documents
        """
        # Generate query embedding
        query_vector = self.embedding_function.embed_query(query)
        
        db_session = get_db_session()
        try:
            # Build query with optional metadata filter
            filter_clause = ""
            params = {
                "session_id": self.session_id,
                "query_vector": str(query_vector),
                "top_k": self.top_k
            }
            
            if self.metadata_filter:
                filter_clause = "AND metadata @> :meta_filter::jsonb"
                import json
                params["meta_filter"] = json.dumps(self.metadata_filter)
            
            query_sql = sql_text(f"""
                SELECT content, metadata, file_name,
                       1 - (embedding <=> CAST(:query_vector AS vector)) AS similarity_score
                FROM document_chunks
                WHERE session_id = :session_id
                {filter_clause}
                ORDER BY embedding <=> CAST(:query_vector AS vector)
                LIMIT :top_k
            """)
            
            result = db_session.execute(query_sql, params)
            
            documents = []
            for row in result:
                metadata = row.metadata or {}
                metadata['similarity_score'] = float(row.similarity_score)
                metadata['retrieval_method'] = 'semantic'
                if row.file_name:
                    metadata['file_name'] = row.file_name
                
                doc = Document(
                    page_content=row.content,
                    metadata=metadata
                )
                documents.append(doc)
            
            logger.info(f"Semantic retrieval returned {len(documents)} documents for query: {query[:50]}...")
            return documents
            
        except Exception as e:
            logger.error(f"Error in semantic retrieval: {e}")
            return []
        finally:
            db_session.close()
    
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Async version — delegates to sync."""
        return self._get_relevant_documents(query, run_manager=run_manager)


class PgHybridRetriever(BaseRetriever):
    """
    Hybrid Retriever combining PostgreSQL full-text search (tsvector/ts_rank)
    with pgvector semantic search using Reciprocal Rank Fusion (RRF).
    
    RRF score = sum(1 / (k + rank)) where k is a constant (default 60).
    
    Attributes:
        embedding_function: Function to generate query embeddings
        session_id: Session ID to scope the search
        top_k: Number of final results to return
        top_k_initial: Number of results to fetch from each search method
        alpha: Weight for semantic search (not used with RRF, used with weighted fusion)
        rrf_k: RRF constant (default 60)
        use_rrf: If True, use RRF fusion. If False, use weighted score fusion.
        metadata_filter: Optional JSONB filter
    """
    
    embedding_function: Any
    session_id: str
    top_k: int = 5
    top_k_initial: int = 20
    alpha: float = 0.5
    rrf_k: int = 60
    use_rrf: bool = True
    metadata_filter: Optional[Dict[str, Any]] = None
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
    
    def _semantic_search(self, query_vector: List[float]) -> List[Tuple[int, float, str, dict, str]]:
        """
        Perform semantic (vector) search using pgvector.
        
        Returns:
            List of (chunk_id, score, content, metadata, file_name) tuples
        """
        db_session = get_db_session()
        try:
            filter_clause = ""
            params = {
                "session_id": self.session_id,
                "query_vector": str(query_vector),
                "limit": self.top_k_initial
            }
            
            if self.metadata_filter:
                filter_clause = "AND metadata @> :meta_filter::jsonb"
                import json
                params["meta_filter"] = json.dumps(self.metadata_filter)
            
            query = sql_text(f"""
                SELECT id, content, metadata, file_name,
                       1 - (embedding <=> CAST(:query_vector AS vector)) AS similarity_score
                FROM document_chunks
                WHERE session_id = :session_id
                {filter_clause}
                ORDER BY embedding <=> CAST(:query_vector AS vector)
                LIMIT :limit
            """)
            
            result = db_session.execute(query, params)
            results = []
            for row in result:
                results.append((
                    row.id,
                    float(row.similarity_score),
                    row.content,
                    row.metadata or {},
                    row.file_name or 'Unknown'
                ))
            return results
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []
        finally:
            db_session.close()
    
    def _fulltext_search(self, query: str) -> List[Tuple[int, float, str, dict, str]]:
        """
        Perform full-text search using PostgreSQL tsvector/ts_rank.
        
        Returns:
            List of (chunk_id, score, content, metadata, file_name) tuples
        """
        db_session = get_db_session()
        try:
            filter_clause = ""
            params = {
                "session_id": self.session_id,
                "query": query,
                "limit": self.top_k_initial
            }
            
            if self.metadata_filter:
                filter_clause = "AND metadata @> :meta_filter::jsonb"
                import json
                params["meta_filter"] = json.dumps(self.metadata_filter)
            
            query_sql = sql_text(f"""
                SELECT id, content, metadata, file_name,
                       ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', :query)) AS rank_score
                FROM document_chunks
                WHERE session_id = :session_id
                AND to_tsvector('english', content) @@ plainto_tsquery('english', :query)
                {filter_clause}
                ORDER BY rank_score DESC
                LIMIT :limit
            """)
            
            result = db_session.execute(query_sql, params)
            results = []
            for row in result:
                results.append((
                    row.id,
                    float(row.rank_score),
                    row.content,
                    row.metadata or {},
                    row.file_name or 'Unknown'
                ))
            return results
        except Exception as e:
            logger.error(f"Full-text search error: {e}")
            return []
        finally:
            db_session.close()
    
    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Tuple],
        fulltext_results: List[Tuple]
    ) -> List[Tuple[int, float, str, dict, str]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF score = sum(1 / (k + rank)) for each ranking list
        """
        doc_scores = {}  # chunk_id -> (rrf_score, content, metadata, file_name)
        
        # Process semantic results
        for rank, (chunk_id, _, content, metadata, file_name) in enumerate(semantic_results, 1):
            rrf_score = 1.0 / (self.rrf_k + rank)
            if chunk_id in doc_scores:
                old_score, _, _, _ = doc_scores[chunk_id]
                doc_scores[chunk_id] = (old_score + rrf_score, content, metadata, file_name)
            else:
                doc_scores[chunk_id] = (rrf_score, content, metadata, file_name)
        
        # Process full-text results
        for rank, (chunk_id, _, content, metadata, file_name) in enumerate(fulltext_results, 1):
            rrf_score = 1.0 / (self.rrf_k + rank)
            if chunk_id in doc_scores:
                old_score, _, _, _ = doc_scores[chunk_id]
                doc_scores[chunk_id] = (old_score + rrf_score, content, metadata, file_name)
            else:
                doc_scores[chunk_id] = (rrf_score, content, metadata, file_name)
        
        # Sort by RRF score (descending)
        sorted_results = sorted(
            [
                (chunk_id, score, content, metadata, file_name)
                for chunk_id, (score, content, metadata, file_name) in doc_scores.items()
            ],
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results
    
    def _weighted_score_fusion(
        self,
        semantic_results: List[Tuple],
        fulltext_results: List[Tuple]
    ) -> List[Tuple[int, float, str, dict, str]]:
        """
        Combine results using weighted score fusion.
        
        Combined score = alpha * norm_semantic + (1-alpha) * norm_fulltext
        """
        doc_scores = {}
        
        # Normalize semantic scores (min-max)
        if semantic_results:
            sem_scores = [s for _, s, _, _, _ in semantic_results]
            sem_min, sem_max = min(sem_scores), max(sem_scores)
            sem_range = sem_max - sem_min if sem_max != sem_min else 1.0
            
            for chunk_id, score, content, metadata, file_name in semantic_results:
                norm_score = (score - sem_min) / sem_range
                doc_scores[chunk_id] = {
                    'semantic': norm_score, 'fulltext': 0.0,
                    'content': content, 'metadata': metadata, 'file_name': file_name
                }
        
        # Normalize full-text scores
        if fulltext_results:
            ft_scores = [s for _, s, _, _, _ in fulltext_results]
            ft_min, ft_max = min(ft_scores), max(ft_scores)
            ft_range = ft_max - ft_min if ft_max != ft_min else 1.0
            
            for chunk_id, score, content, metadata, file_name in fulltext_results:
                norm_score = (score - ft_min) / ft_range
                if chunk_id in doc_scores:
                    doc_scores[chunk_id]['fulltext'] = norm_score
                else:
                    doc_scores[chunk_id] = {
                        'semantic': 0.0, 'fulltext': norm_score,
                        'content': content, 'metadata': metadata, 'file_name': file_name
                    }
        
        # Calculate weighted combined scores
        combined = []
        for chunk_id, scores in doc_scores.items():
            combined_score = self.alpha * scores['semantic'] + (1 - self.alpha) * scores['fulltext']
            combined.append((
                chunk_id, combined_score,
                scores['content'], scores['metadata'], scores['file_name']
            ))
        
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """
        Retrieve documents using hybrid search (full-text + semantic).
        
        Args:
            query: Search query
            
        Returns:
            List of relevant documents
        """
        # Generate query embedding
        query_vector = self.embedding_function.embed_query(query)
        
        # Perform both searches
        semantic_results = self._semantic_search(query_vector)
        fulltext_results = self._fulltext_search(query)
        
        logger.debug(f"Semantic search returned {len(semantic_results)} results")
        logger.debug(f"Full-text search returned {len(fulltext_results)} results")
        
        # If full-text returns nothing (e.g., very short query), fall back to semantic only
        if not fulltext_results:
            fused_results = [(cid, score, content, meta, fn) 
                           for cid, score, content, meta, fn in semantic_results]
        elif not semantic_results:
            fused_results = [(cid, score, content, meta, fn)
                           for cid, score, content, meta, fn in fulltext_results]
        elif self.use_rrf:
            fused_results = self._reciprocal_rank_fusion(semantic_results, fulltext_results)
            logger.debug("Using Reciprocal Rank Fusion (RRF)")
        else:
            fused_results = self._weighted_score_fusion(semantic_results, fulltext_results)
            logger.debug(f"Using Weighted Score Fusion (alpha={self.alpha})")
        
        # Convert to LangChain documents
        documents = []
        for chunk_id, score, content, metadata, file_name in fused_results[:self.top_k_initial]:
            meta = metadata.copy() if metadata else {}
            meta['fusion_score'] = score
            meta['retrieval_method'] = 'hybrid'
            meta['file_name'] = file_name
            
            doc = Document(page_content=content, metadata=meta)
            documents.append(doc)
        
        logger.info(f"Hybrid retrieval returned {len(documents)} documents for query: {query[:50]}...")
        return documents
    
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Async version — delegates to sync."""
        return self._get_relevant_documents(query, run_manager=run_manager)
