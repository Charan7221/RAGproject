"""
Cross-Encoder Reranker for Advanced RAG

Re-ranks retrieved documents using a cross-encoder model for improved precision.
Cross-encoders jointly encode query-document pairs, providing more accurate
relevance scores than bi-encoder similarity.
"""

import logging
from typing import List, Optional, Tuple
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Lazy import for cross-encoder to avoid import errors if not installed
_cross_encoder = None


def get_cross_encoder():
    """Lazy load CrossEncoder to handle import gracefully."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
    return _cross_encoder


class CrossEncoderReranker:
    """
    Re-ranks documents using a cross-encoder model.
    
    Cross-encoders provide more accurate relevance scores by jointly encoding
    the query and document together, allowing for richer interaction between them.
    
    Recommended models:
    - 'cross-encoder/ms-marco-MiniLM-L-6-v2' (fast, good quality)
    - 'cross-encoder/ms-marco-MiniLM-L-12-v2' (balanced)
    - 'BAAI/bge-reranker-base' (high quality)
    - 'BAAI/bge-reranker-large' (highest quality, slower)
    
    Attributes:
        model_name: Name of the cross-encoder model
        top_k: Number of documents to return after reranking
        device: Device to run the model on ('cpu', 'cuda', 'mps')
        batch_size: Batch size for inference
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
        device: str = "cpu",
        batch_size: int = 32
    ):
        """
        Initialize the CrossEncoder reranker.
        
        Args:
            model_name: HuggingFace model name for the cross-encoder
            top_k: Number of documents to return after reranking
            device: Device to run inference on
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.top_k = top_k
        self.device = device
        self.batch_size = batch_size
        self._model = None
        
        logger.info(f"CrossEncoderReranker initialized with model: {model_name}")
    
    @property
    def model(self):
        """Lazy load the model on first use."""
        if self._model is None:
            CrossEncoder = get_cross_encoder()
            self._model = CrossEncoder(
                self.model_name,
                device=self.device
            )
            logger.info(f"Loaded cross-encoder model: {self.model_name}")
        return self._model
    
    def rerank(
        self, 
        query: str, 
        documents: List[Document],
        return_scores: bool = False
    ) -> List[Document]:
        """
        Re-rank documents based on relevance to the query.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            return_scores: If True, add rerank_score to metadata
            
        Returns:
            Reranked list of documents (top_k)
        """
        if not documents:
            logger.warning("No documents to rerank")
            return []
        
        # If fewer documents than top_k, just score and return all
        if len(documents) <= self.top_k:
            logger.debug(f"Only {len(documents)} documents, skipping reranking")
            return documents
        
        try:
            # Prepare query-document pairs for the cross-encoder
            pairs = [(query, doc.page_content) for doc in documents]
            
            # Get relevance scores from cross-encoder
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )
            
            # Pair documents with scores
            doc_score_pairs: List[Tuple[Document, float]] = list(zip(documents, scores))
            
            # Sort by score (descending)
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Take top_k and optionally add scores to metadata
            reranked_docs = []
            for doc, score in doc_score_pairs[:self.top_k]:
                if return_scores:
                    # Create new document with rerank score in metadata
                    new_metadata = doc.metadata.copy()
                    new_metadata['rerank_score'] = float(score)
                    reranked_doc = Document(
                        page_content=doc.page_content,
                        metadata=new_metadata
                    )
                    reranked_docs.append(reranked_doc)
                else:
                    reranked_docs.append(doc)
            
            logger.info(
                f"Reranked {len(documents)} documents, returning top {len(reranked_docs)}. "
                f"Score range: [{doc_score_pairs[-1][1]:.4f}, {doc_score_pairs[0][1]:.4f}]"
            )
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Fallback: return original documents truncated to top_k
            return documents[:self.top_k]
    
    def rerank_with_scores(
        self, 
        query: str, 
        documents: List[Document]
    ) -> List[Tuple[Document, float]]:
        """
        Re-rank documents and return with scores.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            
        Returns:
            List of (Document, score) tuples, sorted by score descending
        """
        if not documents:
            return []
        
        try:
            pairs = [(query, doc.page_content) for doc in documents]
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )
            
            doc_score_pairs = list(zip(documents, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            return [(doc, float(score)) for doc, score in doc_score_pairs[:self.top_k]]
            
        except Exception as e:
            logger.error(f"Error during reranking with scores: {e}")
            return [(doc, 0.0) for doc in documents[:self.top_k]]


class DummyReranker:
    """
    Dummy reranker that returns documents unchanged.
    Used when reranking is disabled or unavailable.
    """
    
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        logger.info("DummyReranker initialized (no reranking)")
    
    def rerank(
        self, 
        query: str, 
        documents: List[Document],
        return_scores: bool = False
    ) -> List[Document]:
        """Return documents unchanged (truncated to top_k)."""
        return documents[:self.top_k]
    
    def rerank_with_scores(
        self, 
        query: str, 
        documents: List[Document]
    ) -> List[Tuple[Document, float]]:
        """Return documents with dummy scores."""
        return [(doc, 1.0) for doc in documents[:self.top_k]]


def create_reranker(
    enabled: bool = True,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 5,
    device: str = "cpu"
) -> CrossEncoderReranker:
    """
    Factory function to create the appropriate reranker.
    
    Args:
        enabled: Whether to enable cross-encoder reranking
        model_name: Model name for the cross-encoder
        top_k: Number of documents to return
        device: Device for inference
        
    Returns:
        CrossEncoderReranker or DummyReranker instance
    """
    if not enabled:
        return DummyReranker(top_k=top_k)
    
    try:
        return CrossEncoderReranker(
            model_name=model_name,
            top_k=top_k,
            device=device
        )
    except Exception as e:
        logger.warning(f"Failed to create CrossEncoderReranker: {e}. Using DummyReranker.")
        return DummyReranker(top_k=top_k)
