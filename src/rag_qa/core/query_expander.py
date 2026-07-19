"""
Query Expansion Module

Uses an LLM to generate alternative phrasings of the user's query
to improve retrieval recall. Multiple query variants are searched
independently and results are deduplicated.
"""

import logging
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from ..utils.config_loader import get_llm_config, get_retrieval_config

logger = logging.getLogger(__name__)


class QueryExpander:
    """
    Generates alternative query phrasings to improve retrieval recall.
    
    For vague or short queries, the LLM produces several semantically 
    similar but differently worded versions. Each variant is searched
    independently, and results are merged + deduplicated.
    """
    
    def __init__(self, num_variants: int = 3, llm: Optional[ChatOpenAI] = None):
        """
        Initialize the query expander.
        
        Args:
            num_variants: Number of query variants to generate
            llm: Optional pre-configured ChatOpenAI instance
        """
        self.num_variants = num_variants
        
        if llm is not None:
            self.llm = llm
        else:
            llm_config = get_llm_config()
            llm_kwargs = {
                'api_key': llm_config.get('api_key', '').strip(),
                'model': llm_config.get('model', 'gpt-4.1-nano'),
                'temperature': 0.7,  # Slightly creative for diverse variants
                'max_tokens': 200
            }
            base_url = llm_config.get('base_url')
            if base_url:
                llm_kwargs['base_url'] = base_url
            
            self.llm = ChatOpenAI(**llm_kwargs)
        
        logger.info(f"QueryExpander initialized (num_variants={num_variants})")
    
    def expand(self, query: str) -> List[str]:
        """
        Generate alternative phrasings of the query.
        
        Args:
            query: Original user query
            
        Returns:
            List of query variants (includes the original)
        """
        try:
            prompt = f"""Generate {self.num_variants} alternative phrasings of this search query.
Each variant should capture the same intent but use different words or angles.
Return ONLY the variants, one per line, no numbering or bullets.

Original query: {query}

Alternative phrasings:"""
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Parse variants from response
            variants = [
                line.strip()
                for line in response.content.strip().split('\n')
                if line.strip() and len(line.strip()) > 5
            ]
            
            # Always include the original query first
            all_queries = [query] + variants[:self.num_variants]
            
            logger.info(f"Query expanded: '{query[:50]}...' → {len(all_queries)} variants")
            return all_queries
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}. Using original query only.")
            return [query]
    
    def retrieve_with_expansion(
        self, 
        query: str, 
        retriever,
        top_k: int = 20
    ) -> List[Document]:
        """
        Expand query and retrieve documents for all variants, then deduplicate.
        
        Args:
            query: Original user query
            retriever: LangChain retriever to use
            top_k: Max results to return after deduplication
            
        Returns:
            Deduplicated list of documents from all query variants
        """
        variants = self.expand(query)
        
        # Collect results from all variants
        seen_contents = set()
        all_docs = []
        
        for variant_query in variants:
            try:
                docs = retriever.invoke(variant_query)
                for doc in docs:
                    # Deduplicate by content hash
                    content_key = hash(doc.page_content[:200])
                    if content_key not in seen_contents:
                        seen_contents.add(content_key)
                        all_docs.append(doc)
            except Exception as e:
                logger.warning(f"Retrieval failed for variant '{variant_query[:50]}...': {e}")
        
        logger.info(
            f"Multi-query retrieval: {len(variants)} variants → "
            f"{len(all_docs)} unique documents (from {len(seen_contents)} seen)"
        )
        
        return all_docs[:top_k]
