"""
RAG System with Google Gemini Generation

Uses PostgreSQL + pgvector for retrieval and Google Gemini API for generation.
Supports advanced RAG features:
- Hybrid Search (Full-text + Semantic via pgvector)
- Cross-Encoder Reranking
- Query Expansion (Multi-Query)
- Streaming Responses
- Conversation Memory

Configuration loaded from config/config.yml.
"""

import logging
from typing import List, Dict, Any, Optional, Generator, AsyncGenerator
import warnings
warnings.filterwarnings("ignore")

# LangChain
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Local imports
from ..utils.config_loader import get_llm_config, get_retrieval_config, get_conversation_config
from .reranker import create_reranker, CrossEncoderReranker, DummyReranker
from .query_expander import QueryExpander

# Setup logging
logger = logging.getLogger(__name__)


class RAGWithOpenAI:
    """
    Advanced RAG system using Google Gemini for generation.
    
    Features:
    - Hybrid Search support (Full-text + Semantic via PgHybridRetriever)
    - Cross-Encoder Reranking for improved precision
    - Query Expansion for better recall
    - Streaming response generation
    - Conversation memory (chat history context)
    - Token usage tracking with reasoning token breakdown
    """
    
    def __init__(self, retriever=None, enable_reranking: Optional[bool] = None,
                 enable_query_expansion: Optional[bool] = None):
        """
        Initialize RAG with OpenAI.
        
        Args:
            retriever: LangChain retriever object (PgVectorRetriever or PgHybridRetriever)
            enable_reranking: Override config to enable/disable reranking. None uses config value.
            enable_query_expansion: Override config to enable/disable query expansion. None uses config value.
        """
        self.retriever = retriever
        self.llm_config = get_llm_config()
        self.retrieval_config = get_retrieval_config()
        self.conversation_config = get_conversation_config()
        
        # Initialize OpenAI LLM
        self._initialize_openai()
        
        # Initialize Reranker
        self._initialize_reranker(enable_reranking)
        
        # Initialize Query Expander
        self._initialize_query_expander(enable_query_expansion)
        
        logger.info("RAG with OpenAI initialized successfully")
    
    def _initialize_openai(self):
        """Initialize Ollama, OpenAI, or Google Gemini LLM."""
        if not self.llm_config.get('use_openai', False):
            logger.warning("use_openai is False, LLM will not be initialized")
            self.llm = None
            return
        
        try:
            model = self.llm_config.get('model', 'gemini-2.0-flash-exp')
            params = self.llm_config.get('params', {})
            base_url = self.llm_config.get('base_url')
            
            # Check model type - PRIORITIZE base_url check first
            if base_url and ('ollama' in base_url.lower() or '11434' in base_url):
                # Ollama model (check for port 11434 as well)
                logger.info(f"Initializing Ollama LLM with model: {model}")
                from langchain_ollama import ChatOllama
                
                self.llm = ChatOllama(
                    model=model,
                    base_url=base_url,
                    temperature=params.get('temperature', 0.1),
                    num_predict=params.get('max_tokens', 2000)
                )
            elif model.startswith('gpt'):
                # OpenAI model
                logger.info(f"Initializing OpenAI LLM with model: {model}")
                from langchain_openai import ChatOpenAI
                import os
                
                self.llm = ChatOpenAI(
                    model=model,
                    temperature=params.get('temperature', 0.1),
                    max_tokens=params.get('max_tokens', 2000),
                    api_key=os.getenv("OPENAI_API_KEY")
                )
            else:
                # Gemini model - add 'models/' prefix if not present
                if not model.startswith('models/'):
                    model = f'models/{model}'
                
                logger.info(f"Initializing Google Gemini LLM with model: {model}")
                llm_kwargs = {
                    'model': model,
                    'temperature': params.get('temperature', 0.1),
                    'max_output_tokens': params.get('max_tokens', 2000)
                }
                
                self.llm = ChatGoogleGenerativeAI(**llm_kwargs)
            
            logger.info(f"LLM initialized successfully: {model}")
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            self.llm = None
    
    def _initialize_reranker(self, enable_reranking: Optional[bool] = None):
        """Initialize the cross-encoder reranker based on config."""
        rerank_config = self.retrieval_config.get('reranking', {})
        enabled = enable_reranking if enable_reranking is not None else rerank_config.get('enabled', False)
        
        if enabled:
            try:
                self.reranker = create_reranker(
                    enabled=True,
                    model_name=rerank_config.get('model', 'cross-encoder/ms-marco-MiniLM-L-6-v2'),
                    top_k=self.retrieval_config.get('top_k', 5),
                    device=rerank_config.get('device', 'cpu')
                )
                self.reranking_enabled = True
                logger.info(f"Cross-encoder reranking enabled with model: {rerank_config.get('model')}")
            except Exception as e:
                logger.warning(f"Failed to initialize reranker: {e}. Disabling reranking.")
                self.reranker = None
                self.reranking_enabled = False
        else:
            self.reranker = None
            self.reranking_enabled = False
            logger.info("Reranking disabled")
    
    def _initialize_query_expander(self, enable_query_expansion: Optional[bool] = None):
        """Initialize the query expander based on config."""
        expansion_config = self.retrieval_config.get('query_expansion', {})
        enabled = enable_query_expansion if enable_query_expansion is not None else expansion_config.get('enabled', False)
        
        if enabled and self.llm is not None:
            try:
                self.query_expander = QueryExpander(
                    num_variants=expansion_config.get('num_variants', 3),
                    llm=self.llm
                )
                self.query_expansion_enabled = True
                logger.info(f"Query expansion enabled (variants={expansion_config.get('num_variants', 3)})")
            except Exception as e:
                logger.warning(f"Failed to initialize query expander: {e}. Disabling.")
                self.query_expander = None
                self.query_expansion_enabled = False
        else:
            self.query_expander = None
            self.query_expansion_enabled = False
            logger.info("Query expansion disabled")
    
    def set_retriever(self, retriever):
        """Set the retriever (PgVectorRetriever or PgHybridRetriever)."""
        self.retriever = retriever
        logger.info("Retriever set successfully")
    
    def _build_prompt(self, question: str, context: str, 
                      chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Build the prompt with context, chat history, and question."""
        history_text = ""
        if chat_history:
            history_parts = []
            for msg in chat_history:
                role = "User" if msg['role'] == 'user' else "Assistant"
                history_parts.append(f"{role}: {msg['content']}")
            history_text = f"\nPrevious conversation:\n" + "\n".join(history_parts) + "\n"
        
        return f"""You are a precise assistant that answers questions strictly based on the provided document context.
{history_text}
Document Context:
{context}

Question: {question}

Instructions:
- Answer ONLY using information explicitly stated in the Document Context above.
- If the exact answer is NOT present in the context, respond with: "I could not find this information in the uploaded document."
- Do NOT guess, infer, or use external knowledge.
- Quote key facts directly from the context when possible.
- Keep your answer concise and under 100 words.

Answer:"""
    
    def _retrieve_documents(self, question: str) -> tuple:
        """Retrieve and optionally rerank documents."""
        top_k = self.retrieval_config.get('top_k', 5)
        top_k_initial = self.retrieval_config.get('top_k_initial', 20) if self.reranking_enabled else top_k
        
        # Step 1: Retrieve (with or without query expansion)
        if self.query_expansion_enabled and self.query_expander:
            docs = self.query_expander.retrieve_with_expansion(
                question, self.retriever, top_k=top_k_initial
            )
        else:
            docs = self.retriever.invoke(question)[:top_k_initial]
        
        if not docs:
            return [], 0
        
        initial_count = len(docs)
        
        # Step 2: Rerank if enabled
        if self.reranking_enabled and self.reranker:
            docs = self.reranker.rerank(question, docs, return_scores=True)
            logger.info(f"Reranked {initial_count} docs to {len(docs)} (top_k={top_k})")
        else:
            docs = docs[:top_k]
        
        return docs, initial_count
    
    def _build_context(self, docs: List[Document]) -> str:
        """Build context string from retrieved documents."""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            # Use FULL content - no truncation, so the model sees everything
            content = doc.page_content
            source = doc.metadata.get('file_name', 'Unknown')
            context_parts.append(f"[Chunk {i} from {source}]\n{content}")
        return "\n\n".join(context_parts)
    
    def answer_question(self, question: str, 
                        chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Answer a question using RAG with OpenAI.
        
        Args:
            question: User's question
            chat_history: Optional list of previous conversation turns
            
        Returns:
            Dictionary with answer, source documents, and token usage
        """
        if self.retriever is None:
            raise ValueError("Retriever not set. Call set_retriever() first.")
        
        if self.llm is None:
            raise ValueError("Google Gemini LLM not initialized. Check configuration.")
        
        try:
            # Step 1: Retrieve documents
            docs, initial_count = self._retrieve_documents(question)
            
            if not docs:
                logger.warning("No relevant documents found for query")
                return {
                    "question": question,
                    "answer": "No relevant documents found.",
                    "source_documents": [],
                    "retrieval_info": {
                        "method": "hybrid" if hasattr(self.retriever, 'use_rrf') else "semantic",
                        "reranking_applied": False,
                        "query_expansion_applied": self.query_expansion_enabled,
                        "initial_retrieved": 0,
                        "final_count": 0
                    }
                }
            
            # Step 2: Build context
            context = self._build_context(docs)
            
            # Step 3: Build prompt with optional chat history
            prompt = self._build_prompt(question, context, chat_history)
            
            # Step 4: Generate answer (with timeout handling)
            logger.debug("Calling Google Gemini API for answer generation")
            message = HumanMessage(content=prompt)
            response = self.llm.invoke([message])
            
            answer = response.content
            
            # Log token usage
            token_usage = None
            if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
                token_usage = response.response_metadata['token_usage']
                completion_details = token_usage.get('completion_tokens_details', {})
                
                log_msg = f"Token Usage - Prompt: {token_usage.get('prompt_tokens', 0)}, " \
                         f"Completion: {token_usage.get('completion_tokens', 0)}"
                
                if completion_details:
                    reasoning = completion_details.get('reasoning_tokens', 0)
                    output = token_usage.get('completion_tokens', 0) - reasoning
                    if reasoning > 0:
                        log_msg += f" (Output: {output}, Reasoning: {reasoning})"
                
                log_msg += f", Total: {token_usage.get('total_tokens', 0)}"
                logger.info(log_msg)
            
            # Handle empty responses
            if not answer or answer.strip() == '':
                logger.warning(f"Empty response. Metadata: {response.response_metadata}")
                answer = "I apologize, but I received an empty response. Please try again."
            
            # Step 5: Format response
            retrieval_method = "semantic"
            if docs and docs[0].metadata.get('retrieval_method'):
                retrieval_method = docs[0].metadata.get('retrieval_method')
            
            result = {
                "question": question,
                "answer": answer,
                "source_documents": [
                    {
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "source": doc.metadata.get("file_name", "Unknown"),
                        "type": doc.metadata.get("content_type", "file"),
                        "rerank_score": doc.metadata.get("rerank_score"),
                        "fusion_score": doc.metadata.get("fusion_score")
                    }
                    for doc in docs
                ],
                "num_retrieved": len(docs),
                "retrieval_info": {
                    "method": retrieval_method,
                    "reranking_applied": self.reranking_enabled,
                    "query_expansion_applied": self.query_expansion_enabled,
                    "initial_retrieved": initial_count,
                    "final_count": len(docs)
                }
            }
            
            # Add token usage if available
            if token_usage:
                result["token_usage"] = {
                    "prompt_tokens": token_usage.get('prompt_tokens', 0),
                    "completion_tokens": token_usage.get('completion_tokens', 0),
                    "total_tokens": token_usage.get('total_tokens', 0)
                }
                
                completion_details = token_usage.get('completion_tokens_details', {})
                if completion_details:
                    reasoning = completion_details.get('reasoning_tokens', 0)
                    if reasoning > 0:
                        result["token_usage"]["reasoning_tokens"] = reasoning
                        result["token_usage"]["output_tokens"] = token_usage.get('completion_tokens', 0) - reasoning
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "source_documents": []
            }
    
    async def answer_question_stream(self, question: str,
                               chat_history: Optional[List[Dict[str, str]]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream answer tokens using RAG with OpenAI.
        
        Yields dictionaries with either:
        - {"type": "token", "content": "..."} for each token
        - {"type": "sources", "sources": [...]} for source documents
        - {"type": "done", "token_usage": {...}} when complete
        - {"type": "error", "content": "..."} on error
        
        Args:
            question: User's question
            chat_history: Optional list of previous conversation turns
        """
        if self.retriever is None:
            yield {"type": "error", "content": "Retriever not set."}
            return
        
        if self.llm is None:
            yield {"type": "error", "content": "Google Gemini LLM not initialized."}
            return
        
        try:
            # Step 1: Retrieve documents
            docs, initial_count = self._retrieve_documents(question)
            
            if not docs:
                yield {"type": "token", "content": "No relevant documents found."}
                yield {"type": "done", "token_usage": None}
                return
            
            # Yield sources first so UI can display them
            yield {
                "type": "sources",
                "sources": [
                    {
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "source": doc.metadata.get("file_name", "Unknown"),
                        "type": doc.metadata.get("content_type", "file"),
                    }
                    for doc in docs
                ]
            }
            
            # Step 2: Build context and prompt
            context = self._build_context(docs)
            prompt = self._build_prompt(question, context, chat_history)
            
            # Step 3: Stream response
            logger.debug("Streaming Google Gemini API response")
            message = HumanMessage(content=prompt)
            
            full_response = ""
            async for chunk in self.llm.astream([message]):
                if chunk.content:
                    full_response += chunk.content
                    yield {"type": "token", "content": chunk.content}
            
            # Step 4: Done
            yield {"type": "done", "token_usage": None}
            
            logger.debug(f"Streamed answer ({len(full_response)} chars)")
            
        except Exception as e:
            logger.error(f"Error streaming answer: {e}")
            yield {"type": "error", "content": f"Error: {str(e)}"}
    
    def batch_answer(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Answer multiple questions.
        
        Args:
            questions: List of questions
            
        Returns:
            List of answer dictionaries
        """
        results = []
        for question in questions:
            result = self.answer_question(question)
            results.append(result)
        return results
