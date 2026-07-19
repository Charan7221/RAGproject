"""
Configuration Loader

Centralized configuration management for the RAG system.
Loads settings from config/config.yml.
"""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path


class ConfigLoader:
    """Singleton configuration loader."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the configuration loader."""
        if self._config is None:
            self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        # Find config file relative to project root
        current_dir = Path(__file__).resolve()
        project_root = current_dir.parent.parent.parent.parent
        config_path = project_root / "config" / "config.yml"
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found at {config_path}. "
                "Please create config/config.yml"
            )
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Path to config value (e.g., 'text_splitter.chunk_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
            
        Example:
            config = ConfigLoader()
            chunk_size = config.get('text_splitter.chunk_size')
            db_host = config.get('database.postgresql.host')
        """
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_text_splitter_config(self) -> Dict[str, Any]:
        """Get text splitter configuration."""
        return {
            'chunk_size': self.get('text_splitter.chunk_size', 512),
            'chunk_overlap': self.get('text_splitter.chunk_overlap', 50),
            'separators': self.get('text_splitter.separators', ["\n\n", "\n", ". ", " ", ""])
        }
    
    def get_embeddings_config(self) -> Dict[str, Any]:
        """Get embeddings configuration."""
        return {
            'default_model': self.get('embeddings.default_model', 
                                     'sentence-transformers/all-MiniLM-L6-v2'),
            'sbert_model': self.get('embeddings.sbert_model',
                                   'BAAI/bge-base-en-v1.5'),
            'device': self.get('embeddings.device', 'cpu')
        }
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return {
            'use_openai': self.get('llm.use_openai', True),
            'api_key': self.get('llm.api_key', ''),
            'model': self.get('llm.model', 'gpt-4.1-nano'),
            'base_url': self.get('llm.base_url', None),
            'params': self.get('llm.params', {})
        }
    
    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval configuration including hybrid search, reranking, and query expansion."""
        return {
            'top_k': self.get('retrieval.top_k', 5),
            'top_k_initial': self.get('retrieval.top_k_initial', 20),
            'search_type': self.get('retrieval.search_type', 'similarity'),
            # Hybrid search configuration
            'hybrid_search': {
                'enabled': self.get('retrieval.hybrid_search.enabled', True),
                'alpha': self.get('retrieval.hybrid_search.alpha', 0.5),
                'use_rrf': self.get('retrieval.hybrid_search.use_rrf', True),
                'rrf_k': self.get('retrieval.hybrid_search.rrf_k', 60)
            },
            # Reranking configuration
            'reranking': {
                'enabled': self.get('retrieval.reranking.enabled', True),
                'model': self.get('retrieval.reranking.model', 'cross-encoder/ms-marco-MiniLM-L-6-v2'),
                'device': self.get('retrieval.reranking.device', 'cpu'),
                'batch_size': self.get('retrieval.reranking.batch_size', 32)
            },
            # Query expansion configuration
            'query_expansion': {
                'enabled': self.get('retrieval.query_expansion.enabled', True),
                'num_variants': self.get('retrieval.query_expansion.num_variants', 3)
            }
        }
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get PostgreSQL database configuration (env vars override YAML)."""
        return {
            'host': os.getenv('DB_HOST', self.get('database.postgresql.host', 'localhost')),
            'port': int(os.getenv('DB_PORT', self.get('database.postgresql.port', 5432))),
            'database': os.getenv('DB_NAME', self.get('database.postgresql.database', 'rag_db')),
            'user': os.getenv('DB_USER', self.get('database.postgresql.user', 'rag_user')),
            'password': os.getenv('DB_PASSWORD', self.get('database.postgresql.password', 'rag_password')),
            'pool_size': self.get('database.postgresql.pool_size', 5),
            'max_overflow': self.get('database.postgresql.max_overflow', 10)
        }
    
    def get_conversation_config(self) -> Dict[str, Any]:
        """Get conversation memory configuration."""
        return {
            'max_history_turns': self.get('conversation.max_history_turns', 5)
        }
    
    def get_metadata_config(self) -> Dict[str, Any]:
        """Get metadata configuration (simplified)."""
        return {
            'content_type': 'file',
            'timestamp_enabled': True
        }
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration."""
        return self._config.copy()


# Global config instance
config = ConfigLoader()


# Convenience functions
def get_config(key_path: str, default: Any = None) -> Any:
    """Get configuration value."""
    return config.get(key_path, default)


def get_text_splitter_config() -> Dict[str, Any]:
    """Get text splitter configuration."""
    return config.get_text_splitter_config()


def get_embeddings_config() -> Dict[str, Any]:
    """Get embeddings configuration."""
    return config.get_embeddings_config()


def get_llm_config() -> Dict[str, Any]:
    """Get LLM configuration."""
    return config.get_llm_config()


def get_retrieval_config() -> Dict[str, Any]:
    """Get retrieval configuration."""
    return config.get_retrieval_config()


def get_database_config() -> Dict[str, Any]:
    """Get PostgreSQL database configuration."""
    return config.get_database_config()


def get_conversation_config() -> Dict[str, Any]:
    """Get conversation memory configuration."""
    return config.get_conversation_config()


def get_metadata_config() -> Dict[str, Any]:
    """Get metadata configuration."""
    return config.get_metadata_config()
