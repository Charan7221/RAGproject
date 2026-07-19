"""
Custom Text Splitter with Tiktoken Encoding

Advanced text splitting using tiktoken for accurate token-based chunking.
Configuration loaded from config/config.yml.
"""

from typing import List, Optional
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import config loader
from ..utils.config_loader import get_text_splitter_config


class TextSplitter(BaseModel):
    """
    Recursive Chunker for multi-level text data chunking.

    This chunker is designed to perform text chunking recursively using a variety 
    of specified separators. It's useful for complex chunking tasks where multiple 
    layers of chunking are necessary.

    Attributes:
    -----------
    chunk_size : Optional[int]
        The optional chunk size, defining the maximum size of each chunk. 
        Default is 500 tokens.

    chunk_overlap : Optional[int]
        The optional chunk overlap size, specifying how much overlap there should 
        be between consecutive chunks. Default is 100 tokens.

    separators : Optional[List[str]]
        A list of optional separators to be used for recursive chunking. Each 
        separator defines a new level of chunking. Default separators include 
        space characters.
    """

    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    separators: Optional[List[str]] = None
    
    def __init__(self, **data):
        """Initialize with config defaults if not provided."""
        config = get_text_splitter_config()
        
        # Use provided values or fall back to config
        if 'chunk_size' not in data or data['chunk_size'] is None:
            data['chunk_size'] = config['chunk_size']
        if 'chunk_overlap' not in data or data['chunk_overlap'] is None:
            data['chunk_overlap'] = config['chunk_overlap']
        if 'separators' not in data or data['separators'] is None:
            data['separators'] = config['separators']
            
        super().__init__(**data)

    def split_documents(self, documents):
        """
        Split the documents into smaller chunks using tiktoken encoding.
        
        @param documents: langchain documents contains content information
        @return: chunked documents
        """
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
        final_documents = splitter.split_documents(documents)
        return final_documents

