# core/embeddings/embedding_manager.py

import os
import pickle
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from config.app_config import config
from config.logging_config import get_module_logger
from core.llm.llm_client import LLMClient

# Create a logger for this module
logger = get_module_logger("embedding_manager")

class EmbeddingCache:
    """Cache for document embeddings."""
    
    def __init__(self, cache_dir: str = ".cache/embeddings"):
        """Initialize with cache directory.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.ttl = config.vector_store.cache_ttl if hasattr(config.vector_store, 'cache_ttl') else 86400  # 24 hours
        logger.debug(f"Initialized embedding cache in {cache_dir}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a text.
        
        Args:
            text: Text to generate key for
            
        Returns:
            Cache key
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, key: str) -> str:
        """Get the cache file path for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Cache file path
        """
        return os.path.join(self.cache_dir, f"{key}.pkl")
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get embeddings from cache if available.
        
        Args:
            text: Text to get embeddings for
            
        Returns:
            Cached embeddings or None if not found/expired
        """
        key = self._get_cache_key(text)
        cache_path = self._get_cache_path(key)
        
        if os.path.exists(cache_path):
            try:
                # Check if cache is still valid
                if time.time() - os.path.getmtime(cache_path) > self.ttl:
                    logger.debug(f"Cache expired for key {key}")
                    return None
                
                # Load cached embeddings
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
                    
            except Exception as e:
                logger.error(f"Error loading cache for key {key}: {str(e)}")
                # Delete corrupted cache file
                try:
                    os.unlink(cache_path)
                except:
                    pass
                
        return None
    
    def set(self, text: str, embeddings: List[float]) -> None:
        """Set embeddings in cache.
        
        Args:
            text: Text to cache embeddings for
            embeddings: Embeddings to cache
        """
        key = self._get_cache_key(text)
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
                
            logger.debug(f"Cached embeddings for key {key}")
                
        except Exception as e:
            logger.error(f"Error caching embeddings for key {key}: {str(e)}")
    
    def clear(self) -> None:
        """Clear the entire cache."""
        try:
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    
            logger.debug("Cleared embedding cache")
                
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")


class TextChunkProcessor:
    """Handles text chunking for embedding."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """Initialize with chunking parameters.
        
        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size or config.vector_store.chunk_size
        self.chunk_overlap = chunk_overlap or config.vector_store.chunk_overlap
        logger.debug(f"Initialized text chunker with size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Simple text splitting by chunk size with overlap
        chunks = []
        
        if not text:
            return chunks
            
        # Determine natural break points (newlines, periods, etc.)
        break_points = []
        for i, char in enumerate(text):
            if char in ['\n', '.', '!', '?']:
                break_points.append(i)
        
        # Add the end of text as a break point
        break_points.append(len(text) - 1)
        
        # Split text using natural break points close to chunk size
        start = 0
        while start < len(text):
            # Find the furthest break point within chunk_size
            end = next((bp for bp in break_points if bp >= start + self.chunk_size), None)
            
            # If no break point found, use chunk_size directly
            if end is None or end >= len(text) - 1:
                end = min(start + self.chunk_size, len(text))
            else:
                # Include the break character
                end = end + 1
            
            # Add the chunk
            chunks.append(text[start:end])
            
            # Move start with overlap
            start = max(start, end - self.chunk_overlap)
            
            # If we're at the end, break
            if start >= len(text):
                break
        
        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks.
        
        Args:
            documents: Documents to split
            
        Returns:
            List of chunked documents
        """
        chunked_docs = []
        
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            
            for i, chunk in enumerate(chunks):
                # Create new document with chunk and metadata
                chunked_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk": i,
                        "chunk_size": self.chunk_size,
                        "total_chunks": len(chunks)
                    }
                )
                chunked_docs.append(chunked_doc)
        
        logger.debug(f"Split {len(documents)} documents into {len(chunked_docs)} chunks")
        return chunked_docs


class EmbeddingManager:
    """Manages document embedding with caching."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None, use_cache: bool = True):
        """Initialize with client and cache.
        
        Args:
            llm_client: LLM client (default: create new one)
            use_cache: Whether to use embedding cache
        """
        self.llm_client = llm_client or LLMClient()
        self.chunk_processor = TextChunkProcessor()
        self.use_cache = use_cache and config.vector_store.cache_embeddings
        
        if self.use_cache:
            self.cache = EmbeddingCache()
        
        logger.debug(f"Initialized embedding manager with cache={'enabled' if self.use_cache else 'disabled'}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts with caching.
        
        Args:
            texts: Texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        # If cache disabled, get all embeddings from API
        if not self.use_cache:
            return self.llm_client.embeddings(texts)
        
        # Check cache for each text
        embeddings = []
        texts_to_embed = []
        cache_indices = []
        
        for i, text in enumerate(texts):
            cached_embedding = self.cache.get(text)
            
            if cached_embedding is not None:
                # Use cached embedding
                embeddings.append((i, cached_embedding))
                logger.debug(f"Using cached embedding for text {i}")
            else:
                # Need to get embedding from API
                texts_to_embed.append(text)
                cache_indices.append(i)
        
        # If we need to get embeddings from API
        if texts_to_embed:
            logger.debug(f"Getting {len(texts_to_embed)} embeddings from API")
            api_embeddings = self.llm_client.embeddings(texts_to_embed)
            
            # Cache new embeddings
            for i, embedding in zip(cache_indices, api_embeddings):
                self.cache.set(texts[i], embedding)
                embeddings.append((i, embedding))
        
        # Sort by original index and return only embeddings
        embeddings.sort(key=lambda x: x[0])
        return [e[1] for e in embeddings]
    
    def embed_documents(self, documents: List[Document]) -> Tuple[List[Document], List[List[float]]]:
        """Embed documents with chunking and caching.
        
        Args:
            documents: Documents to embed
            
        Returns:
            Tuple of (chunked documents, embeddings)
        """
        # Split documents into chunks
        chunked_docs = self.chunk_processor.split_documents(documents)
        
        # Get text content from each chunk
        texts = [doc.page_content for doc in chunked_docs]
        
        # Get embeddings
        embeddings = self.get_embeddings(texts)
        
        logger.debug(f"Embedded {len(chunked_docs)} document chunks")
        return chunked_docs, embeddings
