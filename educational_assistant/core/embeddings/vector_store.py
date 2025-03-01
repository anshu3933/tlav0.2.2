# core/embeddings/vector_store.py

import os
import shutil
import time
from typing import List, Optional, Dict, Any, Tuple, Callable
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
import numpy as np
from config.app_config import config
from config.logging_config import get_module_logger
from core.embeddings.embedding_manager import EmbeddingManager, TextChunkProcessor

# Create a logger for this module
logger = get_module_logger("vector_store")

class VectorStoreError(Exception):
    """Exception raised for vector store errors."""
    pass

class FAISSVectorStore:
    """Manages FAISS vector store operations with error handling."""
    
    def __init__(self, 
                embedding_manager: Optional[EmbeddingManager] = None,
                index_dir: Optional[str] = None):
        """Initialize with components and directories.
        
        Args:
            embedding_manager: Manager for embeddings
            index_dir: Directory to store the index
        """
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.index_dir = index_dir or config.vector_store.index_dir
        self.chunk_processor = TextChunkProcessor()
        self.vectorstore = None
        
        # Create index directory if it doesn't exist
        os.makedirs(self.index_dir, exist_ok=True)
        
        logger.debug(f"Initialized FAISS vector store with index directory: {self.index_dir}")
    
    def build_index(self, documents: List[Document], force_rebuild: bool = False) -> bool:
        """Build a FAISS index from documents.
        
        Args:
            documents: Documents to index
            force_rebuild: Whether to force rebuild even if index exists
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if index already exists
            if not force_rebuild and self._index_exists():
                logger.debug("Index already exists. Loading existing index.")
                return self.load_index()
            
            # Process documents into chunks and get embeddings
            chunked_docs, embeddings = self.embedding_manager.embed_documents(documents)
            
            if not chunked_docs or not embeddings:
                logger.error("No documents or embeddings to index")
                return False
            
            # Create FAISS index
            self._create_index_from_embeddings(chunked_docs, embeddings)
            
            logger.info(f"Successfully built index with {len(chunked_docs)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}", exc_info=True)
            return False
    
    def _create_index_from_embeddings(self, 
                                     documents: List[Document], 
                                     embeddings: List[List[float]]) -> None:
        """Create a FAISS index from documents and embeddings.
        
        Args:
            documents: Chunked documents
            embeddings: Embeddings for each document
        """
        # Convert embeddings to numpy arrays
        embedding_arrays = np.array(embeddings, dtype=np.float32)
        
        # Create FAISS index directly (bypassing langchain to avoid redundant embedding calls)
        import faiss
        
        # Get embedding dimension
        dimension = embedding_arrays.shape[1]
        
        # Create index
        index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        index.add(embedding_arrays)
        
        # Create wrapper class for compatibility with langchain
        class EmbeddingFunction:
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return self._embed(texts)
            
            def embed_query(self, text: str) -> List[float]:
                return self._embed([text])[0]
                
            def _embed(self, texts: List[str]) -> List[List[float]]:
                # This won't actually be called when using FAISS directly
                pass
        
        # Create FAISS wrapper with documents
        self.vectorstore = FAISS(
            embedding_function=EmbeddingFunction(),
            index=index,
            docstore=self._create_docstore(documents),
            index_to_docstore_id=self._create_index_mapping(documents)
        )
        
        # Save index to disk
        self.save_index()
    
    def _create_docstore(self, documents: List[Document]) -> Dict[str, Document]:
        """Create a document store mapping IDs to documents.
        
        Args:
            documents: Documents to store
            
        Returns:
            Document store mapping
        """
        return {str(i): doc for i, doc in enumerate(documents)}
    
    def _create_index_mapping(self, documents: List[Document]) -> Dict[int, str]:
        """Create a mapping from index positions to document IDs.
        
        Args:
            documents: Documents to map
            
        Returns:
            Index to document ID mapping
        """
        return {i: str(i) for i in range(len(documents))}
    
    def save_index(self) -> bool:
        """Save the FAISS index to disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.vectorstore:
                logger.error("No vectorstore to save")
                return False
            
            # Create a backup of existing index if it exists
            if self._index_exists():
                self._backup_index()
            
            # Save index
            self.vectorstore.save_local(self.index_dir)
            logger.info(f"Saved index to {self.index_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}", exc_info=True)
            return False
    
    def load_index(self) -> bool:
        """Load the FAISS index from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._index_exists():
                logger.error(f"Index directory not found: {self.index_dir}")
                return False
            
            # Create embedding function wrapper
            class EmbeddingFunction:
                def __init__(self, embedding_manager):
                    self.embedding_manager = embedding_manager
                
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    return self.embedding_manager.get_embeddings(texts)
                
                def embed_query(self, text: str) -> List[float]:
                    return self.embedding_manager.get_embeddings([text])[0]
            
            # Create embedding function
            embedding_function = EmbeddingFunction(self.embedding_manager)
            
            # Load index
            self.vectorstore = FAISS.load_local(
                self.index_dir,
                embedding_function
            )
            
            logger.info(f"Loaded index from {self.index_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}", exc_info=True)
            return False
    
    def _index_exists(self) -> bool:
        """Check if index exists on disk.
        
        Returns:
            True if index exists, False otherwise
        """
        return (
            os.path.exists(self.index_dir) and 
            os.path.exists(os.path.join(self.index_dir, "index.faiss")) and
            os.path.exists(os.path.join(self.index_dir, "index.pkl"))
        )
    
    def _backup_index(self) -> None:
        """Create a backup of the existing index."""
        try:
            # Create backup directory
            backup_dir = f"{self.index_dir}_backup_{int(time.time())}"
            
            # Copy index files
            shutil.copytree(self.index_dir, backup_dir)
            
            logger.debug(f"Created index backup at {backup_dir}")
            
        except Exception as e:
            logger.error(f"Error creating index backup: {str(e)}")
    
    def search(self, query: str, k: int = None) -> List[Document]:
        """Search the index for similar documents.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of similar documents
            
        Raises:
            VectorStoreError: If search fails
        """
        try:
            if not self.vectorstore:
                # Try to load index
                if not self.load_index():
                    raise VectorStoreError("No index available for search")
            
            # Use configurable k if not specified
            k = k or config.vector_store.similarity_top_k
            
            # Get query embedding
            query_embedding = self.embedding_manager.get_embeddings([query])[0]
            
            # Search FAISS index directly
            docs_and_scores = self.vectorstore.similarity_search_with_score_by_vector(
                embedding=query_embedding,
                k=k
            )
            
            # Extract documents
            docs = [doc for doc, score in docs_and_scores]
            
            logger.debug(f"Found {len(docs)} documents for query: {query[:50]}...")
            return docs
            
        except VectorStoreError:
            # Re-raise specific error
            raise
        except Exception as e:
            logger.error(f"Error searching index: {str(e)}", exc_info=True)
            raise VectorStoreError(f"Search failed: {str(e)}")
    
    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None) -> Callable:
        """Get a retriever function for the vector store.
        
        Args:
            search_kwargs: Search parameters
            
        Returns:
            Retriever function
            
        Raises:
            VectorStoreError: If retriever creation fails
        """
        try:
            if not self.vectorstore and not self.load_index():
                raise VectorStoreError("No index available for retrieval")
            
            # Create search parameters with defaults
            search_kwargs = search_kwargs or {
                "k": config.vector_store.similarity_top_k
            }
            
            # Create retriever function
            def retriever(query: str) -> List[Document]:
                return self.search(query, k=search_kwargs.get("k"))
            
            return retriever
            
        except VectorStoreError:
            # Re-raise specific error
            raise
        except Exception as e:
            logger.error(f"Error creating retriever: {str(e)}", exc_info=True)
            raise VectorStoreError(f"Failed to create retriever: {str(e)}")
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to an existing index.
        
        Args:
            documents: Documents to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.vectorstore and not self.load_index():
                logger.error("No index available to add documents to")
                return False
            
            # Process documents into chunks and get embeddings
            chunked_docs, embeddings = self.embedding_manager.embed_documents(documents)
            
            if not chunked_docs or not embeddings:
                logger.error("No documents or embeddings to add")
                return False
            
            # Add documents to index
            self.vectorstore.add_embeddings(
                text_embeddings=list(zip([doc.page_content for doc in chunked_docs], embeddings)),
                metadatas=[doc.metadata for doc in chunked_docs]
            )
            
            # Save updated index
            self.save_index()
            
            logger.info(f"Added {len(chunked_docs)} document chunks to index")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to index: {str(e)}", exc_info=True)
            return False
    
    def clear_index(self) -> bool:
        """Clear the index and remove all documents.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create backup before clearing
            if self._index_exists():
                self._backup_index()
            
            # Remove index files
            if os.path.exists(os.path.join(self.index_dir, "index.faiss")):
                os.remove(os.path.join(self.index_dir, "index.faiss"))
            
            if os.path.exists(os.path.join(self.index_dir, "index.pkl")):
                os.remove(os.path.join(self.index_dir, "index.pkl"))
            
            # Reset vectorstore
            self.vectorstore = None
            
            logger.info("Cleared index")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing index: {str(e)}", exc_info=True)
            return False