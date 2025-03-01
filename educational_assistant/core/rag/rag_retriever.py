# core/rag/retriever.py

from typing import List, Dict, Any, Optional, Callable, Union
from langchain.schema import Document
from config.app_config import config
from config.logging_config import get_module_logger
from core.embeddings.vector_store import FAISSVectorStore, VectorStoreError

# Create a logger for this module
logger = get_module_logger("rag_retriever")

class RetrievalError(Exception):
    """Exception raised for retrieval errors."""
    pass

class HybridRetriever:
    """Retrieves documents using hybrid search methods."""
    
    def __init__(self, 
                vector_store: Optional[FAISSVectorStore] = None,
                k_documents: int = None):
        """Initialize with vector store.
        
        Args:
            vector_store: Vector store for semantic search
            k_documents: Number of documents to retrieve
        """
        self.vector_store = vector_store or FAISSVectorStore()
        self.k_documents = k_documents or config.vector_store.similarity_top_k
        logger.debug(f"Initialized hybrid retriever with k={self.k_documents}")
    
    def retrieve(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            
        Returns:
            List of relevant documents
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            # Semantic search using vector store
            semantic_docs = self._semantic_search(query)
            
            # Could add keyword search, web search, etc. here
            
            # For now, just return semantic search results
            return semantic_docs
            
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}", exc_info=True)
            raise RetrievalError(f"Retrieval failed: {str(e)}")
    
    def _semantic_search(self, query: str) -> List[Document]:
        """Perform semantic search using vector store.
        
        Args:
            query: Query string
            
        Returns:
            List of relevant documents
            
        Raises:
            RetrievalError: If semantic search fails
        """
        try:
            return self.vector_store.search(query, k=self.k_documents)
        except VectorStoreError as e:
            logger.error(f"Vector store search failed: {str(e)}")
            raise RetrievalError(f"Semantic search failed: {str(e)}")
    
    def as_retriever(self) -> Callable:
        """Get a retriever function for use in RAG chains.
        
        Returns:
            Retriever function
        """
        def retriever(query: str) -> List[Document]:
            return self.retrieve(query)
        
        return retriever


class WebAugmentedRetriever(HybridRetriever):
    """Retrieves documents and augments with web search results."""
    
    def __init__(self, 
                vector_store: Optional[FAISSVectorStore] = None,
                k_documents: int = None,
                web_search_enabled: bool = True,
                max_web_results: int = 3):
        """Initialize with vector store and web search configuration.
        
        Args:
            vector_store: Vector store for semantic search
            k_documents: Number of documents to retrieve
            web_search_enabled: Whether to enable web search
            max_web_results: Maximum number of web search results
        """
        super().__init__(vector_store, k_documents)
        self.web_search_enabled = web_search_enabled
        self.max_web_results = max_web_results
        logger.debug(f"Initialized web-augmented retriever with web_search={web_search_enabled}")
    
    def retrieve(self, query: str) -> List[Document]:
        """Retrieve relevant documents and augment with web search.
        
        Args:
            query: Query string
            
        Returns:
            List of relevant documents
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            # Get documents from vector store
            semantic_docs = self._semantic_search(query)
            
            # If web search is enabled
            if self.web_search_enabled:
                web_docs = self._web_search(query)
                
                # Combine results (semantic search first, then web)
                return semantic_docs + web_docs
            
            return semantic_docs
            
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}", exc_info=True)
            raise RetrievalError(f"Retrieval failed: {str(e)}")
    
    def _web_search(self, query: str) -> List[Document]:
        """Perform web search for query.
        
        Args:
            query: Query string
            
        Returns:
            List of documents from web search
        """
        try:
            # Try to import DuckDuckGo search
            try:
                from duckduckgo_search import DDGS
                ddgs = DDGS()
                
                # Perform search
                results = ddgs.text(query, max_results=self.max_web_results)
                
                # Convert to documents
                web_docs = []
                for r in results:
                    doc = Document(
                        page_content=f"{r['title']}\n\n{r['body']}",
                        metadata={
                            "source": r["link"],
                            "title": r["title"],
                            "type": "web_search"
                        }
                    )
                    web_docs.append(doc)
                
                logger.debug(f"Found {len(web_docs)} web search results for query: {query[:50]}...")
                return web_docs
                
            except ImportError:
                logger.warning("duckduckgo_search not available, web search disabled")
                return []
                
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            # Don't fail the whole retrieval if web search fails
            return []
