# main.py

import os
import streamlit as st
from typing import Dict, Any, Optional
from config.app_config import config
from config.logging_config import get_module_logger
from core.embeddings.vector_store import FAISSVectorStore
from core.embeddings.embedding_manager import EmbeddingManager  # Added missing import
from core.llm.llm_client import LLMClient
from core.rag.chain_builder import RAGChainBuilder
from core.rag.retriever import HybridRetriever, WebAugmentedRetriever  # Added missing import
from ui.state_manager import state_manager

# Create a logger for this module
logger = get_module_logger("main")

def initialize_application() -> Dict[str, Any]:
    """Initialize the application components.
    
    Returns:
        Dictionary with initialized components
    """
    try:
        logger.info("Initializing application")
        
        # Initialize LLM client
        llm_client = LLMClient()
        logger.debug("Initialized LLM client")
        
        # Initialize vector store
        vector_store = FAISSVectorStore()
        if not vector_store.load_index():
            logger.warning("No index found. Documents need to be processed.")
        else:
            logger.debug("Loaded existing vector store index")
        
        # Initialize RAG chain
        rag_chain = RAGChainBuilder.build(
            llm_client=llm_client,
            vector_store=vector_store,
            use_web_search=True
        )
        logger.debug("Initialized RAG chain")
        
        # Update system state
        state_manager.update_system_state(
            llm_initialized=True,
            vector_store_initialized=vector_store._index_exists(),
            chain_initialized=True
        )
        
        return {
            "llm_client": llm_client,
            "vector_store": vector_store,
            "rag_chain": rag_chain
        }
        
    except Exception as e:
        logger.error(f"Error initializing application: {str(e)}", exc_info=True)
        state_manager.add_error(f"Failed to initialize application: {str(e)}")
        return {}

def check_environment() -> bool:
    """Check if the environment is properly configured.
    
    Returns:
        True if environment is valid, False otherwise
    """
    # Check API key
    if not config.llm.api_key:
        logger.error("OpenAI API key not found in environment variables")
        state_manager.add_error("OpenAI API key not found. Please add it to your .env file.")
        return False
    
    # Check data directory
    if not os.path.exists(config.document.data_dir):
        try:
            os.makedirs(config.document.data_dir, exist_ok=True)
            logger.debug(f"Created data directory: {config.document.data_dir}")
        except Exception as e:
            logger.error(f"Failed to create data directory: {str(e)}")
            state_manager.add_error(f"Failed to create data directory: {str(e)}")
            return False
    
    # Check index directory
    if not os.path.exists(config.vector_store.index_dir):
        try:
            os.makedirs(config.vector_store.index_dir, exist_ok=True)
            logger.debug(f"Created index directory: {config.vector_store.index_dir}")
        except Exception as e:
            logger.error(f"Failed to create index directory: {str(e)}")
            state_manager.add_error(f"Failed to create index directory: {str(e)}")
            return False
    
    return True
    
def load_app_components() -> Dict[str, Any]:
    """
    Initialize and load all application components.
    
    Returns:
        Dictionary containing all initialized components
    """
    logger.info("Initializing application components")
    components = {}
    
    try:
        # Initialize LLM client
        logger.debug("Initializing LLM client")
        llm_client = LLMClient()
        components["llm_client"] = llm_client
        
        # Initialize embedding manager
        logger.debug("Initializing embedding manager")
        embedding_manager = EmbeddingManager(llm_client=llm_client)
        components["embedding_manager"] = embedding_manager
        
        # Initialize vector store
        logger.debug("Initializing vector store")
        vector_store = FAISSVectorStore(embedding_manager=embedding_manager)
        components["vector_store"] = vector_store
        
        # Load existing vector store index if available
        if vector_store.load_index():
            logger.info("Loaded existing vector store index")
        else:
            logger.info("No existing vector store index found")
        
        # Initialize retriever
        logger.debug("Initializing retriever")
        retriever = HybridRetriever(vector_store=vector_store)
        components["retriever"] = retriever
        
        # Initialize RAG chain
        logger.debug("Initializing RAG chain")
        rag_chain = RAGChainBuilder.build(
            llm_client=llm_client,
            vector_store=vector_store,
            use_web_search=False
        )
        components["rag_chain"] = rag_chain
        
        # Initialize pipeline components
        logger.debug("Initializing pipeline components")
        # Import these here rather than at the module level
        from core.pipelines.iep_pipeline import IEPGenerationPipeline
        from core.pipelines.lesson_plan_pipeline import LessonPlanGenerationPipeline
        
        # Create pipeline instances
        iep_pipeline = IEPGenerationPipeline(llm_client=llm_client)
        lesson_plan_pipeline = LessonPlanGenerationPipeline(llm_client=llm_client)
        
        # Add to components dictionary
        components["iep_pipeline"] = iep_pipeline
        components["lesson_plan_pipeline"] = lesson_plan_pipeline
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}", exc_info=True)
        # Return any components that were successfully initialized
    
    return components
