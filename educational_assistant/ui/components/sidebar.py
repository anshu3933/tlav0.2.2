"""Sidebar component with document upload and system status."""

import streamlit as st
import os
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional

from config.logging_config import get_module_logger
from core.document_processing.file_handler import FileHandler, FileHandlerError
from core.document_processing.document_loader import DocumentLoader
from ui.state_manager import state_manager
from ui.components.common import display_error, display_success, display_info

# Create a logger for this module
logger = get_module_logger("sidebar_component")

def render_sidebar(app_components: Dict[str, Any]):
    """Render the sidebar with document upload and system status.
    
    Args:
        app_components: Dictionary with application components
    """
    with st.sidebar:
        st.title("Document Upload")
        
        # File upload section
        render_file_upload(app_components)
        
        # System status section
        st.title("System Status")
        render_system_status(app_components)

def render_file_upload(app_components: Dict[str, Any]):
    """Render the file upload section in the sidebar.
    
    Args:
        app_components: Dictionary with application components
    """
    # Create file uploader
    uploaded_files = st.file_uploader(
        "Upload educational documents",
        type=["txt", "docx", "pdf", "md", "csv", "json", "xlsx"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        process_uploaded_files(uploaded_files, app_components)
    
    # Add clear documents button
    if state_manager.get("documents_processed", False):
        if st.button("Clear Documents"):
            clear_documents(app_components)

def process_uploaded_files(uploaded_files, app_components: Dict[str, Any]):
    """Process uploaded files and update state.
    
    Args:
        uploaded_files: List of uploaded files
        app_components: Dictionary with application components
    """
    # Get vector store from components
    vector_store = app_components.get("vector_store")
    if not vector_store:
        display_error("Vector store not initialized. Cannot process documents.")
        return
    
    # Initialize handlers
    file_handler = FileHandler()
    document_loader = DocumentLoader()
    
    # Process only if not already processed
    if not state_manager.get("documents_processed", False):
        processing_success = True
        
        with st.spinner("Processing documents..."):
            st.write("### Processing Files")
            
            for file in uploaded_files:
                status_container = st.empty()
                status_container.info(f"Processing {file.name}...")
                
                try:
                    # Process uploaded file
                    uploaded_file = file_handler.process_uploaded_file(file)
                    
                    # Load document
                    result = document_loader.load_single_document(uploaded_file.temp_path)
                    
                    if not result.success:
                        status_container.error(f"Error processing {file.name}: {result.error_message}")
                        processing_success = False
                        continue
                    
                    # Add to state
                    document = result.document
                    document.metadata["source"] = file.name
                    document.metadata["upload_time"] = datetime.now().isoformat()
                    
                    state_manager.append("documents", document)
                    
                    # Add to vector store
                    vector_store.add_documents([document])
                    
                    status_container.success(f"Successfully processed {file.name}")
                    
                except FileHandlerError as e:
                    status_container.error(f"Error handling {file.name}: {str(e)}")
                    processing_success = False
                except Exception as e:
                    logger.error(f"Unexpected error processing {file.name}: {str(e)}", exc_info=True)
                    status_container.error(f"Unexpected error processing {file.name}")
                    processing_success = False
            
            if processing_success:
                state_manager.set("documents_processed", True)
                display_success("All documents processed successfully!")
                
                # Update system state
                state_manager.update_system_state(
                    vector_store_initialized=True
                )
            else:
                display_error("Error processing some documents.")
    
    # Clean up temporary files
    file_handler.cleanup()

def clear_documents(app_components: Dict[str, Any]):
    """Clear documents and reset state.
    
    Args:
        app_components: Dictionary with application components
    """
    # Get vector store from components
    vector_store = app_components.get("vector_store")
    if vector_store:
        # Clear vector store index
        vector_store.clear_index()
    
    # Reset state
    state_manager.set("documents_processed", False)
    state_manager.set("documents", [])
    state_manager.set("iep_results", [])
    state_manager.set("lesson_plans", [])
    
    # Update system state
    state_manager.update_system_state(
        vector_store_initialized=False
    )
    
    st.rerun()

def render_system_status(app_components: Dict[str, Any]):
    """Render the system status section in the sidebar.
    
    Args:
        app_components: Dictionary with application components
    """
    # Get system state from state manager
    system_state = state_manager.get_system_state()
    
    with st.expander("System Status", expanded=False):
        st.write("### System Components Status")
        
        components = [
            ("LLM Client", system_state.get("llm_initialized", False)),
            ("Vector Store", system_state.get("vector_store_initialized", False)),
            ("RAG Chain", system_state.get("chain_initialized", False)),
            ("Document Count", len(state_manager.get("documents", [])) > 0)
        ]
        
        for component, is_healthy in components:
            if is_healthy:
                st.success(f"✅ {component}: OK")
            else:
                st.error(f"❌ {component}: Failed")
