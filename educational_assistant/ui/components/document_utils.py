"""Utility functions for document handling in the UI."""

import streamlit as st
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from langchain.schema import Document

from config.logging_config import get_module_logger
from ui.state_manager import state_manager
from ui.components.common import display_error, display_info

# Create a logger for this module
logger = get_module_logger("document_utils")

def get_available_documents() -> List[Dict[str, Any]]:
    """Get list of all available documents from all sources.
    
    Returns:
        List of document dictionaries with id, display_name, and source
    """
    documents = []
    
    # Get documents from session state
    state_docs = state_manager.get("documents", [])
    for i, doc in enumerate(state_docs):
        # Ensure document has an ID
        doc_id = doc.metadata.get("id", f"doc_{i}")
        
        # Create display name with icon
        source = doc.metadata.get("source", "Untitled Document")
        doc_type = doc.metadata.get("document_type", "unknown")
        
        # Select icon based on document type
        if doc_type == "pdf":
            icon = "📄"
        elif doc_type == "document":
            icon = "📝"
        elif doc_type == "spreadsheet":
            icon = "📊"
        elif doc_type == "data":
            icon = "📋"
        else:
            icon = "📄"
            
        documents.append({
            "id": doc_id,
            "display_name": f"{icon} {source}",
            "source": source,
            "timestamp": doc.metadata.get("timestamp", None)
        })
    
    # Sort by timestamp if available (newest first)
    documents.sort(
        key=lambda x: x.get("timestamp", ""),
        reverse=True
    )
    
    return documents

def get_document_by_id(doc_id: str) -> Optional[Document]:
    """Get document by ID from session state.
    
    Args:
        doc_id: Document ID
        
    Returns:
        Document object or None if not found
    """
    documents = state_manager.get("documents", [])
    for doc in documents:
        if doc.metadata.get("id") == doc_id:
            return doc
    return None

def get_document_metadata(doc_id: str) -> Dict[str, Any]:
    """Get document metadata by ID.
    
    Args:
        doc_id: Document ID
        
    Returns:
        Document metadata dictionary
    """
    doc = get_document_by_id(doc_id)
    if doc:
        return doc.metadata
    return {}

def format_document_preview(document: Document, max_length: int = 500) -> str:
    """Format document content for preview with truncation.
    
    Args:
        document: Document to preview
        max_length: Maximum preview length
        
    Returns:
        Formatted preview text
    """
    content = document.page_content
    
    if len(content) > max_length:
        preview = content[:max_length] + "..."
    else:
        preview = content
    
    return preview

def display_document_preview(document: Document):
    """Display a document preview in the UI.
    
    Args:
        document: Document to display
    """
    if not document:
        display_info("No document selected")
        return
    
    metadata = document.metadata
    
    # Display document information
    st.subheader(f"Document: {metadata.get('source', 'Untitled')}")
    
    # Display metadata
    with st.expander("Document Metadata", expanded=False):
        st.write(f"**Source:** {metadata.get('source', 'Unknown')}")
        st.write(f"**Type:** {metadata.get('file_type', 'Unknown')}")
        st.write(f"**Size:** {metadata.get('file_size_mb', 0)} MB")
        
        if "last_modified" in metadata:
            timestamp = datetime.fromtimestamp(metadata["last_modified"])
            st.write(f"**Last Modified:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display preview
    st.markdown("### Content Preview")
    st.markdown(format_document_preview(document))
    
    # Display full content in expander
    with st.expander("View Full Content", expanded=False):
        st.markdown(document.page_content)
