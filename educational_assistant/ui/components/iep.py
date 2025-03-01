"""IEP generation component."""

import streamlit as st
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from config.logging_config import get_module_logger
from ui.state_manager import state_manager
from ui.components.common import display_error, display_success, create_download_button, format_timestamp
from ui.components.document_utils import get_available_documents, get_document_by_id, display_document_preview

# Create a logger for this module
logger = get_module_logger("iep_component")

def render_iep_tab(app_components: Dict[str, Any]):
    """Render the IEP generation tab.
    
    Args:
        app_components: Dictionary with application components
    """
    st.header("IEP Generation")
    
    # Document Selection Section
    st.markdown("### Document Selection")
    
    # Get available documents
    available_documents = get_available_documents()
    
    if available_documents:
        selected_doc = st.selectbox(
            "Select document to process",
            options=available_documents,
            format_func=lambda x: x["display_name"],
            key="iep_doc_selector"
        )
        
        # Display document preview
        if selected_doc:
            document = get_document_by_id(selected_doc["id"])
            if document:
                display_document_preview(document)
        
        if st.button("Generate IEP"):
            handle_iep_generation(selected_doc, app_components)
    else:
        st.info("No documents available. Please upload a document.")
    
    # Display existing IEPs
    display_existing_ieps()

def handle_iep_generation(selected_doc: Dict[str, Any], app_components: Dict[str, Any]):
    """Handle IEP generation from selected document.
    
    Args:
        selected_doc: Selected document dictionary
        app_components: Dictionary with application components
    """
    with st.spinner("Generating IEP..."):
        try:
            # Get document
            document = get_document_by_id(selected_doc["id"])
            
            if not document:
                display_error("Could not retrieve document for processing.")
                return
            
            # Get LLM client from components
            llm_client = app_components.get("llm_client")
            
            if not llm_client:
                display_error("LLM client not initialized. Cannot generate IEP.")
                return
            
            # Generate IEP content using LLM
            messages = [
                {"role": "system", "content": "You are an AI assistant that specializes in creating Individualized Education Programs (IEPs) for students with special needs."},
                {"role": "user", "content": f"Based on the following document, create a comprehensive IEP with appropriate goals, accommodations, and services. Document content: {document.page_content}"}
            ]
            
            # Call LLM
            response = llm_client.chat_completion(messages)
            
            if not response or "content" not in response:
                display_error("Failed to generate IEP content.")
                return
            
            # Create IEP result
            iep_result = {
                "id": str(uuid.uuid4()),
                "source": selected_doc["display_name"],
                "source_id": selected_doc["id"],
                "content": response["content"],
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to state
            state_manager.append("iep_results", iep_result)
            
            # Display IEP content
            display_iep_content(iep_result)
            
        except Exception as e:
            logger.error(f"Error generating IEP: {str(e)}", exc_info=True)
            display_error(f"Error generating IEP: {str(e)}")

def display_iep_content(iep_result: Dict[str, Any]):
    """Display formatted IEP content with download option.
    
    Args:
        iep_result: IEP result dictionary
    """
    content = iep_result["content"]
    
    # Display formatted content
    st.subheader(f"IEP for {iep_result['source']}")
    st.markdown(content)
    
    # Add download option
    create_download_button(
        content=content,
        filename=f"IEP_{iep_result['source'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
        display_text="Download IEP as Text"
    )

def display_existing_ieps():
    """Display existing IEPs in collapsible sections."""
    iep_results = state_manager.get("iep_results", [])
    
    if iep_results:
        st.markdown("### Previously Generated IEPs")
        
        for i, iep in enumerate(iep_results):
            with st.expander(f"IEP {i+1}: {iep.get('source', 'Unknown')} ({format_timestamp(iep.get('timestamp', ''))})", expanded=False):
                st.markdown(iep.get("content", "No content available"))
                
                # Add download button
                create_download_button(
                    content=iep.get("content", ""),
                    filename=f"IEP_{iep.get('source', 'unknown').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                    display_text=f"Download IEP {i+1}"
                )
