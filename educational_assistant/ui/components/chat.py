"""Chat interface component."""

import streamlit as st
from typing import Dict, Any, List, Optional
from config.logging_config import get_module_logger
from ui.state_manager import state_manager
from ui.components.common import display_error

# Create a logger for this module
logger = get_module_logger("chat_component")

def render_chat_tab(app_components: Dict[str, Any]):
    """Render the chat interface tab.
    
    Args:
        app_components: Dictionary with application components
    """
    st.header("Chat with your documents")
    
    # Initialize chat history if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    display_chat_history()
    
    # Chat input
    handle_chat_input(app_components)
    
    # Add clear chat button
    if st.session_state.messages and st.button("Clear Chat History"):
        state_manager.set("messages", [])
        st.rerun()

def display_chat_history():
    """Display the chat message history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"Source {i}:")
                        st.write(source.page_content)
                        if source.metadata.get('source'):
                            st.write(f"Source: {source.metadata['source']}")
                        st.write("---")

def handle_chat_input(app_components: Dict[str, Any]):
    """Handle chat input and generate responses.
    
    Args:
        app_components: Dictionary with application components
    """
    if prompt := st.chat_input("Ask a question about the documents..."):
        # Add user message to chat history
        state_manager.append("messages", {"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get RAG chain from components
                    rag_chain = app_components.get("rag_chain")
                    
                    if rag_chain:
                        # Run RAG chain
                        response = rag_chain.run(prompt)
                        
                        # Format response
                        message_data = {
                            "role": "assistant",
                            "content": response["result"],
                            "sources": response.get("source_documents", [])
                        }
                    else:
                        # Fallback response if chain not available
                        message_data = {
                            "role": "assistant",
                            "content": "I can help answer questions about documents once they're uploaded. For now, I can assist with general educational questions.",
                            "sources": []
                        }
                    
                    # Add to chat history
                    state_manager.append("messages", message_data)
                    
                    # Display response
                    st.markdown(message_data["content"])
                    
                    # Display sources if available
                    if message_data["sources"]:
                        with st.expander("View Sources"):
                            for i, doc in enumerate(message_data["sources"], 1):
                                st.write(f"Source {i}:")
                                st.write(doc.page_content)
                                if doc.metadata.get('source'):
                                    st.write(f"Source: {doc.metadata['source']}")
                                st.write("---")
                                
                except Exception as e:
                    logger.error(f"Error generating response: {str(e)}", exc_info=True)
                    error_message = {
                        "role": "assistant",
                        "content": f"I encountered an error while processing your question. Please try again.",
                        "sources": []
                    }
                    state_manager.append("messages", error_message)
                    st.markdown(error_message["content"])
