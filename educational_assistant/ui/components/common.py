"""Common UI components shared across the application."""

import streamlit as st
from datetime import datetime
from typing import Optional, Dict, Any, List

def set_page_config():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="Educational Assistant",
        page_icon=":books:",
        layout="wide"
    )

def render_header():
    """Render the application header."""
    st.title("Educational Assistant")

def render_footer():
    """Render the application footer."""
    st.markdown("---")
    st.markdown("Educational Assistant")
    
def display_error(message: str):
    """Display an error message."""
    st.error(message)
    
def display_success(message: str):
    """Display a success message."""
    st.success(message)
    
def display_info(message: str):
    """Display an information message."""
    st.info(message)
    
def display_warning(message: str):
    """Display a warning message."""
    st.warning(message)

def create_download_button(content: str, filename: str, display_text: str = "Download"):
    """Create a download button for text content.
    
    Args:
        content: Text content to download
        filename: Name of the downloaded file
        display_text: Button display text
    """
    return st.download_button(
        label=display_text,
        data=content,
        file_name=filename,
        mime="text/plain"
    )

def format_timestamp(timestamp: str) -> str:
    """Format an ISO timestamp into a human-readable string.
    
    Args:
        timestamp: ISO format timestamp
        
    Returns:
        Formatted timestamp string
    """
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return timestamp
