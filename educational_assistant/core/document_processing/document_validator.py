# core/document_processing/document_validator.py
import os
from typing import Tuple, List, Optional
from config.app_config import config
from config.logging_config import get_module_logger

# Create a logger for this module
logger = get_module_logger("document_validator")

class DocumentValidationError(Exception):
    """Exception raised when document validation fails."""
    pass

class DocumentValidator:
    """Handles document validation with detailed error reporting."""
    
    def __init__(self):
        """Initialize validator with configuration."""
        self.supported_extensions = set(config.document.supported_formats)
        self.max_file_size = config.document.max_file_size_mb * 1024 * 1024  # Convert to bytes
    
    def validate_file_path(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """Validate a file path with detailed error reporting.
        
        Args:
            file_path: The path to the file
            
        Returns:
            A tuple of (is_valid, error_message)
        """
        # Check file existence
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False, f"File not found: {file_path}"
        
        # Check file extension
        extension = os.path.splitext(file_path)[1].lower()
        if extension not in self.supported_extensions:
            logger.error(f"Unsupported file type: {extension}")
            return False, f"Unsupported file type: {extension}. Supported types: {', '.join(self.supported_extensions)}"
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size:
            logger.error(f"File too large: {file_path}, size: {file_size} bytes, max: {self.max_file_size} bytes")
            return False, f"File too large. Maximum file size is {config.document.max_file_size_mb}MB."
        
        # Check if file is empty
        if file_size == 0:
            logger.error(f"Empty file: {file_path}")
            return False, "File is empty."
        
        return True, None
    
    def validate_content(self, content: str) -> Tuple[bool, Optional[str]]:
        """Validate document content with detailed error reporting.
        
        Args:
            content: The content to validate
            
        Returns:
            A tuple of (is_valid, error_message)
        """
        if not content:
            logger.error("Content is None")
            return False, "No content provided."
        
        content_stripped = content.strip()
        if not content_stripped:
            logger.error("Content is empty")
            return False, "Document content is empty."
        
        # Check if content is too short to be meaningful
        if len(content_stripped) < 50:
            logger.warning(f"Content is very short: {len(content_stripped)} characters")
            return True, "Warning: Document content is very short and may not provide enough context."
        
        return True, None
    
    def validate_uploaded_file(self, uploaded_file) -> Tuple[bool, Optional[str]]:
        """Validate an uploaded file with detailed error reporting.
        
        Args:
            uploaded_file: The uploaded file object
            
        Returns:
            A tuple of (is_valid, error_message)
        """
        # Check file name
        if not hasattr(uploaded_file, 'name') or not uploaded_file.name:
            logger.error("Uploaded file has no name")
            return False, "Uploaded file has no name."
        
        # Check file extension
        extension = os.path.splitext(uploaded_file.name)[1].lower()
        if extension not in self.supported_extensions:
            logger.error(f"Unsupported file type: {extension}")
            return False, f"Unsupported file type: {extension}. Supported types: {', '.join(self.supported_extensions)}"
        
        # Check file size
        if hasattr(uploaded_file, 'size') and uploaded_file.size > self.max_file_size:
            logger.error(f"File too large: {uploaded_file.name}, size: {uploaded_file.size} bytes")
            return False, f"File too large. Maximum file size is {config.document.max_file_size_mb}MB."
        
        # Try to read the file content
        try:
            content = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer
            
            # Check if file is empty
            if not content:
                logger.error(f"Empty file: {uploaded_file.name}")
                return False, "File is empty."
                
        except Exception as e:
            logger.error(f"Error reading uploaded file: {str(e)}")
            return False, f"Error reading file: {str(e)}"
        
        return True, None

    def suggest_fixes(self, error_message: str) -> str:
        """Suggest fixes for common validation errors.
        
        Args:
            error_message: The error message
            
        Returns:
            A suggestion message
        """
        if "Unsupported file type" in error_message:
            return f"Please convert your file to one of the supported formats: {', '.join(self.supported_extensions)}"
            
        if "File too large" in error_message:
            return f"Try splitting your file into smaller parts, each under {config.document.max_file_size_mb}MB."
            
        if "File is empty" in error_message:
            return "Please check that your file contains text content."
            
        if "Error reading file" in error_message:
            return "The file may be corrupted. Try exporting it again from the source application."
            
        return "Please check your file and try again."
