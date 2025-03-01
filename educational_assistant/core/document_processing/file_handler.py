# core/document_processing/file_handler.py

import os
import tempfile
import shutil
import uuid
from typing import List, Optional, Dict, Any, Union
from contextlib import contextmanager
from config.logging_config import get_module_logger
from core.document_processing.document_validator import DocumentValidator
from config.app_config import config

# Create a logger for this module
logger = get_module_logger("file_handler")

class FileHandlerError(Exception):
    """Exception raised for file handling errors."""
    pass

class UploadedFile:
    """Represents an uploaded file with metadata."""
    
    def __init__(self, temp_path: str, original_name: str, file_type: str, size: int):
        """Initialize with file information."""
        self.temp_path = temp_path
        self.original_name = original_name
        self.file_type = file_type
        self.size = size
        self.uuid = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "uuid": self.uuid,
            "original_name": self.original_name,
            "file_type": self.file_type,
            "size": self.size,
            "temp_path": self.temp_path
        }


class FileHandler:
    """Handles file operations with proper cleanup and error handling."""
    
    def __init__(self):
        """Initialize with validator and tracking."""
        self.validator = DocumentValidator()
        self.temp_files: List[str] = []
        self.data_dir = config.document.data_dir
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
    
    def process_uploaded_file(self, uploaded_file) -> Optional[UploadedFile]:
        """Process an uploaded file and return its information.
        
        Args:
            uploaded_file: The uploaded file object
            
        Returns:
            UploadedFile object or None if failed
            
        Raises:
            FileHandlerError: If file validation or processing fails
        """
        try:
            # Validate uploaded file
            is_valid, error_message = self.validator.validate_uploaded_file(uploaded_file)
            if not is_valid:
                raise FileHandlerError(error_message)
                
            # Get file extension
            extension = os.path.splitext(uploaded_file.name)[1].lower()
            if not extension:
                extension = '.txt'  # Default extension
            
            # Create temporary file with proper extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
                # Write uploaded file content to temporary file
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name
            
            # Add to tracking for cleanup
            self.temp_files.append(temp_path)
            
            # Create and return upload info
            return UploadedFile(
                temp_path=temp_path,
                original_name=uploaded_file.name,
                file_type=extension,
                size=uploaded_file.size if hasattr(uploaded_file, 'size') else os.path.getsize(temp_path)
            )
                
        except FileHandlerError as e:
            # Re-raise specific error
            raise
        except Exception as e:
            logger.error(f"Error processing uploaded file: {str(e)}", exc_info=True)
            raise FileHandlerError(f"Failed to process uploaded file: {str(e)}")
    
    def save_file_to_data_dir(self, temp_path: str, filename: str = None) -> str:
        """Save a temporary file to the data directory.
        
        Args:
            temp_path: Path to the temporary file
            filename: Optional custom filename
            
        Returns:
            Path to the saved file
            
        Raises:
            FileHandlerError: If file saving fails
        """
        try:
            if not os.path.exists(temp_path):
                raise FileHandlerError(f"Temporary file not found: {temp_path}")
            
            # Generate filename if not provided
            if not filename:
                basename = os.path.basename(temp_path)
                filename = f"{uuid.uuid4()}_{basename}"
            
            # Ensure filename has the correct extension
            ext = os.path.splitext(temp_path)[1].lower()
            if not filename.lower().endswith(ext):
                filename = f"{os.path.splitext(filename)[0]}{ext}"
            
            # Create destination path
            dest_path = os.path.join(self.data_dir, filename)
            
            # Copy file
            shutil.copy2(temp_path, dest_path)
            
            logger.debug(f"Saved file to {dest_path}")
            return dest_path
            
        except Exception as e:
            logger.error(f"Error saving file to data directory: {str(e)}", exc_info=True)
            raise FileHandlerError(f"Failed to save file: {str(e)}")
    
    def cleanup(self):
        """Clean up all temporary files."""
        for temp_path in self.temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    logger.debug(f"Removed temporary file: {temp_path}")
            except Exception as e:
                logger.error(f"Error cleaning up temp file {temp_path}: {str(e)}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
    
    @contextmanager
    def temporary_directory(self) -> str:
        """Context manager for creating and cleaning up a temporary directory.
        
        Yields:
            Path to a temporary directory
        """
        temp_dir = tempfile.mkdtemp()
        try:
            yield temp_dir
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @contextmanager
    def temporary_file(self, suffix: str = None) -> str:
        """Context manager for creating and cleaning up a temporary file.
        
        Args:
            suffix: Optional file extension
            
        Yields:
            Path to a temporary file
        """
        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            os.close(fd)
            yield path
        finally:
            try:
                os.unlink(path)
            except:
                pass