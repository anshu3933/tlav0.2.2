"""Input validation utilities."""

import os
import re
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import json
from utils.error_handling import ValidationError

def validate_required(data: Dict[str, Any], required_fields: List[str]) -> Dict[str, str]:
    """Validate that all required fields are present and not empty.
    
    Args:
        data: Data dictionary to validate
        required_fields: List of required field names
        
    Returns:
        Dictionary of validation errors (field_name -> error_message)
    """
    errors = {}
    
    for field in required_fields:
        if field not in data:
            errors[field] = f"Field '{field}' is required"
        elif data[field] is None or (isinstance(data[field], str) and not data[field].strip()):
            errors[field] = f"Field '{field}' cannot be empty"
    
    return errors


def validate_field_type(value: Any, expected_type: Union[type, Tuple[type, ...]], field_name: str) -> Optional[str]:
    """Validate field type.
    
    Args:
        value: Field value to validate
        expected_type: Expected type or tuple of types
        field_name: Field name for error message
        
    Returns:
        Error message or None if valid
    """
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            type_names = [t.__name__ for t in expected_type]
            return f"Field '{field_name}' must be one of these types: {', '.join(type_names)}"
        else:
            return f"Field '{field_name}' must be of type {expected_type.__name__}"
    
    return None


def validate_string_length(value: str, min_length: int = 0, max_length: Optional[int] = None, field_name: str = "Field") -> Optional[str]:
    """Validate string length.
    
    Args:
        value: String value to validate
        min_length: Minimum length
        max_length: Maximum length (None for no maximum)
        field_name: Field name for error message
        
    Returns:
        Error message or None if valid
    """
    if not isinstance(value, str):
        return f"{field_name} must be a string"
    
    if len(value) < min_length:
        return f"{field_name} must be at least {min_length} characters"
    
    if max_length is not None and len(value) > max_length:
        return f"{field_name} must be no more than {max_length} characters"
    
    return None


def validate_numeric_range(value: Union[int, float], min_value: Optional[Union[int, float]] = None, max_value: Optional[Union[int, float]] = None, field_name: str = "Field") -> Optional[str]:
    """Validate numeric value range.
    
    Args:
        value: Numeric value to validate
        min_value: Minimum value (None for no minimum)
        max_value: Maximum value (None for no maximum)
        field_name: Field name for error message
        
    Returns:
        Error message or None if valid
    """
    if not isinstance(value, (int, float)):
        return f"{field_name} must be a number"
    
    if min_value is not None and value < min_value:
        return f"{field_name} must be at least {min_value}"
    
    if max_value is not None and value > max_value:
        return f"{field_name} must be no more than {max_value}"
    
    return None


def validate_regex(value: str, pattern: str, field_name: str = "Field") -> Optional[str]:
    """Validate string against a regex pattern.
    
    Args:
        value: String value to validate
        pattern: Regex pattern
        field_name: Field name for error message
        
    Returns:
        Error message or None if valid
    """
    if not isinstance(value, str):
        return f"{field_name} must be a string"
    
    if not re.match(pattern, value):
        return f"{field_name} has an invalid format"
    
    return None


def validate_email(email: str, field_name: str = "Email") -> Optional[str]:
    """Validate email format.
    
    Args:
        email: Email to validate
        field_name: Field name for error message
        
    Returns:
        Error message or None if valid
    """
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return validate_regex(email, pattern, field_name)


def validate_url(url: str, field_name: str = "URL") -> Optional[str]:
    """Validate URL format.
    
    Args:
        url: URL to validate
        field_name: Field name for error message
        
    Returns:
        Error message or None if valid
    """
    pattern = r"^(https?|ftp)://[^\s/$.?#].[^\s]*$"
    return validate_regex(url, pattern, field_name)


def validate_file_path(file_path: str, must_exist: bool = True, file_types: List[str] = None) -> Optional[str]:
    """Validate file path.
    
    Args:
        file_path: File path to validate
        must_exist: Whether the file must exist
        file_types: List of allowed file extensions (e.g., ['.pdf', '.txt'])
        
    Returns:
        Error message or None if valid
    """
    if not isinstance(file_path, str):
        return "File path must be a string"
    
    if must_exist and not os.path.exists(file_path):
        return f"File does not exist: {file_path}"
    
    if file_types:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in file_types:
            return f"Invalid file type. Allowed types: {', '.join(file_types)}"
    
    return None


def validate_json(json_str: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Validate JSON string.
    
    Args:
        json_str: JSON string to validate
        
    Returns:
        Tuple of (is_valid, parsed_json, error_message)
    """
    if not isinstance(json_str, str):
        return False, None, "Input must be a string"
    
    try:
        parsed = json.loads(json_str)
        return True, parsed, None
    except json.JSONDecodeError as e:
        return False, None, f"Invalid JSON: {str(e)}"


def validate_list_items(items: List[Any], validator: Callable[[Any], Optional[str]]) -> List[str]:
    """Validate each item in a list using a validator function.
    
    Args:
        items: List of items to validate
        validator: Function that takes an item and returns an error message or None
        
    Returns:
        List of error messages (empty if all valid)
    """
    if not isinstance(items, list):
        return ["Input must be a list"]
    
    errors = []
    for i, item in enumerate(items):
        error = validator(item)
        if error:
            errors.append(f"Item {i}: {error}")
    
    return errors


def validate_api_key(api_key: str, min_length: int = 8, field_name: str = "API Key") -> Optional[str]:
    """Validate API key.
    
    Args:
        api_key: API key to validate
        min_length: Minimum length
        field_name: Field name for error message
        
    Returns:
        Error message or None if valid
    """
    if not api_key:
        return f"{field_name} is required"
    
    return validate_string_length(api_key, min_length=min_length, field_name=field_name)


def validate_openai_api_key(api_key: str) -> Optional[str]:
    """Validate OpenAI API key format.
    
    Args:
        api_key: API key to validate
        
    Returns:
        Error message or None if valid
    """
    if not api_key:
        return "OpenAI API key is required"
    
    # OpenAI API keys typically start with "sk-" and are 51 characters long
    if not api_key.startswith("sk-"):
        return "OpenAI API key should start with 'sk-'"
    
    if len(api_key) < 40:
        return "OpenAI API key appears to be too short"
    
    return None


def validate_document_content(content: str, min_length: int = 10) -> Optional[str]:
    """Validate document content.
    
    Args:
        content: Document content to validate
        min_length: Minimum content length
        
    Returns:
        Error message or None if valid
    """
    if not content:
        return "Document content is empty"
    
    content_stripped = content.strip()
    if not content_stripped:
        return "Document content is empty (whitespace only)"
    
    if len(content_stripped) < min_length:
        return f"Document content is too short (minimum {min_length} characters)"
    
    return None
