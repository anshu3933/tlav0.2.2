# ui/state_manager.py

import streamlit as st
import sqlite3
import json
import pickle
import os
import threading
import uuid
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, TypeVar, Generic, Callable, Union
from config.logging_config import get_module_logger

# Create a logger for this module
logger = get_module_logger("state_manager")

T = TypeVar('T')

class StateValidationError(Exception):
    """Exception raised for state validation errors."""
    pass

class PersistentStorage:
    """Interface for persistent storage backends."""
    
    def save(self, key: str, value: Any) -> bool:
        """Save value to storage.
        
        Args:
            key: Storage key
            value: Value to store
            
        Returns:
            Success status
        """
        raise NotImplementedError
    
    def load(self, key: str) -> Optional[Any]:
        """Load value from storage.
        
        Args:
            key: Storage key
            
        Returns:
            Stored value or None
        """
        raise NotImplementedError
    
    def delete(self, key: str) -> bool:
        """Delete value from storage.
        
        Args:
            key: Storage key
            
        Returns:
            Success status
        """
        raise NotImplementedError
    
    def list_keys(self) -> List[str]:
        """List all keys in storage.
        
        Returns:
            List of keys
        """
        raise NotImplementedError


class SQLiteStorage(PersistentStorage):
    """SQLite-based persistent storage."""
    
    def __init__(self, db_path: str = ".state/state.db"):
        """Initialize with database path.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_db()
        
        logger.debug(f"Initialized SQLite storage at {db_path}")
    
    def _init_db(self) -> None:
        """Initialize database with schema."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create state table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    updated_at TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
    
    def save(self, key: str, value: Any) -> bool:
        """Save value to SQLite storage.
        
        Args:
            key: Storage key
            value: Value to store
            
        Returns:
            Success status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Pickle the value
            pickled_value = pickle.dumps(value)
            
            # Insert or update
            cursor.execute(
                "INSERT OR REPLACE INTO state (key, value, updated_at) VALUES (?, ?, ?)",
                (key, pickled_value, datetime.now().isoformat())
            )
            
            conn.commit()
            conn.close()
            
            return True
        except Exception as e:
            logger.error(f"Failed to save to SQLite: {str(e)}")
            return False
    
    def load(self, key: str) -> Optional[Any]:
        """Load value from SQLite storage.
        
        Args:
            key: Storage key
            
        Returns:
            Stored value or None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT value FROM state WHERE key=?", (key,))
            result = cursor.fetchone()
            
            conn.close()
            
            if result:
                return pickle.loads(result[0])
            
            return None
        except Exception as e:
            logger.error(f"Failed to load from SQLite: {str(e)}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete value from SQLite storage.
        
        Args:
            key: Storage key
            
        Returns:
            Success status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM state WHERE key=?", (key,))
            
            conn.commit()
            conn.close()
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete from SQLite: {str(e)}")
            return False
    
    def list_keys(self) -> List[str]:
        """List all keys in SQLite storage.
        
        Returns:
            List of keys
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT key FROM state")
            results = cursor.fetchall()
            
            conn.close()
            
            return [result[0] for result in results]
        except Exception as e:
            logger.error(f"Failed to list keys from SQLite: {str(e)}")
            return []


class SessionState(Generic[T]):
    """Thread-safe session state management."""
    
    def __init__(self, 
                init_value: T, 
                validator: Optional[Callable[[T], bool]] = None,
                storage: Optional[PersistentStorage] = None,
                storage_key: Optional[str] = None):
        """Initialize with initial value and optional validator.
        
        Args:
            init_value: Initial value
            validator: Optional validation function
            storage: Optional persistent storage backend
            storage_key: Key for persistent storage
        """
        self.value = init_value
        self.validator = validator
        self.lock = threading.RLock()
        self.storage = storage
        self.storage_key = storage_key
        
        # Load from storage if available
        self._load_from_storage()
    
    def _load_from_storage(self) -> None:
        """Load value from storage if available."""
        if self.storage and self.storage_key:
            stored_value = self.storage.load(self.storage_key)
            if stored_value is not None:
                with self.lock:
                    self.value = stored_value
    
    def _save_to_storage(self) -> None:
        """Save value to storage if available."""
        if self.storage and self.storage_key:
            self.storage.save(self.storage_key, self.value)
    
    def get(self) -> T:
        """Get the current value (thread-safe).
        
        Returns:
            Current value
        """
        with self.lock:
            return self.value
    
    def set(self, new_value: T) -> None:
        """Set a new value with validation (thread-safe).
        
        Args:
            new_value: New value
            
        Raises:
            StateValidationError: If validation fails
        """
        with self.lock:
            if self.validator and not self.validator(new_value):
                raise StateValidationError(f"Invalid value: {new_value}")
            self.value = new_value
            self._save_to_storage()


class AppStateManager:
    """Manages application state with validation, persistence, and session management."""
    
    def __init__(self, storage_backend: Optional[PersistentStorage] = None):
        """Initialize state manager.
        
        Args:
            storage_backend: Optional persistent storage backend
        """
        # Create default storage if not provided
        self.storage = storage_backend or SQLiteStorage()
        
        # Session ID for user-specific storage
        self.session_id = self._get_or_create_session_id()
        
        # Initialize session state
        self._initialize_session_state()
        
        logger.debug(f"Initialized app state manager with session ID: {self.session_id}")
    
    def _get_or_create_session_id(self) -> str:
        """Get existing session ID or create a new one.
        
        Returns:
            Session ID
        """
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        
        return st.session_state.session_id
    
    def _get_storage_key(self, key: str) -> str:
        """Get storage key with session ID prefix.
        
        Args:
            key: Original key
            
        Returns:
            Storage key with session ID prefix
        """
        return f"{self.session_id}:{key}"
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state with default values."""
        defaults = {
            "user_id": self._generate_user_id(),
            "session_id": self.session_id,
            "session_start": datetime.now().isoformat(),
            "documents_processed": False,
            "documents": [],
            "iep_results": [],
            "lesson_plans": [],
            "messages": [],
            "current_plan": None,
            "errors": [],
            "warnings": [],
            "system_state": {
                "chain_initialized": False,
                "vector_store_initialized": False,
                "llm_initialized": False
            }
        }
        
        # Initialize each key if not exists
        for key, default_value in defaults.items():
            if key not in st.session_state:
                # Try to load from storage first
                storage_key = self._get_storage_key(key)
                stored_value = self.storage.load(storage_key)
                
                if stored_value is not None:
                    st.session_state[key] = stored_value
                else:
                    st.session_state[key] = default_value
    
    def _generate_user_id(self) -> str:
        """Generate a stable user ID based on session properties.
        
        Returns:
            User ID
        """
        # For demo purposes, just generate a random ID
        # In production, this could be linked to authentication
        return str(uuid.uuid4())
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from session state.
        
        Args:
            key: State key
            default: Default value if key doesn't exist
            
        Returns:
            Value from session state
        """
        return st.session_state.get(key, default)
    
    def set(self, key: str, value: Any, persist: bool = True) -> None:
        """Set a value in session state.
        
        Args:
            key: State key
            value: Value to set
            persist: Whether to persist to storage
        """
        st.session_state[key] = value
        
        # Persist to storage if requested
        if persist:
            storage_key = self._get_storage_key(key)
            self.storage.save(storage_key, value)
    
    def update(self, key: str, update_func: Callable[[Any], Any], persist: bool = True) -> None:
        """Update a value in session state using a function.
        
        Args:
            key: State key
            update_func: Function that takes old value and returns new value
            persist: Whether to persist to storage
        """
        if key in st.session_state:
            old_value = st.session_state[key]
            new_value = update_func(old_value)
            self.set(key, new_value, persist=persist)
    
    def append(self, key: str, value: Any, persist: bool = True) -> None:
        """Append a value to a list in session state.
        
        Args:
            key: State key (must be a list)
            value: Value to append
            persist: Whether to persist to storage
            
        Raises:
            TypeError: If the key doesn't exist or isn't a list
        """
        if key not in st.session_state:
            st.session_state[key] = []
            
        if not isinstance(st.session_state[key], list):
            raise TypeError(f"Key '{key}' is not a list")
            
        st.session_state[key].append(value)
        
        # Persist to storage if requested
        if persist:
            storage_key = self._get_storage_key(key)
            self.storage.save(storage_key, st.session_state[key])
    
    def clear(self, key: Optional[str] = None, persist: bool = True) -> None:
        """Clear a specific key or all session state.
        
        Args:
            key: Optional key to clear (None clears all)
            persist: Whether to persist to storage
        """
        if key is None:
            # Preserve user ID and session info when clearing
            user_id = st.session_state.get("user_id")
            session_id = st.session_state.get("session_id")
            session_start = st.session_state.get("session_start")
            
            # Get list of keys to clear
            keys_to_clear = list(st.session_state.keys())
            
            # Clear each key
            for k in keys_to_clear:
                if k not in {"user_id", "session_id", "session_start"}:
                    del st.session_state[k]
                    
                    # Remove from storage if requested
                    if persist:
                        storage_key = self._get_storage_key(k)
                        self.storage.delete(storage_key)
            
            # Restore user and session info
            st.session_state["user_id"] = user_id
            st.session_state["session_id"] = session_id
            st.session_state["session_start"] = session_start
            
            # Reinitialize with defaults
            self._initialize_session_state()
        elif key in st.session_state:
            del st.session_state[key]
            
            # Remove from storage if requested
            if persist:
                storage_key = self._get_storage_key(key)
                self.storage.delete(storage_key)
    
    def add_error(self, error: str) -> None:
        """Add an error message.
        
        Args:
            error: Error message
        """
        timestamp = datetime.now().isoformat()
        self.append("errors", {"message": error, "timestamp": timestamp})
        logger.error(f"UI Error: {error}")
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message.
        
        Args:
            warning: Warning message
        """
        timestamp = datetime.now().isoformat()
        self.append("warnings", {"message": warning, "timestamp": timestamp})
        logger.warning(f"UI Warning: {warning}")
    
    def has_errors(self) -> bool:
        """Check if there are any errors.
        
        Returns:
            True if there are errors, False otherwise
        """
        return len(self.get("errors", [])) > 0
    
    def get_latest_error(self) -> Optional[Dict[str, str]]:
        """Get the latest error message.
        
        Returns:
            Latest error or None if no errors
        """
        errors = self.get("errors", [])
        return errors[-1] if errors else None
    
    def clear_errors(self) -> None:
        """Clear all error messages."""
        self.set("errors", [])
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings.
        
        Returns:
            True if there are warnings, False otherwise
        """
        return len(self.get("warnings", [])) > 0
    
    def get_system_state(self) -> Dict[str, bool]:
        """Get the system state.
        
        Returns:
            System state dictionary
        """
        return self.get("system_state", {
            "chain_initialized": False,
            "vector_store_initialized": False,
            "llm_initialized": False
        })
    
    def update_system_state(self, **kwargs) -> None:
        """Update the system state.
        
        Args:
            **kwargs: System state key-value pairs
        """
        system_state = self.get("system_state", {})
        system_state.update(kwargs)
        self.set("system_state", system_state)
    
    def export_state(self, include_system_state: bool = False) -> Dict[str, Any]:
        """Export session state for persistence.
        
        Args:
            include_system_state: Whether to include system state
            
        Returns:
            Dictionary with session state
        """
        # Get all keys except for large objects and system state
        excluded_keys = {"_lock", "vector_store", "chain", "llm_client"}
        if not include_system_state:
            excluded_keys.add("system_state")
        
        # Copy state
        state_copy = {}
        for key, value in st.session_state.items():
            if key not in excluded_keys:
                state_copy[key] = value
        
        return state_copy
    
    def import_state(self, state_dict: Dict[str, Any]) -> None:
        """Import session state from a dictionary.
        
        Args:
            state_dict: Dictionary with session state
        """
        for key, value in state_dict.items():
            self.set(key, value)
    
    def restore_session(self, session_id: str) -> bool:
        """Restore a previous session by ID.
        
        Args:
            session_id: Session ID to restore
            
        Returns:
            Success status
        """
        try:
            # Get list of keys for the session
            session_keys = []
            for key in self.storage.list_keys():
                if key.startswith(f"{session_id}:"):
                    session_keys.append(key)
            
            if not session_keys:
                logger.warning(f"No session found with ID: {session_id}")
                return False
            
            # Update session ID
            old_session_id = self.session_id
            self.session_id = session_id
            st.session_state.session_id = session_id
            
            # Restore each key
            for storage_key in session_keys:
                # Extract the original key
                key = storage_key.split(":", 1)[1] if ":" in storage_key else storage_key
                
                # Load value from storage
                value = self.storage.load(storage_key)
                
                if value is not None:
                    st.session_state[key] = value
            
            logger.info(f"Restored session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore session: {str(e)}")
            return False
    
    def list_saved_sessions(self) -> List[Dict[str, Any]]:
        """List all saved sessions.
        
        Returns:
            List of session information dictionaries
        """
        sessions = {}
        
        for key in self.storage.list_keys():
            if ":" in key:
                session_id, key_name = key.split(":", 1)
                
                if key_name == "session_start":
                    # Get session start time
                    start_time = self.storage.load(key)
                    
                    if session_id not in sessions:
                        sessions[session_id] = {
                            "id": session_id,
                            "start_time": start_time
                        }
        
        # Convert to list and sort by start time (newest first)
        session_list = list(sessions.values())
        session_list.sort(key=lambda s: s.get("start_time", ""), reverse=True)
        
        return session_list
    
    def get_session_info(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a session.
        
        Args:
            session_id: Session ID (or current session if None)
            
        Returns:
            Session information dictionary
        """
        session_id = session_id or self.session_id
        
        # Get list of keys for the session
        session_keys = []
        for key in self.storage.list_keys():
            if key.startswith(f"{session_id}:"):
                session_keys.append(key.split(":", 1)[1])
        
        # Get session start time
        start_time = self.storage.load(f"{session_id}:session_start")
        
        # Count documents, IEPs, and lesson plans
        documents = self.storage.load(f"{session_id}:documents") or []
        iep_results = self.storage.load(f"{session_id}:iep_results") or []
        lesson_plans = self.storage.load(f"{session_id}:lesson_plans") or []
        
        return {
            "id": session_id,
            "start_time": start_time,
            "key_count": len(session_keys),
            "document_count": len(documents),
            "iep_count": len(iep_results),
            "lesson_plan_count": len(lesson_plans),
            "keys": session_keys
        }


# Create a global state manager instance with SQLite persistence
state_manager = AppStateManager(SQLiteStorage())