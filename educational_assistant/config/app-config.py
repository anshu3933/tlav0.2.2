# config/app_config.py
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class LLMConfig:
    """Configuration for LLM integration."""
    api_key: str
    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 2000
    request_timeout: int = 60
    max_retries: int = 3
    rate_limit_rpm: int = 50  # Requests per minute
    cache_enabled: bool = True
    cache_ttl: int = 3600  # Cache time-to-live in seconds

@dataclass
class VectorStoreConfig:
    """Configuration for vector storage."""
    index_dir: str = "models/faiss_index"
    embedding_model: str = "text-embedding-ada-002"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    similarity_top_k: int = 4
    cache_embeddings: bool = True

@dataclass
class DocumentConfig:
    """Configuration for document processing."""
    data_dir: str = "data"
    supported_formats: list = None
    max_file_size_mb: int = 10
    extraction_timeout: int = 30
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.pdf', '.docx', '.txt']

@dataclass
class AppConfig:
    """Application configuration."""
    environment: str
    llm: LLMConfig
    vector_store: VectorStoreConfig
    document: DocumentConfig
    debug: bool = False
    
    @classmethod
    def from_environment(cls) -> 'AppConfig':
        """Create configuration from environment variables."""
        # Get API key safely
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        # Get environment
        environment = os.getenv("APP_ENV", "development")
        
        # Create LLM config
        llm_config = LLMConfig(
            api_key=api_key,
            model_name=os.getenv("LLM_MODEL", "gpt-4o"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2000")),
            request_timeout=int(os.getenv("LLM_TIMEOUT", "60")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
            rate_limit_rpm=int(os.getenv("LLM_RATE_LIMIT", "50")),
            cache_enabled=os.getenv("LLM_CACHE_ENABLED", "true").lower() == "true",
            cache_ttl=int(os.getenv("LLM_CACHE_TTL", "3600"))
        )
        
        # Create vector store config
        vector_config = VectorStoreConfig(
            index_dir=os.getenv("VECTOR_INDEX_DIR", "models/faiss_index"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            similarity_top_k=int(os.getenv("SIMILARITY_TOP_K", "4")),
            cache_embeddings=os.getenv("CACHE_EMBEDDINGS", "true").lower() == "true"
        )
        
        # Create document config
        document_config = DocumentConfig(
            data_dir=os.getenv("DATA_DIR", "data"),
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "10")),
            extraction_timeout=int(os.getenv("EXTRACTION_TIMEOUT", "30"))
        )
        
        # Create app config
        return cls(
            environment=environment,
            llm=llm_config,
            vector_store=vector_config,
            document=document_config,
            debug=os.getenv("DEBUG", "false").lower() == "true"
        )

# Default configuration singleton
config = AppConfig.from_environment()
