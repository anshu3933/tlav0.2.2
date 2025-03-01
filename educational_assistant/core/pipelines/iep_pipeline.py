"""IEP generation pipeline."""

from typing import Dict, Any, List, Optional
from langchain.schema import Document
from datetime import datetime
import uuid

from config.logging_config import get_module_logger
from core.llm.llm_client import LLMClient

# Create a logger for this module
logger = get_module_logger("iep_pipeline")

class IEPGenerationPipeline:
    """Pipeline for generating IEPs from educational documents."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize with components.
        
        Args:
            llm_client: LLM client for generating IEPs
        """
        self.llm_client = llm_client or LLMClient()
        logger.debug("Initialized IEP generation pipeline")
    
    def generate_iep(self, document: Document) -> Dict[str, Any]:
        """Generate an IEP from a document.
        
        Args:
            document: Document to generate IEP from
            
        Returns:
            Generated IEP result dictionary
        """
        try:
            logger.debug(f"Generating IEP from document: {document.metadata.get('source', 'Unknown')}")
            
            # Build system prompt
            system_prompt = "You are an AI assistant that specializes in creating Individualized Education Programs (IEPs) for students with special needs."
            
            # Build user prompt with detailed instructions
            user_prompt = self._build_iep_prompt(document)
            
            # Generate IEP content using LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call LLM
            response = self.llm_client.chat_completion(messages)
            
            if not response or "content" not in response:
                logger.error("Failed to generate IEP content.")
                raise ValueError("Failed to generate IEP content")
            
            # Create IEP result
            iep_result = {
                "id": str(uuid.uuid4()),
                "source": document.metadata.get("source", "Unknown Document"),
                "source_id": document.metadata.get("id", ""),
                "content": response["content"],
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "model": response.get("model", "Unknown"),
                    "usage": response.get("usage", {})
                }
            }
            
            logger.info(f"Successfully generated IEP for document: {document.metadata.get('source', 'Unknown')}")
            return iep_result
            
        except Exception as e:
            logger.error(f"Error generating IEP: {str(e)}", exc_info=True)
            raise
    
    def _build_iep_prompt(self, document: Document) -> str:
        """Build detailed prompt for IEP generation.
        
        Args:
            document: Document to generate IEP from
            
        Returns:
            Formatted prompt string
        """
        return f"""
        Based on the following document, create a comprehensive Individualized Education Program (IEP) with appropriate goals, accommodations, and services.
        
        Include these sections in your IEP:
        1. Student Information
        2. Present Levels of Academic Achievement and Functional Performance
        3. Annual Goals and Short-Term Objectives
        4. Accommodations and Modifications
        5. Special Education and Related Services
        6. Assessment Information
        7. Transition Services (if appropriate)
        
        Document content:
        {document.page_content}
        
        Format the IEP in a clear, professional structure that would be useful to educators, parents, and students.
        """
    
    def analyze_document(self, document: Document) -> Dict[str, Any]:
        """Analyze a document to extract relevant IEP information.
        
        Args:
            document: Document to analyze
            
        Returns:
            Dictionary of extracted information
        """
        # This would contain more sophisticated analysis logic
        # For now, it's a placeholder for future enhancements
        try:
            logger.debug(f"Analyzing document for IEP information: {document.metadata.get('source', 'Unknown')}")
            
            # Build analysis prompt
            analysis_prompt = f"""
            Analyze the following document and extract information relevant for creating an IEP.
            Extract:
            1. Student information (name, age, grade)
            2. Current performance levels
            3. Areas of need
            4. Suggested goals
            5. Suggested accommodations
            
            Document content:
            {document.page_content}
            
            Format your response as a structured JSON object.
            """
            
            # Generate analysis using LLM
            messages = [
                {"role": "system", "content": "You are an expert in analyzing educational documents for IEP development."},
                {"role": "user", "content": analysis_prompt}
            ]
            
            # Call LLM
            response = self.llm_client.chat_completion(messages)
            
            if not response or "content" not in response:
                logger.error("Failed to analyze document.")
                return {}
            
            # For now, just return the raw content
            # In a real implementation, this would parse the JSON response
            return {
                "analysis": response["content"],
                "document_id": document.metadata.get("id", ""),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing document: {str(e)}", exc_info=True)
            return {}
