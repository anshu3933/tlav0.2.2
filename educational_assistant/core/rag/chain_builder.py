# core/rag/chain_builder.py

from typing import List, Dict, Any, Optional, Callable, Union
from config.app_config import config
from config.logging_config import get_module_logger
from core.llm.llm_client import LLMClient
from core.rag.retriever import HybridRetriever, WebAugmentedRetriever
from core.embeddings.vector_store import FAISSVectorStore

# Create a logger for this module
logger = get_module_logger("rag_chain_builder")

class RAGPromptBuilder:
    """Handles building and configuring RAG prompts."""
    
    @staticmethod
    def build_education_prompt() -> str:
        """Build the education-focused RAG prompt template.
        
        Returns:
            Prompt template
        """
        return """You are a helpful AI assistant specializing in special education, learning disabilities, learning design and IEPs. Use the following pieces of context to answer the question. 
If the context doesn't contain all the information needed, you can:
1. Use the relevant parts of the context that are available
2. Combine it with your general knowledge about education and IEPs
3. Clearly indicate which parts of your response are from the context and which are general knowledge

Context:
{context}

Question: {question}

Please provide a detailed answer citing specific information from the context when available:"""

    @staticmethod
    def build_general_prompt() -> str:
        """Build the general-purpose RAG prompt template.
        
        Returns:
            Prompt template
        """
        return """Use the following pieces of context to answer the question. If the information isn't contained in the context, say so clearly rather than making up an answer.

Context:
{context}

Question: {question}

Answer:"""


class RAGChain:
    """RAG chain for document-based question answering."""
    
    def __init__(self, 
                llm_client: Optional[LLMClient] = None,
                retriever: Optional[Union[HybridRetriever, Callable]] = None,
                prompt_template: Optional[str] = None,
                use_web_search: bool = False):
        """Initialize with components.
        
        Args:
            llm_client: LLM client for generating responses
            retriever: Document retriever
            prompt_template: Prompt template for RAG
            use_web_search: Whether to use web search
        """
        self.llm_client = llm_client or LLMClient()
        
        # Setup retriever if not provided
        if retriever is None:
            vector_store = FAISSVectorStore()
            if not vector_store.load_index():
                logger.warning("No index available for retrieval")
            
            if use_web_search:
                retriever = WebAugmentedRetriever(vector_store=vector_store)
            else:
                retriever = HybridRetriever(vector_store=vector_store)
        
        # If retriever is a HybridRetriever, get the retriever function
        if isinstance(retriever, (HybridRetriever, WebAugmentedRetriever)):
            self.retriever = retriever.as_retriever()
        else:
            self.retriever = retriever
        
        # Set prompt template
        self.prompt_template = prompt_template or RAGPromptBuilder.build_education_prompt()
        
        logger.debug("Initialized RAG chain")
    
    def run(self, query: str) -> Dict[str, Any]:
        """Run the RAG chain on a query.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with response and source documents
        """
        try:
            # Retrieve relevant documents
            documents = self.retriever(query)
            
            # Format context
            context = self._format_context(documents)
            
            # Prepare messages for LLM
            messages = [
                {"role": "system", "content": "You are an AI assistant that provides educational guidance."},
                {"role": "user", "content": self._format_prompt(query, context)}
            ]
            
            # Call LLM
            response = self.llm_client.chat_completion(messages)
            
            # Return result with source documents
            return {
                "result": response["content"],
                "source_documents": documents,
                "usage": response.get("usage", {})
            }
            
        except Exception as e:
            logger.error(f"Error in RAG chain: {str(e)}", exc_info=True)
            return {
                "result": f"I encountered an error while processing your question: {str(e)}",
                "source_documents": [],
                "error": str(e)
            }
    
    def _format_context(self, documents: List[Any]) -> str:
        """Format retrieved documents into context string.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."
        
        # Format each document
        formatted_docs = []
        for i, doc in enumerate(documents):
            # Get document content
            content = getattr(doc, "page_content", str(doc)) if hasattr(doc, "page_content") else str(doc)
            
            # Get metadata
            metadata = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
            source = metadata.get("source", f"Document {i+1}")
            
            # Format document
            formatted_docs.append(f"[Document {i+1}] Source: {source}\n{content}")
        
        # Join documents with separator
        return "\n\n".join(formatted_docs)
    
    def _format_prompt(self, query: str, context: str) -> str:
        """Format the prompt with query and context.
        
        Args:
            query: User query
            context: Context from retrieved documents
            
        Returns:
            Formatted prompt
        """
        return self.prompt_template.format(
            context=context,
            question=query
        )


class RAGChainBuilder:
    """Builder for creating and configuring RAG chains."""
    
    @staticmethod
    def build(
        llm_client: Optional[LLMClient] = None,
        vector_store: Optional[FAISSVectorStore] = None,
        k_documents: int = None,
        use_web_search: bool = False,
        max_web_results: int = 3,
        prompt_type: str = "education"
    ) -> RAGChain:
        """Build a RAG chain with configuration.
        
        Args:
            llm_client: LLM client
            vector_store: Vector store for document retrieval
            k_documents: Number of documents to retrieve
            use_web_search: Whether to use web search
            max_web_results: Maximum number of web search results
            prompt_type: Type of prompt template to use
            
        Returns:
            Configured RAG chain
        """
        # Create LLM client if not provided
        llm_client = llm_client or LLMClient()
        
        # Create vector store if not provided
        vector_store = vector_store or FAISSVectorStore()
        if not vector_store.load_index():
            logger.warning("No index available for retrieval")
        
        # Use configured k if not specified
        k_documents = k_documents or config.vector_store.similarity_top_k
        
        # Create retriever
        if use_web_search:
            retriever = WebAugmentedRetriever(
                vector_store=vector_store,
                k_documents=k_documents,
                max_web_results=max_web_results
            )
        else:
            retriever = HybridRetriever(
                vector_store=vector_store,
                k_documents=k_documents
            )
        
        # Get prompt template
        if prompt_type == "education":
            prompt_template = RAGPromptBuilder.build_education_prompt()
        else:
            prompt_template = RAGPromptBuilder.build_general_prompt()
        
        # Create and return chain
        return RAGChain(
            llm_client=llm_client,
            retriever=retriever,
            prompt_template=prompt_template,
            use_web_search=use_web_search
        )
