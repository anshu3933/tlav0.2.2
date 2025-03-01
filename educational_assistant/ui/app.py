
# ui/app.py

import streamlit as st
import os
from typing import List, Dict, Any
from datetime import datetime
import uuid

# Import configuration and state management
from config.app_config import config
from config.logging_config import get_module_logger
from ui.state_manager import state_manager

# Import core functionality
from core.document_processing.file_handler import FileHandler, FileHandlerError
from core.document_processing.document_loader import DocumentLoader
from core.embeddings.vector_store import FAISSVectorStore
from core.rag.chain_builder import RAGChainBuilder

# Import main initialization
from main import load_app_components

# Create a logger for this module
logger = get_module_logger("streamlit_app")

# Page Configuration
st.set_page_config(
    page_title="Educational Assistant",
    page_icon=":books:",
    layout="wide"
)

def run_app():
    """Main function to run the application."""
    try:
        # Display app header
        st.title("Educational Assistant")
        
        # Initialize application components
        app_components = load_app_components()
        
        if not app_components:
            st.error("Failed to initialize application. Please check the logs.")
            return
        
        # Create sidebar for API Key and File Upload
        create_sidebar(app_components)
        
        # Create main content tabs
        tab1, tab2, tab3 = st.tabs(["Chat", "IEP Generation", "Lesson Plans"])
        
        # Populate tabs
        with tab1:
            create_chat_tab(app_components)
        
        with tab2:
            create_iep_tab(app_components)
        
        with tab3:
            create_lesson_plan_tab(app_components)
            
        # Add footer
        st.markdown("---")
        st.markdown("Educational Assistant")
        
    except Exception as e:
        logger.error(f"Error in main app: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")

def create_sidebar(app_components: Dict[str, Any]):
    """Create sidebar with file upload and system status."""
    with st.sidebar:
        st.title("Document Upload")
        
        # Create file uploader
        uploaded_files = st.file_uploader(
            "Upload educational documents",
            type=["txt", "docx", "pdf", "md", "csv", "json", "xlsx"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            process_uploaded_files(uploaded_files, app_components)
        
        # Add clear documents button
        if state_manager.get("documents_processed", False):
            if st.button("Clear Documents"):
                clear_documents(app_components)
        
        # System status section
        st.title("System Status")
        display_system_status(app_components)

def process_uploaded_files(uploaded_files, app_components: Dict[str, Any]):
    """Process uploaded files and update state."""
    # Get vector store from components
    vector_store = app_components.get("vector_store")
    if not vector_store:
        st.error("Vector store not initialized. Cannot process documents.")
        return
    
    # Initialize handlers
    file_handler = FileHandler()
    document_loader = DocumentLoader()
    
    # Process only if not already processed
    if not state_manager.get("documents_processed", False):
        processing_success = True
        
        with st.spinner("Processing documents..."):
            st.write("### Processing Files")
            
            for file in uploaded_files:
                status_container = st.empty()
                status_container.info(f"Processing {file.name}...")
                
                try:
                    # Process uploaded file
                    uploaded_file = file_handler.process_uploaded_file(file)
                    
                    # Load document
                    result = document_loader.load_single_document(uploaded_file.temp_path)
                    
                    if not result.success:
                        status_container.error(f"Error processing {file.name}: {result.error_message}")
                        processing_success = False
                        continue
                    
                    # Add to state
                    document = result.document
                    document.metadata["source"] = file.name
                    document.metadata["upload_time"] = datetime.now().isoformat()
                    
                    state_manager.append("documents", document)
                    
                    # Add to vector store
                    vector_store.add_documents([document])
                    
                    status_container.success(f"Successfully processed {file.name}")
                    
                except FileHandlerError as e:
                    status_container.error(f"Error handling {file.name}: {str(e)}")
                    processing_success = False
                except Exception as e:
                    logger.error(f"Unexpected error processing {file.name}: {str(e)}", exc_info=True)
                    status_container.error(f"Unexpected error processing {file.name}")
                    processing_success = False
            
            if processing_success:
                state_manager.set("documents_processed", True)
                st.success("All documents processed successfully!")
                
                # Update system state
                state_manager.update_system_state(
                    vector_store_initialized=True
                )
            else:
                st.error("Error processing some documents.")
    
    # Clean up temporary files
    file_handler.cleanup()

def clear_documents(app_components: Dict[str, Any]):
    """Clear documents and reset state."""
    # Get vector store from components
    vector_store = app_components.get("vector_store")
    if vector_store:
        # Clear vector store index
        vector_store.clear_index()
    
    # Reset state
    state_manager.set("documents_processed", False)
    state_manager.set("documents", [])
    state_manager.set("iep_results", [])
    state_manager.set("lesson_plans", [])
    
    # Update system state
    state_manager.update_system_state(
        vector_store_initialized=False
    )
    
    st.rerun()

def display_system_status(app_components: Dict[str, Any]):
    """Display system component status."""
    # Get system state from state manager
    system_state = state_manager.get_system_state()
    
    with st.expander("System Status", expanded=False):
        st.write("### System Components Status")
        
        components = [
            ("LLM Client", system_state.get("llm_initialized", False)),
            ("Vector Store", system_state.get("vector_store_initialized", False)),
            ("RAG Chain", system_state.get("chain_initialized", False)),
            ("Document Count", len(state_manager.get("documents", [])) > 0)
        ]
        
        for component, is_healthy in components:
            if is_healthy:
                st.success(f"✅ {component}: OK")
            else:
                st.error(f"❌ {component}: Failed")

def create_chat_tab(app_components: Dict[str, Any]):
    """Create chat interface tab."""
    st.header("Chat with your documents")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
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
    
    # Chat input
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
    
    # Add clear chat button
    if st.session_state.messages and st.button("Clear Chat History"):
        state_manager.set("messages", [])
        st.rerun()

def create_iep_tab(app_components: Dict[str, Any]):
    """Create IEP generation tab."""
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
        
        if st.button("Generate IEP"):
            handle_iep_generation(selected_doc, app_components)
    else:
        st.info("No documents available. Please upload a document.")
    
    # Display existing IEPs
    display_existing_ieps()

def handle_iep_generation(selected_doc: Dict[str, Any], app_components: Dict[str, Any]):
    """Handle IEP generation from selected document."""
    with st.spinner("Generating IEP..."):
        try:
            # Get document
            document = get_document_by_id(selected_doc["id"])
            
            if not document:
                st.error("Could not retrieve document for processing.")
                return
            
            # Check if pipeline component is available
            if "iep_pipeline" in app_components:
                # Use the pipeline component
                iep_pipeline = app_components["iep_pipeline"]
                iep_result = iep_pipeline.generate_iep(document)
            else:
                # Get LLM client from components
                llm_client = app_components.get("llm_client")
                
                if not llm_client:
                    st.error("LLM client not initialized. Cannot generate IEP.")
                    return
                
                # Generate IEP content using LLM
                messages = [
                    {"role": "system", "content": "You are an AI assistant that specializes in creating Individualized Education Programs (IEPs) for students with special needs."},
                    {"role": "user", "content": f"Based on the following document, create a comprehensive IEP with appropriate goals, accommodations, and services. Document content: {document.page_content}"}
                ]
                
                # Call LLM
                response = llm_client.chat_completion(messages)
                
                if not response or "content" not in response:
                    st.error("Failed to generate IEP content.")
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
            st.error(f"Error generating IEP: {str(e)}")

def display_iep_content(iep_result: Dict[str, Any]):
    """Display formatted IEP content with download option."""
    content = iep_result["content"]
    
    # Display formatted content
    st.subheader(f"IEP for {iep_result['source']}")
    st.markdown(content)
    
    # Add download option
    if st.button("Download IEP as Text"):
        st.download_button(
            label="Download IEP",
            data=content,
            file_name=f"IEP_{iep_result['source']}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

def display_existing_ieps():
    """Display existing IEPs in collapsible sections."""
    iep_results = state_manager.get("iep_results", [])
    
    if iep_results:
        st.markdown("### Previously Generated IEPs")
        
        for i, iep in enumerate(iep_results):
            with st.expander(f"IEP {i+1}: {iep.get('source', 'Unknown')}"):
                st.markdown(iep.get("content", "No content available"))
                
                # Add download button
                st.download_button(
                    label=f"Download IEP {i+1}",
                    data=iep.get("content", ""),
                    file_name=f"IEP_{iep.get('source', 'unknown').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    key=f"download_iep_{i}"
                )

def get_available_documents() -> List[Dict[str, Any]]:
    """Get list of all available documents from session state."""
    documents = []
    
    # Get documents from session state
    state_docs = state_manager.get("documents", [])
    for i, doc in enumerate(state_docs):
        # Ensure document has an ID
        doc_id = doc.metadata.get("id", f"doc_{i}")
        
        # Create display name with icon
        source = doc.metadata.get("source", "Untitled Document")
        doc_type = doc.metadata.get("document_type", "unknown")
        
        # Select icon based on document type
        if doc_type == "pdf" or source.lower().endswith(".pdf"):
            icon = "📄"
        elif doc_type == "document" or source.lower().endswith((".docx", ".doc")):
            icon = "📝"
        elif doc_type == "spreadsheet" or source.lower().endswith((".csv", ".xlsx")):
            icon = "📊"
        elif doc_type == "data" or source.lower().endswith(".json"):
            icon = "📋"
        else:
            icon = "📄"
            
        documents.append({
            "id": doc_id,
            "display_name": f"{icon} {source}",
            "source": source
        })
    
    return documents

def get_document_by_id(doc_id: str):
    """Get document by ID from session state."""
    documents = state_manager.get("documents", [])
    for doc in documents:
        if doc.metadata.get("id") == doc_id:
            return doc
    return None

def create_lesson_plan_tab(app_components: Dict[str, Any]):
    """Create lesson plan generation tab."""
    st.header("Lesson Plan Generation")
    
    # Combined form for all lesson plan generation
    st.subheader("Lesson Plan Details")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("lesson_plan_form"):
            # Required form fields
            st.markdown("### Basic Information")
            subject = st.text_input("Subject *", placeholder="e.g., Mathematics, Reading, Science")
            grade_level = st.text_input("Grade Level *", placeholder="e.g., 3rd Grade, High School")
            
            # Timeframe selection
            timeframe = st.radio(
                "Schedule Type *",
                ["Daily", "Weekly"],
                help="Choose between a daily lesson plan or a weekly schedule"
            )
            
            duration = st.text_input(
                "Daily Duration *",
                placeholder="e.g., 45 minutes per session"
            )
            
            if timeframe == "Weekly":
                days_per_week = st.multiselect(
                    "Select Days *",
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                    default=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                )
            else:
                days_per_week = ["Daily"]
            
            st.markdown("### Learning Details")
            specific_goals = st.text_area(
                "Specific Learning Goals *",
                placeholder="Enter specific goals for this lesson, one per line"
            )
            
            materials = st.text_area(
                "Materials Needed",
                placeholder="List required materials, one per line"
            )
            
            st.markdown("### Additional Support")
            additional_accommodations = st.text_area(
                "Additional Accommodations",
                placeholder="Enter any specific accommodations beyond those in the IEP"
            )
            
            # IEP Selection
            st.markdown("### IEP Integration")
            iep_results = state_manager.get("iep_results", [])
            
            if iep_results:
                selected_iep = st.selectbox(
                    "Select IEP to Integrate *",
                    options=[iep["id"] for iep in iep_results],
                    format_func=lambda x: f"IEP from {next((iep['source'] for iep in iep_results if iep['id'] == x), 'Unknown')}"
                )
            else:
                st.error("No IEPs available. Please generate an IEP first.")
                selected_iep = None
            
            st.markdown("*Required fields")
            
            generate_button = st.form_submit_button("Generate Enhanced Lesson Plan")

            if generate_button:
                if not all([subject, grade_level, duration, specific_goals, selected_iep]):
                    st.error("Please fill in all required fields.")
                else:
                    handle_lesson_plan_generation(
                        subject, grade_level, timeframe, duration, days_per_week,
                        specific_goals, materials, additional_accommodations,
                        selected_iep, app_components
                    )
    
    # Display generated lesson plans
    display_lesson_plans()

def handle_lesson_plan_generation(
    subject: str, grade_level: str, timeframe: str, duration: str, days_per_week: List[str],
    specific_goals: str, materials: str, additional_accommodations: str,
    selected_iep_id: str, app_components: Dict[str, Any]
):
    """Handle lesson plan generation with proper error handling."""
    try:
        # Find selected IEP
        iep_results = state_manager.get("iep_results", [])
        selected_iep = next((iep for iep in iep_results if iep["id"] == selected_iep_id), None)
        
        if not selected_iep:
            st.error("Selected IEP not found.")
            return
        
        # Prepare input data
        goals_list = specific_goals.strip().split("\n")
        materials_list = materials.strip().split("\n") if materials else []
        accommodations_list = additional_accommodations.strip().split("\n") if additional_accommodations else []

        # Check if pipeline component is available
        if "lesson_plan_pipeline" in app_components:
            # Use the pipeline component
            lesson_plan_pipeline = app_components["lesson_plan_pipeline"]
            
            with st.spinner("Generating lesson plan..."):
                plan_data = lesson_plan_pipeline.generate_lesson_plan(
                    subject=subject,
                    grade_level=grade_level,
                    timeframe=timeframe,
                    duration=duration,
                    days_per_week=days_per_week,
                    specific_goals=goals_list,
                    materials=materials_list,
                    additional_accommodations=accommodations_list,
                    iep_content=selected_iep["content"]
                )
                
                # Add source IEP reference
                plan_data["source_iep_id"] = selected_iep_id
                plan_data["source_iep_source"] = selected_iep["source"]
        else:
            # Get LLM client from components
            llm_client = app_components.get("llm_client")
            
            if not llm_client:
                st.error("LLM client not initialized. Cannot generate lesson plan.")
                return
            
            # Prepare prompt for lesson plan generation
            prompt = f"""
            Create a detailed {timeframe.lower()} lesson plan for {subject} for {grade_level} students.
            
            The plan should be based on the following IEP:
            {selected_iep['content']}
            
            Class details:
            - Subject: {subject}
            - Grade Level: {grade_level}
            - Duration: {duration}
            - Schedule: {', '.join(days_per_week) if timeframe == 'Weekly' else 'Daily'}
            
            Learning Goals:
            {chr(10).join(f'- {goal}' for goal in goals_list if goal)}
            
            Materials Needed:
            {chr(10).join(f'- {item}' for item in materials_list if item)}
            
            Additional Accommodations:
            {chr(10).join(f'- {acc}' for acc in accommodations_list if acc)}
            
            Please create a comprehensive lesson plan with:
            1. Learning objectives
            2. Detailed schedule/timeline
            3. Teaching strategies with specific IEP accommodations
            4. Assessment methods
            5. Resources and materials organization
            
            Format the plan clearly with sections and bullet points where appropriate.
            """
            
            # Call LLM for lesson plan generation
            messages = [
                {"role": "system", "content": "You are an AI assistant specialized in creating educational lesson plans that accommodate students with special needs."},
                {"role": "user", "content": prompt}
            ]
            
            with st.spinner("Generating lesson plan..."):
                response = llm_client.chat_completion(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=4000
                )
            
            if not response or "content" not in response:
                st.error("Failed to generate lesson plan.")
                return
            
            # Create plan data structure
            plan_data = {
                "id": str(uuid.uuid4()),
                # Input data
                "subject": subject,
                "grade_level": grade_level,
                "duration": duration,
                "timeframe": timeframe,
                "days": days_per_week,
                "specific_goals": goals_list,
                "materials": materials_list,
                "additional_accommodations": accommodations_list,
                # Generated content
                "content": response["content"],
                # Metadata
                "source_iep_id": selected_iep_id,
                "source_iep_source": selected_iep["source"],
                "timestamp": datetime.now().isoformat()
            }
        
        # Save to state
        state_manager.append("lesson_plans", plan_data)
        
        # Set as current plan
        state_manager.set("current_plan", plan_data)
        
        st.success("Lesson plan generated successfully!")
        st.rerun()
            
    except Exception as e:
        logger.error(f"Error generating lesson plan: {str(e)}", exc_info=True)
        st.error(f"Error generating lesson plan: {str(e)}")

def display_lesson_plans():
    """Display generated lesson plans."""
    lesson_plans = state_manager.get("lesson_plans", [])
    
    if lesson_plans:
        st.markdown("### Generated Lesson Plans")
        
        for i, plan in enumerate(lesson_plans):
            with st.expander(f"Lesson Plan {i+1}: {plan.get('subject', 'Untitled')} - {plan.get('timeframe', 'Unknown')} ({plan.get('grade_level', 'Unspecified')})", expanded=(i == 0)):
                # Display basic info
                st.markdown(f"**Subject:** {plan.get('subject', 'Not specified')}")
                st.markdown(f"**Grade Level:** {plan.get('grade_level', 'Not specified')}")
                st.markdown(f"**Duration:** {plan.get('duration', 'Not specified')}")
                st.markdown(f"**Created:** {datetime.fromisoformat(plan.get('timestamp', datetime.now().isoformat())).strftime('%Y-%m-%d %H:%M')}")
                
                # Display content
                st.markdown("### Lesson Plan Content")
                st.markdown(plan.get("content", "No content available"))
                
                # Download button
                st.download_button(
                    label=f"Download Plan {i+1}",
                    data=plan.get("content", ""),
                    file_name=f"lesson_plan_{plan.get('subject', 'untitled').lower().replace(' ', '_')}_{plan.get('timeframe', 'unknown').lower()}.txt",
                    mime="text/plain",
                    key=f"download_plan_{i}"
                )

# Run the app
if __name__ == "__main__":
    run_app()
