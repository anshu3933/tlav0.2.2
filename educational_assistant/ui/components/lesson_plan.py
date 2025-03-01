"""Lesson plan generation component."""

import streamlit as st
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from config.logging_config import get_module_logger
from ui.state_manager import state_manager
from ui.components.common import display_error, display_success, create_download_button, format_timestamp

# Create a logger for this module
logger = get_module_logger("lesson_plan_component")

def render_lesson_plan_tab(app_components: Dict[str, Any]):
    """Render the lesson plan generation tab.
    
    Args:
        app_components: Dictionary with application components
    """
    st.header("Lesson Plan Generation")
    
    # Combined form for lesson plan generation
    st.subheader("Lesson Plan Details")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_lesson_plan_form(app_components)
    
    # Display generated lesson plans
    display_lesson_plans()

def render_lesson_plan_form(app_components: Dict[str, Any]):
    """Render the lesson plan generation form.
    
    Args:
        app_components: Dictionary with application components
    """
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
                display_error("Please fill in all required fields.")
            else:
                handle_lesson_plan_generation(
                    subject, grade_level, timeframe, duration, days_per_week,
                    specific_goals, materials, additional_accommodations,
                    selected_iep, app_components
                )

def handle_lesson_plan_generation(
    subject: str, grade_level: str, timeframe: str, duration: str, days_per_week: List[str],
    specific_goals: str, materials: str, additional_accommodations: str,
    selected_iep_id: str, app_components: Dict[str, Any]
):
    """Handle lesson plan generation with proper error handling.
    
    Args:
        subject: Subject area
        grade_level: Grade level
        timeframe: Timeframe (Daily or Weekly)
        duration: Duration of lesson
        days_per_week: Days of the week
        specific_goals: Specific learning goals
        materials: Required materials
        additional_accommodations: Additional accommodations
        selected_iep_id: Selected IEP ID
        app_components: Dictionary with application components
    """
    try:
        # Find selected IEP
        iep_results = state_manager.get("iep_results", [])
        selected_iep = next((iep for iep in iep_results if iep["id"] == selected_iep_id), None)
        
        if not selected_iep:
            display_error("Selected IEP not found.")
            return
        
        # Get LLM client from components
        llm_client = app_components.get("llm_client")
        
        if not llm_client:
            display_error("LLM client not initialized. Cannot generate lesson plan.")
            return
        
        # Prepare input data
        goals_list = specific_goals.strip().split("\n")
        materials_list = materials.strip().split("\n") if materials else []
        accommodations_list = additional_accommodations.strip().split("\n") if additional_accommodations else []
        
        # Prepare prompt for lesson plan generation
        prompt = prepare_lesson_plan_prompt(
            subject, grade_level, timeframe, duration, days_per_week,
            goals_list, materials_list, accommodations_list,
            selected_iep
        )
        
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
            display_error("Failed to generate lesson plan.")
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
        
        display_success("Lesson plan generated successfully!")
        st.rerun()
            
    except Exception as e:
        logger.error(f"Error generating lesson plan: {str(e)}", exc_info=True)
        display_error(f"Error generating lesson plan: {str(e)}")

def prepare_lesson_plan_prompt(
    subject: str, grade_level: str, timeframe: str, duration: str, days_per_week: List[str],
    goals_list: List[str], materials_list: List[str], accommodations_list: List[str],
    selected_iep: Dict[str, Any]
) -> str:
    """Prepare the prompt for lesson plan generation.
    
    Args:
        subject: Subject area
        grade_level: Grade level
        timeframe: Timeframe (Daily or Weekly)
        duration: Duration of lesson
        days_per_week: Days of the week
        goals_list: Specific learning goals
        materials_list: Required materials
        accommodations_list: Additional accommodations
        selected_iep: Selected IEP dictionary
        
    Returns:
        Formatted prompt string
    """
    return f"""
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
                st.markdown(f"**Created:** {format_timestamp(plan.get('timestamp', datetime.now().isoformat()))}")
                
                # Display content
                st.markdown("### Lesson Plan Content")
                st.markdown(plan.get("content", "No content available"))
                
                # Download button
                create_download_button(
                    content=plan.get("content", ""),
                    filename=f"lesson_plan_{plan.get('subject', 'untitled').lower().replace(' ', '_')}_{plan.get('timeframe', 'unknown').lower()}.txt",
                    display_text=f"Download Plan {i+1}"
                )
