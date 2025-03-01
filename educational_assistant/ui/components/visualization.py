"""Data visualization components for educational analytics."""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from config.logging_config import get_module_logger
from ui.state_manager import state_manager
from ui.components.common import display_error, display_success

# Create a logger for this module
logger = get_module_logger("visualization_component")

def render_analytics_tab(app_components: Dict[str, Any]):
    """Render the analytics and visualization tab.
    
    Args:
        app_components: Dictionary with application components
    """
    st.header("Educational Analytics")
    
    # Get available lesson plans and IEPs
    lesson_plans = state_manager.get("lesson_plans", [])
    iep_results = state_manager.get("iep_results", [])
    
    if not lesson_plans and not iep_results:
        st.info("No lesson plans or IEPs available for analysis. Please generate them first.")
        return
    
    # Create tabs for different visualizations
    viz_tabs = st.tabs(["Overview", "Timeline", "Goals", "Accommodations"])
    
    with viz_tabs[0]:
        render_overview_dashboard(lesson_plans, iep_results)
    
    with viz_tabs[1]:
        render_timeline_visualization(lesson_plans, iep_results)
    
    with viz_tabs[2]:
        render_goals_analysis(lesson_plans, iep_results)
    
    with viz_tabs[3]:
        render_accommodations_analysis(lesson_plans, iep_results)

def render_overview_dashboard(lesson_plans: List[Dict[str, Any]], iep_results: List[Dict[str, Any]]):
    """Render the overview dashboard with key metrics.
    
    Args:
        lesson_plans: List of lesson plan dictionaries
        iep_results: List of IEP result dictionaries
    """
    st.subheader("Educational Overview Dashboard")
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total IEPs",
            value=len(iep_results),
            delta=None
        )
    
    with col2:
        st.metric(
            label="Total Lesson Plans",
            value=len(lesson_plans),
            delta=None
        )
    
    with col3:
        # Calculate average goals per lesson plan
        total_goals = sum(len(plan.get("specific_goals", [])) for plan in lesson_plans)
        avg_goals = total_goals / max(len(lesson_plans), 1)
        
        st.metric(
            label="Avg. Goals per Plan",
            value=f"{avg_goals:.1f}",
            delta=None
        )
    
    with col4:
        # Calculate unique subjects
        subjects = set(plan.get("subject", "Unknown") for plan in lesson_plans)
        
        st.metric(
            label="Unique Subjects",
            value=len(subjects),
            delta=None
        )
    
    # Create simple charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Subject distribution chart
        subject_counts = {}
        for plan in lesson_plans:
            subject = plan.get("subject", "Unknown")
            subject_counts[subject] = subject_counts.get(subject, 0) + 1
        
        if subject_counts:
            df = pd.DataFrame({
                'Subject': list(subject_counts.keys()),
                'Count': list(subject_counts.values())
            })
            
            st.subheader("Lesson Plans by Subject")
            st.bar_chart(df.set_index('Subject'))
    
    with col2:
        # Plans by grade level
        grade_counts = {}
        for plan in lesson_plans:
            grade = plan.get("grade_level", "Unknown")
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
        
        if grade_counts:
            df = pd.DataFrame({
                'Grade Level': list(grade_counts.keys()),
                'Count': list(grade_counts.values())
            })
            
            st.subheader("Lesson Plans by Grade Level")
            st.bar_chart(df.set_index('Grade Level'))

def render_timeline_visualization(lesson_plans: List[Dict[str, Any]], iep_results: List[Dict[str, Any]]):
    """Render timeline visualization of activities.
    
    Args:
        lesson_plans: List of lesson plan dictionaries
        iep_results: List of IEP result dictionaries
    """
    st.subheader("Activity Timeline")
    
    # Combine and prepare data
    timeline_data = []
    
    for iep in iep_results:
        try:
            timestamp = datetime.fromisoformat(iep.get("timestamp", datetime.now().isoformat()))
            timeline_data.append({
                "date": timestamp,
                "type": "IEP",
                "title": f"IEP for {iep.get('source', 'Unknown')}"
            })
        except (ValueError, TypeError):
            pass
    
    for plan in lesson_plans:
        try:
            timestamp = datetime.fromisoformat(plan.get("timestamp", datetime.now().isoformat()))
            timeline_data.append({
                "date": timestamp,
                "type": "Lesson Plan",
                "title": f"{plan.get('subject', 'Unknown')} for {plan.get('grade_level', 'Unknown')}"
            })
        except (ValueError, TypeError):
            pass
    
    # Sort by date
    timeline_data.sort(key=lambda x: x["date"])
    
    if not timeline_data:
        st.info("No timeline data available.")
        return
    
    # Create DataFrame for visualization
    df = pd.DataFrame(timeline_data)
    
    # Create timeline chart
    fig_data = []
    
    for activity_type in df["type"].unique():
        type_data = df[df["type"] == activity_type]
        
        # Count activities per day
        daily_counts = type_data.groupby(type_data["date"].dt.date).size().reset_index()
        daily_counts.columns = ["date", "count"]
        
        # Add to figure data
        fig_data.append({
            "date": daily_counts["date"].tolist(),
            "count": daily_counts["count"].tolist(),
            "type": activity_type
        })
    
    # Create line chart with streamlit
    chart_data = pd.DataFrame()
    
    # Find date range
    all_dates = []
    for data in fig_data:
        all_dates.extend(data["date"])
    
    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
        
        # Create date range
        date_range = [min_date + timedelta(days=i) for i in range((max_date - min_date).days + 1)]
        
        # Create data for each type
        for data in fig_data:
            counts = []
            for date in date_range:
                try:
                    idx = data["date"].index(date)
                    counts.append(data["count"][idx])
                except ValueError:
                    counts.append(0)
            
            chart_data[data["type"]] = counts
        
        chart_data["date"] = date_range
        chart_data = chart_data.set_index("date")
        
        st.line_chart(chart_data)
    else:
        st.info("No timeline data available for chart.")

def render_goals_analysis(lesson_plans: List[Dict[str, Any]], iep_results: List[Dict[str, Any]]):
    """Render analysis of educational goals.
    
    Args:
        lesson_plans: List of lesson plan dictionaries
        iep_results: List of IEP result dictionaries
    """
    st.subheader("Educational Goals Analysis")
    
    # Extract goals from lesson plans
    all_goals = []
    for plan in lesson_plans:
        goals = plan.get("specific_goals", [])
        subject = plan.get("subject", "Unknown")
        grade = plan.get("grade_level", "Unknown")
        
        for goal in goals:
            if goal and isinstance(goal, str):
                all_goals.append({
                    "goal": goal,
                    "subject": subject,
                    "grade": grade,
                    "words": len(goal.split())
                })
    
    if not all_goals:
        st.info("No goals data available for analysis.")
        return
    
    # Create DataFrame for analysis
    df = pd.DataFrame(all_goals)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Goals",
            value=len(df),
            delta=None
        )
    
    with col2:
        st.metric(
            label="Avg. Goal Length (words)",
            value=f"{df['words'].mean():.1f}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Subjects with Goals",
            value=len(df["subject"].unique()),
            delta=None
        )
    
    # Create visualization of goal distribution by subject
    goal_counts = df.groupby("subject").size().reset_index()
    goal_counts.columns = ["Subject", "Goal Count"]
    
    st.subheader("Goals by Subject")
    st.bar_chart(goal_counts.set_index("Subject"))
    
    # Word length distribution
    st.subheader("Goal Complexity (Word Count Distribution)")
    
    # Create histogram data
    hist_data = np.histogram(
        df["words"], 
        bins=range(0, max(df["words"]) + 5, 5)
    )
    
    hist_df = pd.DataFrame({
        "Word Count Range": [f"{i}-{i+4}" for i in range(0, max(df["words"]) + 5, 5)][:-1],
        "Frequency": hist_data[0]
    })
    
    st.bar_chart(hist_df.set_index("Word Count Range"))

def render_accommodations_analysis(lesson_plans: List[Dict[str, Any]], iep_results: List[Dict[str, Any]]):
    """Render analysis of accommodations.
    
    Args:
        lesson_plans: List of lesson plan dictionaries
        iep_results: List of IEP result dictionaries
    """
    st.subheader("Accommodations Analysis")
    
    # Extract accommodations from lesson plans
    all_accommodations = []
    for plan in lesson_plans:
        accommodations = plan.get("additional_accommodations", [])
        subject = plan.get("subject", "Unknown")
        grade = plan.get("grade_level", "Unknown")
        
        for accommodation in accommodations:
            if accommodation and isinstance(accommodation, str):
                all_accommodations.append({
                    "accommodation": accommodation,
                    "subject": subject,
                    "grade": grade
                })
    
    if not all_accommodations:
        st.info("No accommodations data available for analysis.")
        return
    
    # Create DataFrame for analysis
    df = pd.DataFrame(all_accommodations)
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Total Accommodations",
            value=len(df),
            delta=None
        )
    
    with col2:
        st.metric(
            label="Avg. Accommodations per Plan",
            value=f"{len(df) / max(len(lesson_plans), 1):.1f}",
            delta=None
        )
    
    # Create visualization of accommodations by subject
    accommodation_counts = df.groupby("subject").size().reset_index()
    accommodation_counts.columns = ["Subject", "Accommodation Count"]
    
    st.subheader("Accommodations by Subject")
    st.bar_chart(accommodation_counts.set_index("Subject"))
    
    # Accommodation word cloud (simplified version with bar chart)
    st.subheader("Common Accommodation Types")
    
    # Simple categorization of accommodations
    categories = {
        "visual": ["visual", "diagram", "picture", "image", "chart", "graph"],
        "audio": ["audio", "sound", "listen", "verbal", "speech"],
        "time": ["time", "extended", "additional", "break", "pause"],
        "physical": ["seat", "position", "move", "stand", "fidget"],
        "assistive": ["device", "technology", "computer", "calculator", "aid"]
    }
    
    category_counts = {cat: 0 for cat in categories}
    other_count = 0
    
    for _, row in df.iterrows():
        accommodation = row["accommodation"].lower()
        
        categorized = False
        for cat, keywords in categories.items():
            if any(keyword in accommodation for keyword in keywords):
                category_counts[cat] += 1
                categorized = True
                break
        
        if not categorized:
            other_count += 1
    
    # Add "other" category
    category_counts["other"] = other_count
    
    # Create bar chart
    cat_df = pd.DataFrame({
        "Category": list(category_counts.keys()),
        "Count": list(category_counts.values())
    })
    
    st.bar_chart(cat_df.set_index("Category"))
