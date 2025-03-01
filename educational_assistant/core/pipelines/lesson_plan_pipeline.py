"""Lesson plan generation pipeline."""

from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from config.logging_config import get_module_logger
from core.llm.llm_client import LLMClient

# Create a logger for this module
logger = get_module_logger("lesson_plan_pipeline")

class LessonPlanGenerationPipeline:
    """Pipeline for generating lesson plans that incorporate IEP accommodations."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize with components.
        
        Args:
            llm_client: LLM client for generating lesson plans
        """
        self.llm_client = llm_client or LLMClient()
        logger.debug("Initialized lesson plan generation pipeline")
    
    def generate_lesson_plan(self, 
                          subject: str, 
                          grade_level: str, 
                          timeframe: str, 
                          duration: str,
                          days_per_week: List[str],
                          specific_goals: List[str],
                          materials: List[str],
                          additional_accommodations: List[str],
                          iep_content: str) -> Dict[str, Any]:
        """Generate a lesson plan incorporating IEP accommodations.
        
        Args:
            subject: Subject area
            grade_level: Grade level
            timeframe: Timeframe (Daily or Weekly)
            duration: Duration of lesson
            days_per_week: Days of the week
            specific_goals: Specific learning goals
            materials: Required materials
            additional_accommodations: Additional accommodations
            iep_content: IEP content to incorporate
            
        Returns:
            Generated lesson plan result dictionary
        """
        try:
            logger.debug(f"Generating {timeframe} lesson plan for {subject} ({grade_level})")
            
            # Build prompt for lesson plan generation
            prompt = self._build_lesson_plan_prompt(
                subject, grade_level, timeframe, duration, days_per_week,
                specific_goals, materials, additional_accommodations,
                iep_content
            )
            
            # Generate lesson plan using LLM
            messages = [
                {"role": "system", "content": "You are an AI assistant specialized in creating educational lesson plans that accommodate students with special needs."},
                {"role": "user", "content": prompt}
            ]
            
            # Call LLM
            response = self.llm_client.chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=4000
            )
            
            if not response or "content" not in response:
                logger.error("Failed to generate lesson plan.")
                raise ValueError("Failed to generate lesson plan")
            
            # Create plan data structure
            plan_data = {
                "id": str(uuid.uuid4()),
                # Input data
                "subject": subject,
                "grade_level": grade_level,
                "duration": duration,
                "timeframe": timeframe,
                "days": days_per_week,
                "specific_goals": specific_goals,
                "materials": materials,
                "additional_accommodations": additional_accommodations,
                # Generated content
                "content": response["content"],
                # Metadata
                "source_iep": iep_content,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "model": response.get("model", "Unknown"),
                    "usage": response.get("usage", {})
                }
            }
            
            logger.info(f"Successfully generated {timeframe} lesson plan for {subject} ({grade_level})")
            return plan_data
            
        except Exception as e:
            logger.error(f"Error generating lesson plan: {str(e)}", exc_info=True)
            raise
    
    def _build_lesson_plan_prompt(self,
                                subject: str, 
                                grade_level: str, 
                                timeframe: str, 
                                duration: str,
                                days_per_week: List[str],
                                specific_goals: List[str],
                                materials: List[str],
                                additional_accommodations: List[str],
                                iep_content: str) -> str:
        """Build detailed prompt for lesson plan generation.
        
        Args:
            subject: Subject area
            grade_level: Grade level
            timeframe: Timeframe (Daily or Weekly)
            duration: Duration of lesson
            days_per_week: Days of the week
            specific_goals: Specific learning goals
            materials: Required materials
            additional_accommodations: Additional accommodations
            iep_content: IEP content to incorporate
            
        Returns:
            Formatted prompt string
        """
        return f"""
        Create a detailed {timeframe.lower()} lesson plan for {subject} for {grade_level} students.
        
        The plan should be based on the following IEP:
        {iep_content}
        
        Class details:
        - Subject: {subject}
        - Grade Level: {grade_level}
        - Duration: {duration}
        - Schedule: {', '.join(days_per_week) if timeframe == 'Weekly' else 'Daily'}
        
        Learning Goals:
        {chr(10).join(f'- {goal}' for goal in specific_goals if goal)}
        
        Materials Needed:
        {chr(10).join(f'- {item}' for item in materials if item)}
        
        Additional Accommodations:
        {chr(10).join(f'- {acc}' for acc in additional_accommodations if acc)}
        
        Please create a comprehensive lesson plan with:
        1. Learning objectives
        2. Detailed schedule/timeline
        3. Teaching strategies with specific IEP accommodations
        4. Assessment methods
        5. Resources and materials organization
        
        Format the plan clearly with sections and bullet points where appropriate.
        """
    
    def analyze_iep_for_accommodations(self, iep_content: str, subject: str) -> List[str]:
        """Analyze an IEP to extract relevant accommodations for a subject.
        
        Args:
            iep_content: IEP content to analyze
            subject: Subject area to focus on
            
        Returns:
            List of relevant accommodations
        """
        try:
            logger.debug(f"Analyzing IEP for accommodations relevant to {subject}")
            
            # Build analysis prompt
            analysis_prompt = f"""
            Analyze the following IEP and extract accommodations that would be relevant for a {subject} lesson.
            Focus on accommodations that:
            1. Are specifically mentioned for {subject}
            2. Are generally applicable to {subject} activities
            3. Would help overcome barriers mentioned in the IEP for similar subjects
            
            IEP content:
            {iep_content}
            
            Extract just the accommodations and list each one separately.
            """
            
            # Generate analysis using LLM
            messages = [
                {"role": "system", "content": "You are an expert in analyzing IEPs and identifying appropriate accommodations for different subjects."},
                {"role": "user", "content": analysis_prompt}
            ]
            
            # Call LLM
            response = self.llm_client.chat_completion(messages)
            
            if not response or "content" not in response:
                logger.error("Failed to analyze IEP for accommodations.")
                return []
            
            # Process the response into a list of accommodations
            # This is a simple implementation; in reality, would need more robust parsing
            accommodations = [
                line.strip('- ').strip() 
                for line in response["content"].split('\n') 
                if line.strip() and not line.strip().startswith('#')
            ]
            
            logger.debug(f"Extracted {len(accommodations)} accommodations from IEP for {subject}")
            return accommodations
            
        except Exception as e:
            logger.error(f"Error analyzing IEP for accommodations: {str(e)}", exc_info=True)
            return []
