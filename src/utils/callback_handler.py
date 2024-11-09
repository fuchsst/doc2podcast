"""Callback handler for tracking progress of pipeline steps"""
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass
from enum import Enum

class StepType(Enum):
    DOCUMENT_PROCESSING = "document_processing"
    CONTENT_ANALYSIS = "content_analysis"
    TOPIC_EXTRACTION = "topic_extraction"
    INSIGHT_EXTRACTION = "insight_extraction"
    QUESTION_ANALYSIS = "question_analysis"
    RESULT_INTEGRATION = "result_integration"
    SCRIPT_GENERATION = "script_generation" 
    VOICE_GENERATION = "voice_generation"
    AUDIO_PROCESSING = "audio_processing"

@dataclass
class ProgressUpdate:
    step: StepType
    progress: float  # 0-100
    message: str
    substeps: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

class PipelineCallback:
    """Handles callbacks for pipeline progress updates"""
    
    def __init__(self):
        self._subscribers = []
        
    def subscribe(self, callback_fn):
        """Add a subscriber to receive progress updates"""
        self._subscribers.append(callback_fn)
        
    def unsubscribe(self, callback_fn):
        """Remove a subscriber"""
        if callback_fn in self._subscribers:
            self._subscribers.remove(callback_fn)
            
    def on_progress(self, step: Union[StepType, ProgressUpdate], progress: Optional[float] = None, message: Optional[str] = None, substeps: Optional[List[Dict[str, Any]]] = None):
        """Send progress update to all subscribers
        
        Can be called either with:
        1. A ProgressUpdate object: callback.on_progress(progress_update)
        2. Individual parameters: callback.on_progress(step, progress, message, substeps)
        """
        if isinstance(step, ProgressUpdate):
            update = step
        else:
            update = ProgressUpdate(
                step=step,
                progress=progress if progress is not None else 0,
                message=message if message is not None else "",
                substeps=substeps
            )
            
        for subscriber in self._subscribers:
            subscriber(update)
            
    def on_error(self, step: str, error: str):
        """Send error update to all subscribers"""
        # Convert string step to enum if needed
        if isinstance(step, str):
            try:
                step = StepType[step.upper()]
            except KeyError:
                step = StepType.DOCUMENT_PROCESSING  # Default to document processing
                
        update = ProgressUpdate(
            step=step,
            progress=0,
            message="Error occurred",
            error=error
        )
        self.on_progress(update)

    def on_step_start(self, step: StepType, message: str):
        """Called when a major pipeline step starts"""
        self.on_progress(ProgressUpdate(
            step=step,
            progress=0,
            message=message
        ))

    def on_step_complete(self, step: StepType, message: str):
        """Called when a major pipeline step completes"""
        self.on_progress(ProgressUpdate(
            step=step,
            progress=100,
            message=message
        ))

    def on_substep_complete(self, step: StepType, message: str):
        """Called when a substep within a major pipeline step completes"""
        self.on_progress(ProgressUpdate(
            step=step,
            progress=50,  # Use intermediate progress for substeps
            message=message
        ))
        
    def on_document_processing(self, progress: float, message: str, substeps: Optional[List[Dict[str, Any]]] = None):
        """Update document processing progress"""
        self.on_progress(ProgressUpdate(
            step=StepType.DOCUMENT_PROCESSING,
            progress=progress,
            message=message,
            substeps=substeps
        ))
        
    def on_content_analysis(self, progress: float, message: str, substeps: Optional[List[Dict[str, Any]]] = None):
        """Update content analysis progress"""
        self.on_progress(ProgressUpdate(
            step=StepType.CONTENT_ANALYSIS,
            progress=progress,
            message=message,
            substeps=substeps
        ))
        
    def on_topic_extraction(self, progress: float, message: str, substeps: Optional[List[Dict[str, Any]]] = None):
        """Update topic extraction progress"""
        self.on_progress(ProgressUpdate(
            step=StepType.TOPIC_EXTRACTION,
            progress=progress,
            message=message,
            substeps=substeps
        ))
        
    def on_insight_extraction(self, progress: float, message: str, substeps: Optional[List[Dict[str, Any]]] = None):
        """Update insight extraction progress"""
        self.on_progress(ProgressUpdate(
            step=StepType.INSIGHT_EXTRACTION,
            progress=progress,
            message=message,
            substeps=substeps
        ))
        
    def on_question_analysis(self, progress: float, message: str, substeps: Optional[List[Dict[str, Any]]] = None):
        """Update question analysis progress"""
        self.on_progress(ProgressUpdate(
            step=StepType.QUESTION_ANALYSIS,
            progress=progress,
            message=message,
            substeps=substeps
        ))
        
    def on_result_integration(self, progress: float, message: str, substeps: Optional[List[Dict[str, Any]]] = None):
        """Update result integration progress"""
        self.on_progress(ProgressUpdate(
            step=StepType.RESULT_INTEGRATION,
            progress=progress,
            message=message,
            substeps=substeps
        ))
        
    def on_script_generation(self, progress: float, message: str, substeps: Optional[List[Dict[str, Any]]] = None):
        """Update script generation progress"""
        self.on_progress(ProgressUpdate(
            step=StepType.SCRIPT_GENERATION,
            progress=progress,
            message=message,
            substeps=substeps
        ))
        
    def on_voice_generation(self, progress: float, message: str, substeps: Optional[List[Dict[str, Any]]] = None):
        """Update voice generation progress"""
        self.on_progress(ProgressUpdate(
            step=StepType.VOICE_GENERATION,
            progress=progress,
            message=message,
            substeps=substeps
        ))
        
    def on_audio_processing(self, progress: float, message: str, substeps: Optional[List[Dict[str, Any]]] = None):
        """Update audio processing progress"""
        self.on_progress(ProgressUpdate(
            step=StepType.AUDIO_PROCESSING,
            progress=progress,
            message=message,
            substeps=substeps
        ))
        
    def on_analysis_step(self, step: StepType, agent_role: str, task_description: str, progress: float):
        """Helper method for analysis step updates"""
        self.on_progress(ProgressUpdate(
            step=step,
            progress=progress,
            message=task_description,
            substeps=[{
                "agent_role": agent_role,
                "task_description": task_description
            }]
        ))
