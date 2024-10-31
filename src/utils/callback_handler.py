"""Callback handler for tracking progress of pipeline steps"""
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

class StepType(Enum):
    DOCUMENT_PROCESSING = "document_processing"
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
            
    def on_progress(self, update: ProgressUpdate):
        """Send progress update to all subscribers"""
        for subscriber in self._subscribers:
            subscriber(update)
            
    def on_error(self, step: StepType, error: str):
        """Send error update to all subscribers"""
        update = ProgressUpdate(
            step=step,
            progress=0,
            message="Error occurred",
            error=error
        )
        self.on_progress(update)
        
    def on_document_processing(self, progress: float, message: str):
        """Update document processing progress"""
        self.on_progress(ProgressUpdate(
            step=StepType.DOCUMENT_PROCESSING,
            progress=progress,
            message=message
        ))
        
    def on_script_generation(self, progress: float, message: str, substeps: Optional[List[Dict[str, Any]]] = None):
        """Update script generation progress"""
        self.on_progress(ProgressUpdate(
            step=StepType.SCRIPT_GENERATION,
            progress=progress,
            message=message,
            substeps=substeps
        ))
        
    def on_voice_generation(self, progress: float, message: str):
        """Update voice generation progress"""
        self.on_progress(ProgressUpdate(
            step=StepType.VOICE_GENERATION,
            progress=progress,
            message=message
        ))
        
    def on_audio_processing(self, progress: float, message: str):
        """Update audio processing progress"""
        self.on_progress(ProgressUpdate(
            step=StepType.AUDIO_PROCESSING,
            progress=progress,
            message=message
        ))
