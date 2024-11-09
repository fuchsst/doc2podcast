"""Main application module"""
from pathlib import Path
from typing import Optional, Dict, Any

from .config.settings import Settings
from .processors.document_processor import DocumentProcessor
from .generators.script_generator import ScriptGenerator
from .generators.voice_generator import VoiceGenerator
from .pipeline.podcast_pipeline import PodcastPipeline
from .utils.callback_handler import PipelineCallback

class DocToPodcast:
    """Main class for converting documents to podcast format.
    
    This class handles the end-to-end process of converting a document into
    a podcast, including document processing, script generation, and audio
    generation.
    
    Attributes:
        output_dir: Directory where generated podcasts will be saved
        logger: Logger instance for tracking operations
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        callback: Optional[PipelineCallback] = None
    ):
        """Initialize application"""
        self.settings = settings or Settings()
        self.callback = callback
        
        # Initialize pipeline
        self.pipeline = PodcastPipeline.from_settings(
            settings=self.settings,
            callback=self.callback
        )
        
    def process_document(
        self,
        file_path: str,
        output_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Process document through pipeline
        
        Args:
            file_path: Path to input document
            output_name: Optional name for output files
            config: Optional processing configuration
            
        Returns:
            bool: True if processing successful
        """
        try:
            # Update config if provided
            if config:
                self.pipeline.update_config(config)
                
            # Process through pipeline
            success = self.pipeline.process_file(
                file_path=file_path,
                output_name=output_name
            )
            
            return success
            
        except Exception as e:
            if self.callback:
                self.callback.on_error("pipeline_processing", str(e))
            raise
