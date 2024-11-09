"""Main pipeline for podcast generation"""

from pathlib import Path
from typing import Dict, Any, Optional
from ..config.settings import Settings
from ..processors.document_processor import DocumentProcessor, ProcessingConfig
from ..generators.script_generator import PodcastScriptGenerator
from ..generators.voice_generator import VoiceGenerator
from ..utils.callback_handler import PipelineCallback, StepType
from ..pipeline.analysis_agents import AnalysisAgents, AgentConfig
from ..pipeline.analysis_tools import AnalysisConfig

class PodcastPipeline:
    """Main pipeline for converting documents to podcasts"""
    
    def __init__(
        self,
        settings: Settings,
        document_processor: DocumentProcessor,
        script_generator: PodcastScriptGenerator,
        voice_generator: VoiceGenerator,
        callback: Optional[PipelineCallback] = None
    ):
        self.settings = settings
        self.document_processor = document_processor
        self.script_generator = script_generator
        self.voice_generator = voice_generator
        self.callback = callback
        
        # Initialize configs from settings
        self.processing_config = ProcessingConfig(
            chunk_size=settings.project_config.processing.chunk_size,
            overlap=settings.project_config.processing.overlap,
            cache_enabled=True,
            analysis_config=AnalysisConfig(
                chunk_size=settings.project_config.processing.chunk_size,
                max_features=100,  # Default
                num_topics=5,  # Default
                context_window=2,  # Default
                min_importance=0.3  # Default
            ),
            agent_config=AgentConfig(
                verbose=True,
                allow_delegation=True,
                process="sequential",
                max_iterations=3
            )
        )
        
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process document with enhanced analysis"""
        try:
            # Update processor config
            if hasattr(self.document_processor, 'config'):
                self.document_processor.config = self.processing_config
                
            # Process document
            result = self.document_processor.process(
                Path(file_path),
                callback=self.callback
            )
            
            return result
            
        except Exception as e:
            if self.callback:
                self.callback.on_error(StepType.DOCUMENT_PROCESSING, str(e))
            raise
            
    def generate_script(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Generate podcast script"""
        try:
            # Generate script
            script = self.script_generator.generate_script(content)
            
            return script
            
        except Exception as e:
            if self.callback:
                self.callback.on_error(StepType.SCRIPT_GENERATION, str(e))
            raise
            
    def generate_audio(
        self,
        script: Dict[str, Any],
        output_name: str
    ) -> bool:
        """Generate audio from script"""
        try:
            # Get voice synthesis config
            voice_config = self.settings.voice_synthesis_config
            
            # Generate audio
            success = self.voice_generator.generate_audio(
                script,
                output_name,
                voice_config=voice_config
            )
            
            return success
            
        except Exception as e:
            if self.callback:
                self.callback.on_error(StepType.AUDIO_GENERATION, str(e))
            raise
            
    def process_file(
        self,
        file_path: str,
        output_name: Optional[str] = None
    ) -> bool:
        """Process file through complete pipeline"""
        try:
            # Process document
            content = self.process_document(file_path)
            
            # Generate script
            script = self.generate_script(content)
            
            # Generate audio
            output_name = output_name or Path(file_path).stem
            success = self.generate_audio(script, output_name)
            
            return success
            
        except Exception as e:
            if self.callback:
                self.callback.on_error(StepType.PIPELINE_PROCESSING, str(e))
            raise
            
    def update_config(self, config: ProcessingConfig):
        """Update pipeline configuration"""
        self.processing_config = config
        
        # Update component configs
        if hasattr(self.document_processor, 'config'):
            self.document_processor.config = config
        if hasattr(self.script_generator, 'config'):
            self.script_generator.config = config
            
    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        callback: Optional[PipelineCallback] = None
    ) -> 'PodcastPipeline':
        """Create pipeline instance from settings"""
        # Create components
        document_processor = DocumentProcessor(settings)
        script_generator = PodcastScriptGenerator(settings, callback=callback)
        voice_generator = VoiceGenerator(settings)
        
        return cls(
            settings=settings,
            document_processor=document_processor,
            script_generator=script_generator,
            voice_generator=voice_generator,
            callback=callback
        )
