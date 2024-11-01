"""Podcast generation pipeline using CrewAI"""
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from crewai import Agent, Task, Crew, Process, LLM

from src.config.settings import Settings
from src.generators.script_generator import ScriptGenerator
from src.generators.voice_generator import VoiceGenerator
from src.models.podcast_script import PodcastScript
from src.processors.document_processor import DocumentProcessor
from src.pipeline.analysis_agents import AnalysisAgents
from src.utils.audio_utils import combine_audio_segments
from src.utils.logging_utils import setup_logger
from src.utils.cache_manager import cache_manager
from src.utils.callback_handler import PipelineCallback, StepType

logger = logging.getLogger("podcast_pipeline")

class PodcastPipeline:
    """Pipeline for converting documents to podcast episodes using CrewAI."""

    def __init__(
        self,
        settings: Settings,
        document_processor: DocumentProcessor,
        script_generator: ScriptGenerator,
        voice_generator: VoiceGenerator,
        callback: Optional[PipelineCallback] = None
    ):
        """Initialize the podcast pipeline with CrewAI integration."""
        self.settings = settings
        self.document_processor = document_processor
        self.script_generator = script_generator
        self.voice_generator = voice_generator
        self.callback = callback

        # Initialize LLM
        self.llm = LLM(
            model=settings.text_generation_config.default,
            temperature=settings.text_generation_config.temperature,
            max_tokens=settings.text_generation_config.max_new_tokens,
            api_key=settings.ANTHROPIC_API_KEY
        )

        # Initialize analysis agents
        self.analysis_agents = AnalysisAgents(self.llm)

        # Ensure output directories exist
        os.makedirs(settings.project_config.output.script_dir, exist_ok=True)
        os.makedirs(settings.project_config.output.audio_dir, exist_ok=True)
        os.makedirs(settings.project_config.output.segments_dir, exist_ok=True)

        # Initialize CrewAI agents
        self.initialize_agents()

    def initialize_agents(self):
        """Initialize CrewAI agents for the pipeline."""
        # Document Processing Agent
        self.doc_processor_agent = Agent(
            role="Document Processor",
            goal="Process and analyze document content effectively",
            backstory="Expert in document analysis and preprocessing",
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            step_callback=self._on_doc_processing_step
        )

        # Audio Processing Agent
        self.audio_processor_agent = Agent(
            role="Audio Engineer",
            goal="Generate high-quality audio from processed scripts",
            backstory="Expert in voice synthesis and audio processing",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            step_callback=self._on_audio_processing_step
        )

    def _on_doc_processing_step(self, step_output: Dict[str, Any]):
        """Handle document processing step callback"""
        if self.callback:
            # Update substeps for analysis tasks
            substeps = []
            if "agent_role" in step_output:
                substeps.append({
                    "agent_role": step_output["agent_role"],
                    "task_description": step_output.get("task_description", "Processing...")
                })
            
            self.callback.on_document_processing(
                progress=25,
                message=step_output.get("task_description", "Processing document..."),
                substeps=substeps
            )

    def _on_audio_processing_step(self, step_output: Dict[str, Any]):
        """Handle audio processing step callback"""
        if self.callback:
            self.callback.on_audio_processing(
                progress=75,
                message=step_output.get("task_description", "Processing audio...")
            )

    def create_document_task(self, document_path: str) -> Task:
        """Create document processing task"""
        return Task(
            description=f"Process and analyze the document at {document_path}",
            expected_output="Processed document content with analysis",
            agent=self.doc_processor_agent
        )

    def create_audio_task(self, script: PodcastScript, output_path: str) -> Task:
        """Create audio generation task"""
        return Task(
            description=f"Generate audio for script and save to {output_path}",
            expected_output="Path to generated audio file",
            agent=self.audio_processor_agent
        )

    @cache_manager.cache()
    def process_document(self, document_path: str) -> Dict[str, Any]:
        """Process document content with caching and CrewAI."""
        try:
            if self.callback:
                self.callback.on_document_processing(0, "Starting document processing...")

            # Create and execute document processing task
            doc_task = self.create_document_task(document_path)
            crew = Crew(
                agents=[self.doc_processor_agent],
                tasks=[doc_task],
                verbose=True,
                process=Process.sequential
            )

            # Process document with analysis agents
            path = Path(document_path)
            content = self.document_processor.process(
                path,
                analysis_agents=self.analysis_agents,
                callback=self.callback
            )

            if self.callback:
                self.callback.on_document_processing(100, "Document processing complete")

            return content

        except Exception as e:
            logger.error(f"Document processing error: {str(e)}", exc_info=True)
            if self.callback:
                self.callback.on_error(StepType.DOCUMENT_PROCESSING, str(e))
            raise

    def generate_script(self, document_content: Dict[str, Any]) -> PodcastScript:
        """Generate script from processed document content."""
        try:
            logger.info("Generating script")
            if self.callback:
                self.callback.on_script_generation(0, "Starting script generation...")

            # Use analysis results for better script generation
            analysis = document_content.get("analysis", {})
            enhanced_content = {
                **document_content,
                "title": analysis.get("title", document_content.get("title", "")),
                "briefing": analysis.get("briefing", {}),
                "topics": analysis.get("topics", {}),
                "key_insights": analysis.get("key_insights", {}),
                "questions": analysis.get("questions", {})
            }

            script = self.script_generator.generate_script(enhanced_content)

            if self.callback:
                self.callback.on_script_generation(100, "Script generation complete")

            return script

        except Exception as e:
            logger.error(f"Script generation error: {str(e)}", exc_info=True)
            if self.callback:
                self.callback.on_error(StepType.SCRIPT_GENERATION, str(e))
            raise

    def generate_audio(self, script: PodcastScript, output_name: str) -> bool:
        """Generate audio from script."""
        try:
            logger.info("Generating audio segments")
            segments_dir = os.path.join(
                self.settings.project_config.output.segments_dir,
                output_name
            )
            os.makedirs(segments_dir, exist_ok=True)

            if self.callback:
                self.callback.on_voice_generation(0, "Starting voice generation...")

            audio_segments = []
            total_segments = len(script.segments)
            for i, segment in enumerate(script.segments):
                try:
                    # Generate segment with wav format for better quality during processing
                    segment_path = os.path.join(segments_dir, f"segment_{i}.wav")
                    self.voice_generator.generate_speech(
                        text=segment.text,
                        output_path=segment_path,
                        voice_id=segment.speaker.voice_preset
                    )
                    audio_segments.append(segment_path)

                    if self.callback:
                        progress = ((i + 1) / total_segments) * 100
                        self.callback.on_voice_generation(
                            progress,
                            f"Generated voice segment {i + 1} of {total_segments}"
                        )

                except Exception as e:
                    logger.error(f"Error generating segment {i}: {str(e)}")
                    if self.callback:
                        self.callback.on_error(StepType.VOICE_GENERATION, str(e))
                    continue

            # Combine segments
            logger.info("Combining audio segments")
            if self.callback:
                self.callback.on_audio_processing(0, "Starting audio processing...")

            output_format = self.settings.project_config.output.audio.format.lower()
            output_path = os.path.join(
                self.settings.project_config.output.audio_dir,
                f"{output_name}.{output_format}"
            )

            # Create and execute audio processing task
            audio_task = self.create_audio_task(script, output_path)
            crew = Crew(
                agents=[self.audio_processor_agent],
                tasks=[audio_task],
                verbose=True,
                process=Process.sequential
            )

            # Convert paths to Path objects and use configured format and parameters
            segment_paths = [Path(p) for p in audio_segments]
            combine_audio_segments(
                segment_paths,
                Path(output_path),
                format=output_format,
                bitrate=self.settings.project_config.output.audio.bitrate,
                parameters=self.settings.project_config.output.audio.parameters
            )

            if self.callback:
                self.callback.on_audio_processing(100, "Audio processing complete")

            return True

        except Exception as e:
            logger.error(f"Audio generation error: {str(e)}", exc_info=True)
            if self.callback:
                self.callback.on_error(StepType.AUDIO_PROCESSING, str(e))
            return False
