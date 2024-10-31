from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging
from datetime import datetime

from .config.settings import Settings
from .processors import DocumentProcessor
from .generators.script_generator import  ScriptGenerator
from .generators.voice_generator import  VoiceGenerator
from .models.podcast_script import PodcastScript, PodcastMetadata
from .utils import audio_utils, cache_manager, logging_utils

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
        output_dir: Optional[Path] = None,
        log_file: Optional[Path] = None
    ):
        """Initialize DocToPodcast instance.
        
        Args:
            output_dir: Optional output directory path. Defaults to ./outputs
            log_file: Optional log file path. Defaults to None (console only)
        """
        self.output_dir = Path(output_dir) if output_dir else Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging_utils.setup_logger(
            "doc_to_podcast",
            log_file or self.output_dir / "podcast_generation.log"
        )
        
    def _validate_document_path(self, document_path: Path) -> None:
        """Validate document path exists and is a file.
        
        Args:
            document_path: Path to document
            
        Raises:
            ValueError: If path is invalid
        """
        if not document_path.exists():
            raise ValueError(f"Document path does not exist: {document_path}")
        if not document_path.is_file():
            raise ValueError(f"Document path is not a file: {document_path}")
            
    def process_document(
        self,
        document_path: Path,
        settings_override: Optional[Dict[str, Any]] = None
    ) -> PodcastScript:
        """Process document and generate podcast script.
        
        Args:
            document_path: Path to source document
            settings_override: Optional settings to override defaults
            
        Returns:
            Generated podcast script
            
        Raises:
            ValueError: If document path is invalid
            RuntimeError: If processing fails
        """
        self.logger.info(f"Processing document: {document_path}")
        self._validate_document_path(document_path)
        
        try:
            # Merge settings
            processing_settings = Settings()
            if settings_override:
                processing_settings.update(settings_override)
            
            # Process document
            self.logger.debug("Extracting document content")
            content = DocumentProcessor(processing_settings).process_document(document_path)
            
            # Generate script
            self.logger.debug("Generating script from content")
            script_data = ScriptGenerator(processing_settings).generate_script(
                content,
                processing_settings
            )
            
            # Create podcast script
            script = PodcastScript(
                metadata=PodcastMetadata(
                    title=script_data.get("title", document_path.stem),
                    description=script_data.get("description"),
                    source_document=document_path.name,
                    tags=script_data.get("tags", []),
                    created_at=datetime.now().isoformat()
                ),
                segments=script_data["segments"],
                settings=processing_settings
            )
            
            self.logger.info("Document processing completed successfully")
            return script
            
        except Exception as e:
            self.logger.error(f"Failed to process document: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to process document: {str(e)}") from e
            
    def generate_audio(
        self,
        script: PodcastScript,
        output_dir: Optional[Path] = None
    ) -> PodcastScript:
        """Generate audio for podcast script.
        
        Args:
            script: Podcast script to generate audio for
            output_dir: Optional custom output directory
            
        Returns:
            Updated podcast script with audio information
            
        Raises:
            RuntimeError: If audio generation fails
        """
        # Setup output directory
        output_path = Path(output_dir or self.output_dir / script.metadata.title)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Generating audio in: {output_path}")
        
        try:
            # Generate voice segments
            self.logger.debug("Generating voice segments")
            results = VoiceGenerator().process_script(
                script.to_dict(),
                output_path / "segments"
            )
            
            # Update script with audio information
            for i, segment_info in enumerate(results["segments"]):
                script.update_segment_duration(
                    i,
                    segment_info["duration"],
                    segment_info["path"]
                )
            
            # Combine segments
            segment_paths = [
                Path(segment.audio_path)
                for segment in script.segments
                if segment.audio_path
            ]
            
            if segment_paths:
                self.logger.debug("Combining audio segments")
                final_path = output_path / f"{script.metadata.title}.mp3"
                combined_path = audio_utils.combine_audio_segments(
                    segment_paths,
                    final_path
                )
                
                # Post-process audio
                self.logger.debug("Post-processing audio")
                processed_path = audio_utils.normalize_audio(combined_path)
                processed_path = audio_utils.remove_silence(processed_path)
                
                # Save script data
                script_path = output_path / "script.json"
                self.logger.debug(f"Saving script data to: {script_path}")
                with open(script_path, "w") as f:
                    json.dump(script.to_dict(), f, indent=2)
                    
                self.logger.info("Audio generation completed successfully")
                
            return script
            
        except Exception as e:
            self.logger.error(f"Failed to generate audio: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate audio: {str(e)}") from e
            
    def process(
        self,
        document_path: Path,
        settings_override: Optional[Dict[str, Any]] = None,
        output_dir: Optional[Path] = None
    ) -> PodcastScript:
        """Process document and generate complete podcast.
        
        This is the main entry point that handles the complete pipeline from
        document to podcast.
        
        Args:
            document_path: Path to source document
            settings_override: Optional settings to override defaults
            output_dir: Optional custom output directory
            
        Returns:
            Complete podcast script with audio
        """
        self.logger.info(f"Starting complete podcast generation for: {document_path}")
        
        # Process document
        script = self.process_document(document_path, settings_override)
        
        # Generate audio
        script = self.generate_audio(script, output_dir)
        
        self.logger.info("Podcast generation completed successfully")
        return script
