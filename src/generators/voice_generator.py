import logging
from typing import Dict, List, Optional, Tuple

from src.config.settings import Settings
from src.generators.f5tts_client import F5TTSClient
from src.generators.glm4_client import GLM4Client
from src.models.podcast_script import PodcastScript
from src.utils.audio_utils import merge_audio_files

logger = logging.getLogger(__name__)


class VoiceGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.initialize_models()

    def initialize_models(self):
        """Initialize TTS models."""
        try:
            # Initialize F5-TTS client
            self.f5_client = F5TTSClient(self.settings)
            
            # Temporarily comment out GLM4 initialization due to CUDA memory issues
            # self.glm_client = GLM4Client()
            
        except Exception as e:
            logger.error(f"Failed to initialize voice clients: {str(e)}")
            raise RuntimeError(f"Failed to initialize voice clients: {str(e)}")
        
    def generate_voice(
        self,
        script: PodcastScript,
        output_path: str,
        speaker_mapping: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, List[str]]:
        """
        Generate voice for the given script.

        Args:
            script (PodcastScript): The script to generate voice for
            output_path (str): Path to save the final audio file
            speaker_mapping (Optional[Dict[str, str]]): Mapping of speaker names to voice IDs

        Returns:
            Tuple[str, List[str]]: Path to the final audio file and list of intermediate files
        """
        try:
            audio_segments = []
            intermediate_files = []

            # Process each segment
            for segment in script.segments:
                speaker = segment.speaker
                text = segment.text

                # Get voice ID from mapping if provided
                voice_id = speaker_mapping.get(speaker) if speaker_mapping else None

                # Generate audio for segment
                audio_path = self.f5_client.generate_speech(
                    text=text,
                    voice_id=voice_id,
                    output_path=f"{output_path}_{len(audio_segments)}.wav",
                )

                audio_segments.append(audio_path)
                intermediate_files.append(audio_path)

            # Merge all audio segments
            final_path = merge_audio_files(audio_segments, output_path)
            return final_path, intermediate_files

        except Exception as e:
            logger.error(f"Failed to generate voice: {str(e)}")
            raise RuntimeError(f"Failed to generate voice: {str(e)}")

