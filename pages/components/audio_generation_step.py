"""Podcast creation wizard interface with enhanced configuration management"""

import streamlit as st
from pathlib import Path
import traceback
from pages.components import script_generation_step, voice_settings_step, wizard_ui, audio_player, status_tracker, file_uploader
from src.config import Settings, PromptManager
from src.pipeline.config import ConfigurationManager
from src.pipeline.podcast_pipeline import PodcastPipeline
from src.processors.document_processor import DocumentProcessor
from src.generators.script_generator import PodcastScriptGenerator, ScriptGenerationConfig
from src.generators.voice_generator import VoiceGenerator
from src.utils.callback_handler import PipelineCallback, ProgressUpdate, StepType
from src.models.podcast_script import PodcastScript


def audio_generation_step(pipeline):
    """Step 4: Final Audio Generation"""
    st.header("Audio Generation")
    
    if not st.session_state.voice_settings:
        st.warning("Please configure voice settings first")
        return
    
    try:
        # Show model preview
        wizard_ui.show_settings_preview(
            "Voice Synthesis Model",
            Settings().voice_synthesis_config.primary.model_dump()
        )
        
        if st.button("Generate Audio"):
            # Reset substeps
            st.session_state.current_substeps = []
            
            # Create progress placeholders
            st.session_state.progress_placeholder = st.progress(0)
            st.session_state.status_placeholder = st.empty()
            
            # Generate audio
            output_name = Path(st.session_state.current_file).stem
            success = pipeline.generate_audio(
                script=st.session_state.current_script,
                output_name=output_name
            )
            
            if success:
                st.success("Audio generated successfully!")
                
                # Show audio player if available
                audio_path = Path(Settings().project_config.output.audio_dir) / f"{output_name}.mp3"
                if audio_path.exists():
                    audio_player.play_audio(
                        audio_path,
                        title="Generated Podcast"
                    )
                
    except Exception as e:
        wizard_ui.show_error(
            "Failed to generate audio",
            traceback.format_exc()
        )
