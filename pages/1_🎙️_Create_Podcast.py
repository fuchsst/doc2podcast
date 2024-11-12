"""Podcast creation wizard interface with enhanced configuration management"""

import streamlit as st
from pathlib import Path
import traceback
from pages.components import audio_generation_step, document_upload_step, script_generation_step, voice_settings_step, wizard_ui, audio_player, status_tracker, file_uploader
from src.config import Settings, PromptManager
from src.pipeline.config import ConfigurationManager
from src.pipeline.podcast_pipeline import PodcastPipeline
from src.processors.document_processor import DocumentProcessor
from src.generators.script_generator import PodcastScriptGenerator, ScriptGenerationConfig
from src.generators.voice_generator import VoiceGenerator
from src.utils.callback_handler import PipelineCallback, ProgressUpdate, StepType
from src.models.podcast_script import PodcastScript

st.set_page_config(
    page_title="Create Podcast",
    page_icon="🎙️",
    layout="wide"
)

# Initialize session state
def init_session_state():
    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 1
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = None
    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    if "processed_content" not in st.session_state:
        st.session_state.processed_content = None
    if "current_script" not in st.session_state:
        st.session_state.current_script = None
    if "script_settings" not in st.session_state:
        st.session_state.script_settings = None
    if "voice_settings" not in st.session_state:
        st.session_state.voice_settings = None
    if "generated_audio" not in st.session_state:
        st.session_state.generated_audio = None
    if "status_placeholder" not in st.session_state:
        st.session_state.status_placeholder = st.sidebar.empty()

# Cache resource instances
@st.cache_resource
def get_settings():
    """Cache settings instance"""
    return Settings()

@st.cache_resource
def get_callback_handler():
    """Get pipeline callback handler"""
    callback = PipelineCallback()
    callback.subscribe(handle_progress_update)
    return callback

@st.cache_resource
def get_document_processor(config):
    """Cache document processor instance with config"""
    return DocumentProcessor(get_settings(), config=config)

@st.cache_resource
def get_script_generator():
    """Cache script generator instance"""
    return PodcastScriptGenerator(get_settings(), callback=get_callback_handler())

@st.cache_resource
def get_voice_generator():
    """Cache voice generator instance"""
    return VoiceGenerator(get_settings())

@st.cache_resource
def get_pipeline(config):
    """Cache pipeline instance with config"""
    return PodcastPipeline(
        settings=get_settings(),
        document_processor=get_document_processor(config),
        script_generator=get_script_generator(),
        voice_generator=get_voice_generator(),
        callback=get_callback_handler()
    )

@st.cache_resource
def get_config_manager():
    """Cache configuration manager instance"""
    return ConfigurationManager(get_settings())

def get_processing_config(config_manager: ConfigurationManager):
    """Get processing configuration with UI overrides"""
    # Get base config
    config = config_manager.get_processing_config()
    
    # Show configuration options in UI
    with st.expander("Advanced Processing Settings"):
        chunk_col1, chunk_col2 = st.columns(2)

        chunk_size = chunk_col1.number_input(
            "Chunk Size",
            min_value=100,
            max_value=100000,
            value=config.chunk_size,
            help="Size of text chunks for processing"
        )
        
        overlap = chunk_col2.number_input(
            "Chunk Overlap",
            min_value=0,
            max_value=1000,
            value=config.overlap,
            help="Overlap between chunks"
        )
        
        cache_enabled = st.checkbox(
            "Enable Caching",
            value=config.cache_enabled,
            help="Cache intermediate results"
        )
        
        # Analysis Tool Settings
        st.subheader("Analysis Tool Settings")
        analysis_config = config.analysis_config
        content_param_col1, content_param_col2, content_param_col3 = st.columns(3)

        max_features = content_param_col1.number_input(
            "Max Features",
            min_value=10,
            max_value=1000,
            value=analysis_config.max_features,
            help="Maximum number of features for analysis"
        )
        
        num_topics = content_param_col2.number_input(
            "Number of Topics",
            min_value=1,
            max_value=20,
            value=analysis_config.num_topics,
            help="Number of topics to extract"
        )
        
        min_importance = content_param_col3.slider(
            "Minimum Importance Score",
            min_value=0.0,
            max_value=1.0,
            value=analysis_config.min_importance,
            help="Minimum importance score for content"
        )
        
        # Create override dict
        overrides = {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "cache_enabled": cache_enabled,
            "analysis_config": {
                "chunk_size": chunk_size,  # Pass through chunk_size
                "max_features": max_features,
                "num_topics": num_topics,
                "min_importance": min_importance
            }
        }
        
        return config_manager.get_processing_config(overrides)

def handle_progress_update(update: ProgressUpdate):
    """Handle progress updates from the pipeline"""
    try:
        # Store status for state management
        st.session_state.processing_status = {
            "current_step": update.step.value,
            "progress": update.progress,
            "message": update.message,
            "substeps": update.substeps,
            "error": update.error
        }
        
    except Exception as e:
        st.error(f"Error updating progress: {str(e)}")

def main():
    """Main render function for the podcast creation page"""
    st.title("Create Your Podcast")
    init_session_state()
    
    # Initialize configuration
    processing_config=get_config_manager().get_processing_config()
    pipeline = get_pipeline(processing_config)
    
    # Show wizard progress
    wizard_ui.show_progress_bar([
        "Document Upload",
        "Script Generation", 
        "Voice Settings",
        "Audio Generation"
    ])
    
    # Show navigation buttons
    wizard_ui.navigation_buttons()
    
    # Show current step
    if st.session_state.wizard_step == 1:
        document_upload_step.document_upload_step(processing_config, pipeline)
    elif st.session_state.wizard_step == 2:
        script_generation_step.script_generation_step(pipeline)
    elif st.session_state.wizard_step == 3:
        voice_settings_step.voice_settings_step(pipeline)
    elif st.session_state.wizard_step == 4:
        audio_generation_step.audio_generation_step(pipeline)
    
    # Show status tracker in sidebar
    if st.session_state.processing_status:
        # Show current status in placeholder
        st.session_state.status_placeholder.empty()  # Clear previous content
        with st.session_state.status_placeholder:
            status_tracker.render_status(st.session_state.processing_status, sidebar=True)
        
        # Show substeps directly (not in placeholder)
        if st.session_state.processing_status.get("substeps"):
            with st.sidebar:
                status_tracker.render_substeps(st.session_state.processing_status["substeps"])

if __name__ == "__main__":
    main()
