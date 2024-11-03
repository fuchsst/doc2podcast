"""Podcast creation wizard interface with enhanced configuration management"""

import streamlit as st
from pathlib import Path
import traceback
from pages.components import wizard_ui, audio_player, status_tracker, file_uploader
from src.config import Settings, PromptManager
from src.pipeline.config import ConfigurationManager
from src.pipeline.podcast_pipeline import PodcastPipeline
from src.processors.document_processor import DocumentProcessor
from src.generators.script_generator import ScriptGenerator
from src.generators.voice_generator import VoiceGenerator
from src.utils.callback_handler import PipelineCallback, ProgressUpdate, StepType

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
    if "progress_placeholder" not in st.session_state:
        st.session_state.progress_placeholder = None
    if "status_placeholder" not in st.session_state:
        st.session_state.status_placeholder = None
    if "substeps_placeholder" not in st.session_state:
        st.session_state.substeps_placeholder = None

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
    return ScriptGenerator(get_settings(), callback=get_callback_handler())

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
    if st.session_state.progress_placeholder:
        # Update progress bar
        st.session_state.progress_placeholder.progress(update.progress / 100)
        
        # Update status message
        st.session_state.status_placeholder.write(f"**{update.step.value}**: {update.message}")
        
        # Update processing status for sidebar display
        st.session_state.processing_status = {
            "current_step": update.step.value,
            "progress": update.progress,
            "message": update.message,
            "error": update.error
        }
        
        # Show substeps if available
        if update.substeps:
            status_tracker.render_substeps(update.substeps)
                
        # Show error if present
        if update.error:
            st.error(f"Error: {update.error}")

def document_upload_step():
    """Step 1: Document Upload and Processing"""
    
    # Get configuration
    config_manager = get_config_manager()
    processing_config = get_processing_config(config_manager)
    
    # File upload component
    doc_path = file_uploader.render_file_uploader()
    
    if doc_path:
        st.session_state.current_file = str(doc_path)
        
        # Show processing settings preview
        wizard_ui.show_settings_preview(
            "Processing Settings",
            processing_config.__dict__
        )
        
        if st.button("Process Document"):
            try:
                # Reset processed content
                st.session_state.processed_content = None
                
                # Create progress placeholders
                st.session_state.progress_placeholder = st.progress(0)
                st.session_state.status_placeholder = st.empty()
                
                # Process document with config
                pipeline = get_pipeline(processing_config)
                processed_content = pipeline.process_document(st.session_state.current_file)
                
                # Store processed content
                st.session_state.processed_content = processed_content
                
                # Show preview
                with st.expander("Document Analysis").container(height=200):
                    st.json(processed_content)
                
                st.success("Document processed successfully!")
                    
            except Exception as e:
                error_details = f"Error: {str(e)}\n\nStacktrace:\n{traceback.format_exc()}"
                st.session_state.processing_status = {
                    "current_step": "Error",
                    "progress": 0,
                    "error": error_details
                }
                
                wizard_ui.show_error("Error processing document", error_details)

def render_analysis_results(analysis_results):
    """Render analysis results preview in main content area
    
    Args:
        analysis_results (dict): Analysis results containing:
            - title (str): Document title
            - topics (dict): Extracted topics
            - key_insights (dict): Key insights
            - questions (dict): Analyzed questions
    """
    if not analysis_results:
        return
        
    st.markdown("### Document Analysis Results")
    
    # Show title if available
    if "title" in analysis_results:
        st.markdown(f"**Generated Title:** {analysis_results['title']}")
        
    col1, col2 = st.columns(2)
    
    # Show topic preview
    with col1:
        if "topics" in analysis_results:
            with st.expander("Topics", expanded=True):
                topics = analysis_results["topics"]
                if isinstance(topics, dict):
                    for topic, details in topics.items():
                        st.markdown(f"- {topic}")
                        
        # Show insights preview
        if "key_insights" in analysis_results:
            with st.expander("Key Insights", expanded=True):
                insights = analysis_results["key_insights"]
                if isinstance(insights, dict):
                    for category, items in insights.items():
                        st.markdown(f"**{category}**")
                        if isinstance(items, list):
                            for item in items:
                                st.markdown(f"- {item}")
    
    # Show questions preview
    with col2:
        if "questions" in analysis_results:
            with st.expander("Research Questions", expanded=True):
                questions = analysis_results["questions"]
                if isinstance(questions, dict):
                    for q_type, q_list in questions.items():
                        st.markdown(f"**{q_type}**")
                        if isinstance(q_list, list):
                            for q in q_list:
                                st.markdown(f"- {q}")

def script_generation_step():
    """Step 2: Script Generation with Presets"""

    if not st.session_state.processed_content:
        st.warning("Please upload and process a document first")
        return

    # Show analysis results if available
    if "analysis" in st.session_state.processed_content:
        render_analysis_results(st.session_state.processed_content["analysis"])

    prompt_manager = PromptManager(settings=Settings())
    
    # Get available formats from prompt manager
    format_type = st.selectbox(
        "Select Podcast Format",
        options=["technical_deep_dive"],  # Default format
        help="Choose the style of podcast you want to create"
    )
    
    if format_type:
        try:
            # Get format details through prompt manager
            format_config = prompt_manager.get_interview_prompt(format_type)
            
            # Show format preview
            wizard_ui.show_settings_preview(
                "Format Details",
                format_config
            )
            
            if st.button("Generate Script"):
                # Create progress placeholders
                st.session_state.progress_placeholder = st.progress(0)
                st.session_state.status_placeholder = st.empty()
                
                with st.spinner("Generating script..."):
                    pipeline = get_pipeline()
                    
                    # Generate script with the processed content
                    script = pipeline.generate_script(st.session_state.processed_content)
                    
                    # Store the script for later use
                    st.session_state.current_script = script
                    
                    # Show script preview with editing
                    st.subheader("Generated Script")
                    
                    # Get the script text, handling both string and object cases
                    script_text = ""
                    if isinstance(script, str):
                        script_text = script
                    elif hasattr(script, 'segments'):
                        # If script has segments, combine their text
                        script_text = "\n\n".join(f"{segment.speaker}: {segment.text}" for segment in script.segments)
                    elif hasattr(script, 'text'):
                        script_text = script.text
                    else:
                        script_text = str(script)
                    
                    edited_script = st.text_area(
                        "Review and edit the script if needed:",
                        script_text,
                        height=300
                    )
                    
                    # Store script settings
                    st.session_state.script_settings = {
                        "format_type": format_type,
                        "format_config": format_config,
                        "edited_script": edited_script
                    }
                
        except Exception as e:
            wizard_ui.show_error(
                "Failed to generate script",
                traceback.format_exc()
            )

def voice_settings_step():
    """Step 3: Voice Settings Configuration"""
    
    if not st.session_state.script_settings or not st.session_state.current_script:
        st.warning("Please generate a script first")
        return

    prompt_manager = PromptManager(settings=Settings())
    
    try:
        # Get voice categories from speakers config
        voices = prompt_manager.speakers_config["voices"]
        
        # Use the stored script from script generation step
        script = st.session_state.current_script
        
        for i, segment in enumerate(script.segments):
            with st.expander(f"Settings for {segment.speaker}", expanded=i==0):
                # Voice settings
                col1, col2 = st.columns(2)
                with col1:
                    category = st.selectbox(
                        "Voice Category",
                        options=list(voices.keys()),
                        key=f"category_{i}",
                        help="Choose the category of voices"
                    )
                    
                    if category:
                        available_voices = voices[category]
                        voice_name = st.selectbox(
                            "Voice",
                            options=list(available_voices.keys()),
                            key=f"voice_{i}",
                            help="Choose a specific voice"
                        )
                
                with col2:
                    pace = st.slider(
                        "Speaking Pace",
                        min_value=0.5,
                        max_value=2.0,
                        value=1.0,
                        step=0.1,
                        key=f"pace_{i}"
                    )
                    energy = st.slider(
                        "Energy Level",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.5,
                        step=0.1,
                        key=f"energy_{i}"
                    )
                
                # Show segment text
                st.text_area(
                    "Segment Text",
                    segment.text,
                    height=100,
                    key=f"text_{i}",
                    disabled=True
                )
                
                if voice_name:
                    # Get and show voice profile
                    profile_type = "technical" if category == "professional" else "casual"
                    voice_profile = prompt_manager.get_voice_profile(
                        category,
                        voice_name,
                        profile_type
                    )
                    
                    wizard_ui.show_settings_preview(
                        "Voice Profile",
                        voice_profile
                    )
        
        if st.button("Apply Voice Settings"):
            # Store voice settings
            st.session_state.voice_settings = {
                "segments": [{
                    "speaker": segment.speaker,
                    "voice": st.session_state[f"voice_{i}"],
                    "pace": st.session_state[f"pace_{i}"],
                    "energy": st.session_state[f"energy_{i}"]
                } for i, segment in enumerate(script.segments)]
            }
                    
    except Exception as e:
        wizard_ui.show_error(
            "Failed to load voice configuration",
            traceback.format_exc()
        )

def audio_generation_step():
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
            # Create progress placeholders
            st.session_state.progress_placeholder = st.progress(0)
            st.session_state.status_placeholder = st.empty()
            
            # Generate audio
            pipeline = get_pipeline()
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

def main():
    """Main render function for the podcast creation page"""
    st.title("Create Your Podcast")
    init_session_state()
    
    # Initialize configuration
    config_manager = get_config_manager()
    
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
        document_upload_step()
    elif st.session_state.wizard_step == 2:
        script_generation_step()
    elif st.session_state.wizard_step == 3:
        voice_settings_step()
    elif st.session_state.wizard_step == 4:
        audio_generation_step()
    
    # Show status tracker in sidebar
    if st.session_state.processing_status:
        status_tracker.render_status(st.session_state.processing_status, sidebar=True)

if __name__ == "__main__":
    main()
