import streamlit as st
from pathlib import Path
import traceback
from pages.components import wizard_ui, audio_player, status_tracker, file_uploader
from src.config import Settings, PromptManager
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
def get_document_processor():
    """Cache document processor instance"""
    return DocumentProcessor(get_settings())

@st.cache_resource
def get_script_generator():
    """Cache script generator instance"""
    return ScriptGenerator(get_settings(), callback=get_callback_handler())

@st.cache_resource
def get_voice_generator():
    """Cache voice generator instance"""
    return VoiceGenerator(get_settings())

@st.cache_resource
def get_pipeline():
    """Cache pipeline instance"""
    return PodcastPipeline(
        settings=get_settings(),
        document_processor=get_document_processor(),
        script_generator=get_script_generator(),
        voice_generator=get_voice_generator(),
        callback=get_callback_handler()
    )

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
    
    # File upload component
    doc_path = file_uploader.render_file_uploader()
    
    if doc_path:
        st.session_state.current_file = str(doc_path)  # Convert Path to string
        
        # Show processing settings preview
        wizard_ui.show_settings_preview(
            "Processing Settings",
            Settings().project_config.processing.model_dump()
        )
        
        if st.button("Process Document"):
            try:
                # Create progress placeholders
                st.session_state.progress_placeholder = st.progress(0)
                st.session_state.status_placeholder = st.empty()
                
                # Process document only
                pipeline = get_pipeline()
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

def script_generation_step():
    """Step 2: Script Generation with Presets"""

    if not st.session_state.processed_content:
        st.warning("Please upload and process a document first")
        return

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
