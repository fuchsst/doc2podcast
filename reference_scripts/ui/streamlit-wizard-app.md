```python
# streamlit_app/app.py
import streamlit as st
import json
from pathlib import Path
from doc2podcast.pipeline import PodcastPipeline
from doc2podcast.config import Settings

def init_session_state():
    """Initialize session state variables"""
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'processed_text' not in st.session_state:
        st.session_state.processed_text = None
    if 'generated_script' not in st.session_state:
        st.session_state.generated_script = None
    if 'tts_ready_script' not in st.session_state:
        st.session_state.tts_ready_script = None
    if 'final_audio' not in st.session_state:
        st.session_state.final_audio = None

def show_progress_bar():
    """Display progress bar with current step"""
    steps = ["Document Upload", "Script Generation", "TTS Optimization", "Audio Generation"]
    current_step = st.session_state.step - 1
    
    progress_html = f"""
        <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
            {"".join([
                f'<div style="flex: 1; text-align: center; '
                f'color: {"#00CC00" if i <= current_step else "#888888"};">'
                f'Step {i+1}: {step}</div>'
                for i, step in enumerate(steps)
            ])}
        </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)

def document_upload_step():
    """Step 1: Document Upload and Preprocessing"""
    st.header("Document Upload and Preprocessing")
    
    uploaded_file = st.file_uploader("Upload your document", type=["pdf", "txt", "docx"])
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            # Process document
            pipeline = PodcastPipeline()
            processed_text = pipeline.process_document(uploaded_file)
            
            # Show preview with option to edit
            st.subheader("Processed Text Preview")
            edited_text = st.text_area(
                "Review and edit the processed text if needed:",
                processed_text,
                height=300
            )
            
            if st.button("Proceed to Script Generation"):
                st.session_state.processed_text = edited_text
                st.session_state.step = 2
                st.experimental_rerun()

def script_generation_step():
    """Step 2: Script Generation with Controls"""
    st.header("Script Generation")
    
    with st.expander("Script Generation Controls", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            topic_focus = st.selectbox(
                "Topic Focus",
                ["Technical", "Casual", "Educational", "Entertainment"]
            )
            target_audience = st.selectbox(
                "Target Audience",
                ["General", "Technical", "Beginner", "Expert"]
            )
        
        with col2:
            host_style = st.selectbox(
                "Host Style",
                ["Professional", "Friendly", "Academic", "Energetic"]
            )
            guest_style = st.selectbox(
                "Guest Style",
                ["Curious", "Expert", "Skeptical", "Enthusiastic"]
            )
    
    if st.button("Generate Script"):
        with st.spinner("Generating podcast script..."):
            # Generate script using Claude
            pipeline = PodcastPipeline()
            script = pipeline.generate_script(
                text=st.session_state.processed_text,
                style_config={
                    "topic_focus": topic_focus,
                    "target_audience": target_audience,
                    "host_style": host_style,
                    "guest_style": guest_style
                }
            )
            
            st.session_state.generated_script = script
    
    if st.session_state.generated_script:
        st.subheader("Generated Script")
        # Rich text editor for script editing
        edited_script = st.text_area(
            "Edit the generated script:",
            st.session_state.generated_script,
            height=400
        )
        
        if st.button("Proceed to TTS Optimization"):
            st.session_state.generated_script = edited_script
            st.session_state.step = 3
            st.experimental_rerun()

def tts_optimization_step():
    """Step 3: TTS Optimization"""
    st.header("TTS Optimization")
    
    # Parse script into speaker segments
    script_segments = json.loads(st.session_state.generated_script)
    
    st.subheader("Voice and Style Settings")
    
    for i, (speaker, text) in enumerate(script_segments):
        with st.expander(f"Settings for {speaker}", expanded=i==0):
            col1, col2 = st.columns(2)
            
            with col1:
                voice = st.selectbox(
                    "Voice",
                    ["Male 1", "Male 2", "Female 1", "Female 2"],
                    key=f"voice_{i}"
                )
                
                speed = st.slider(
                    "Speed",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    key=f"speed_{i}"
                )
            
            with col2:
                style = st.selectbox(
                    "Speaking Style",
                    ["Professional", "Conversational", "Energetic", "Calm"],
                    key=f"style_{i}"
                )
                
                pitch = st.slider(
                    "Pitch",
                    min_value=-20,
                    max_value=20,
                    value=0,
                    step=1,
                    key=f"pitch_{i}"
                )
    
    if st.button("Preview Optimization"):
        # Apply TTS settings to script
        optimized_script = pipeline.optimize_for_tts(
            script_segments,
            tts_settings={
                # Collect all settings
            }
        )
        st.session_state.tts_ready_script = optimized_script
        
        st.subheader("Optimized Script Preview")
        st.json(optimized_script)
    
    if st.session_state.tts_ready_script and st.button("Proceed to Audio Generation"):
        st.session_state.step = 4
        st.experimental_rerun()

def audio_generation_step():
    """Step 4: Final Audio Generation"""
    st.header("Audio Generation")
    
    if st.button("Generate Audio"):
        with st.spinner("Generating audio..."):
            # Generate audio using F5-TTS
            pipeline = PodcastPipeline()
            audio = pipeline.generate_audio(st.session_state.tts_ready_script)
            st.session_state.final_audio = audio
    
    if st.session_state.final_audio:
        st.subheader("Generated Podcast")
        st.audio(st.session_state.final_audio)
        
        # Download options
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download MP3",
                st.session_state.final_audio,
                file_name="podcast.mp3"
            )
        with col2:
            st.download_button(
                "Download Script",
                st.session_state.tts_ready_script,
                file_name="podcast_script.json"
            )

def main():
    st.set_page_config(
        page_title="Document to Podcast Converter",
        page_icon="🎙️",
        layout="wide"
    )
    
    init_session_state()
    
    st.title("🎙️ Document to Podcast Converter")
    show_progress_bar()
    
    # Show current step
    if st.session_state.step == 1:
        document_upload_step()
    elif st.session_state.step == 2:
        script_generation_step()
    elif st.session_state.step == 3:
        tts_optimization_step()
    elif st.session_state.step == 4:
        audio_generation_step()

if __name__ == "__main__":
    main()
```

Key Features:
1. Clear step-by-step progression
2. Visual progress tracking
3. Extensive controls for each stage
4. Preview capabilities
5. State management between steps
6. Rich text editing
7. Detailed TTS controls
8. Download options

Would you like me to:
1. Add more control options for any step?
2. Enhance the UI with additional styling?
3. Add error handling and validation?
4. Include additional preview capabilities?