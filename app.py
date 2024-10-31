import streamlit as st
from pathlib import Path
import sys

# Add src directory to Python path
src_path = str(Path(__file__).parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from src.app import DocToPodcast
from src.config.settings import Settings

def initialize_session_state():
    """Initialize session state variables"""
    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 1
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = None
    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    if "script_settings" not in st.session_state:
        st.session_state.script_settings = None
    if "voice_settings" not in st.session_state:
        st.session_state.voice_settings = None
    if "generated_audio" not in st.session_state:
        st.session_state.generated_audio = None

def main():
    # Configure page
    st.set_page_config(
        page_title="Doc2Podcast",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize settings
    settings = Settings()
    
    # Create DocToPodcast instance for processing
    if "doc_to_podcast" not in st.session_state:
        st.session_state.doc_to_podcast = DocToPodcast()
    
    # Main page content
    st.title("🎙️ Doc2Podcast")
    
    st.markdown("""
    Welcome to Doc2Podcast! Convert your documents into engaging podcasts with AI-powered voice synthesis.
    
    ### Getting Started
    1. Navigate to the **Create Podcast** page to start a new conversion
    2. Check the **Library** to browse your generated podcasts
    
    ### Features
    - Document analysis and preprocessing
    - AI-powered script generation
    - Professional voice synthesis
    - Custom voice profiles and settings
    """)

if __name__ == "__main__":
    main()
