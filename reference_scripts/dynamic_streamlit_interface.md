```python
# app/main.py

import streamlit as st
from pathlib import Path
import yaml
from typing import Dict, Any
import json
from src.crews.base import CrewPipeline
import tempfile
import os

class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_dir: Path = Path("config")):
        self.config_dir = config_dir
        self.configs = self._load_all_configs()
        
    def _load_all_configs(self) -> Dict[str, Any]:
        """Load all configuration files"""
        configs = {}
        for config_file in self.config_dir.glob("**/*.yaml"):
            with open(config_file, 'r') as f:
                key = config_file.stem
                configs[key] = yaml.safe_load(f)
        return configs
    
    def get_speaker_options(self) -> Dict[str, Dict]:
        """Get available speaker configurations"""
        return self.configs.get("speakers", {}).get("speakers", {})
    
    def get_style_options(self) -> Dict[str, Dict]:
        """Get available style configurations"""
        return self.configs.get("styles", {}).get("styles", {})

class PodcastApp:
    """Main Streamlit application"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.crew_pipeline = CrewPipeline()
        self.setup_page()
        
    def setup_page(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="PDF to Podcast Generator",
            page_icon="🎙️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("PDF to Podcast Generator")
        
    def render_file_upload(self):
        """Render file upload section"""
        st.header("Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            help="Upload one or more documents to convert into a podcast"
        )
        
        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} files")
            return uploaded_files
        return None

    def render_speaker_selection(self):
        """Render speaker selection section"""
        st.header("Speaker Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Host")
            speakers = self.config_manager.get_speaker_options()
            host_speaker = st.selectbox(
                "Select host voice",
                options=list(speakers.keys()),
                key="host_speaker"
            )
            
            if host_speaker:
                speaker_config = speakers[host_speaker]
                selected_style = st.selectbox(
                    "Select host style",
                    options=list(speaker_config["voice_profiles"].keys()),
                    key="host_style"
                )
                
                # Show voice parameters if available
                if selected_style:
                    voice_params = speaker_config["voice_profiles"][selected_style]["voice_parameters"]
                    st.subheader("Voice Parameters")
                    modified_params = {}
                    for param, value in voice_params.items():
                        if isinstance(value, (int, float)):
                            modified_params[param] = st.slider(
                                f"Host {param}",
                                min_value=0.0,
                                max_value=2.0,
                                value=float(value),
                                key=f"host_{param}"
                            )
        
        with col2:
            st.subheader("Guest")
            guest_speaker = st.selectbox(
                "Select guest voice",
                options=list(speakers.keys()),
                key="guest_speaker"
            )
            
            if guest_speaker:
                speaker_config = speakers[guest_speaker]
                selected_style = st.selectbox(
                    "Select guest style",
                    options=list(speaker_config["voice_profiles"].keys()),
                    key="guest_style"
                )
                
                # Show voice parameters if available
                if selected_style:
                    voice_params = speaker_config["voice_profiles"][selected_style]["voice_parameters"]
                    st.subheader("Voice Parameters")
                    modified_params = {}
                    for param, value in voice_params.items():
                        if isinstance(value, (int, float)):
                            modified_params[param] = st.slider(
                                f"Guest {param}",
                                min_value=0.0,
                                max_value=2.0,
                                value=float(value),
                                key=f"guest_{param}"
                            )

    def render_content_configuration(self):
        """Render content configuration section"""
        st.header("Content Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Content Focus")
            focus_areas = st.multiselect(
                "Select focus areas",
                options=[
                    "Technical Details",
                    "Practical Applications",
                    "Industry Impact",
                    "Research Findings",
                    "Future Implications"
                ],
                default=["Technical Details", "Practical Applications"]
            )
            
            content_depth = st.select_slider(
                "Content Depth",
                options=["Basic", "Intermediate", "Advanced", "Expert"],
                value="Intermediate"
            )
        
        with col2:
            st.subheader("Style Preferences")
            conversation_tone = st.select_slider(
                "Conversation Tone",
                options=["Formal", "Professional", "Casual", "Entertaining"],
                value="Professional"
            )
            
            pacing = st.slider(
                "Conversation Pace",
                min_value=0.8,
                max_value=1.5,
                value=1.0,
                step=0.1
            )

    def render_processing_options(self):
        """Render processing options section"""
        st.header("Processing Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Selection")
            transcript_model = st.selectbox(
                "Transcript Generation Model",
                options=["Llama-3.1-70B", "Llama-3.1-8B"],
                help="Select the model for transcript generation"
            )
            
            tts_model = st.selectbox(
                "Text-to-Speech Model",
                options=["F5-TTS", "GLM-4-Voice"],
                help="Select the text-to-speech model"
            )
        
        with col2:
            st.subheader("Output Options")
            target_duration = st.number_input(
                "Target Duration (minutes)",
                min_value=5,
                max_value=60,
                value=20,
                step=5
            )
            
            audio_quality = st.select_slider(
                "Audio Quality",
                options=["Standard", "High", "Premium"],
                value="High"
            )

    def process_files(self, files):
        """Process uploaded files"""
        results = []
        
        with st.spinner("Processing documents..."):
            for file in files:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file.name.split(".")[-1]}') as tmp_file:
                    tmp_file.write(file.getvalue())
                    file_path = Path(tmp_file.name)
                
                try:
                    # Process the document
                    result = self.crew_pipeline.process_document(file_path)
                    results.append(result)
                finally:
                    # Cleanup
                    os.unlink(file_path)
        
        return results

    def render_results(self, results):
        """Render processing results"""
        st.header("Results")
        
        # Show processing results
        for i, result in enumerate(results):
            with st.expander(f"Document {i+1} Results", expanded=True):
                # Show analysis results
                st.subheader("Content Analysis")
                st.json(result["analysis"])
                
                # Show generated script
                st.subheader("Generated Script")
                st.text_area(
                    "Podcast Script",
                    value=result["content"]["script"],
                    height=300,
                    key=f"script_{i}"
                )
                
                # Show audio preview if available
                if "audio" in result:
                    st.subheader("Audio Preview")
                    st.audio(result["audio"])
                
                # Show quality metrics
                st.subheader("Quality Metrics")
                st.json(result["quality"])

    def run(self):
        """Run the Streamlit application"""
        # Sidebar configuration
        with st.sidebar:
            st.header("Processing Status")
            status_placeholder = st.empty()
            
            if st.button("Clear All"):
                st.session_state.clear()
                st.experimental_rerun()
        
        # Main interface
        uploaded_files = self.render_file_upload()
        
        if uploaded_files:
            self.render_speaker_selection()
            self.render_content_configuration()
            self.render_processing_options()
            
            if st.button("Generate Podcast"):
                results = self.process_files(uploaded_files)
                self.render_results(results)
                status_placeholder.success("Processing complete!")

if __name__ == "__main__":
    app = PodcastApp()
    app.run()
```

This Streamlit app provides:

1. Dynamic Configuration:
- Loads from YAML files
- Configurable options
- Flexible controls

2. Organized Sections:
- File upload
- Speaker configuration
- Content settings
- Processing options

3. User-Friendly Interface:
- Clear sections
- Helpful tooltips
- Progress indicators

4. Processing Pipeline:
- File handling
- Progress tracking
- Result display

5. Result Visualization:
- Analysis results
- Generated script
- Audio preview
- Quality metrics

To run the app:
```bash
streamlit run app/main.py
```

Would you like me to:
1. Add more configuration options?
2. Enhance the result visualization?
3. Add error handling?
4. Create additional controls?