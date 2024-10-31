```python
# streamlit_app/app.py
import streamlit as st
import yaml
from pathlib import Path
from doc2podcast.config.models import (
    ProjectSettings, 
    SpeakerConfig, 
    VoiceConfig,
    SpeakerStyle
)

class ConfigManager:
    def __init__(self):
        self.config_path = Path("config")
        self._load_configs()
    
    def _load_configs(self):
        """Load and validate all configurations"""
        # Load voice configs
        with open(self.config_path / "voices.yaml") as f:
            voice_data = yaml.safe_load(f)
            self.voices = {
                k: VoiceConfig(**v) 
                for k, v in voice_data["voices"].items()
            }
        
        # Load speaker configs
        with open(self.config_path / "speakers.yaml") as f:
            speaker_data = yaml.safe_load(f)
            self.speakers = {
                k: SpeakerConfig(**v) 
                for k, v in speaker_data["speakers"].items()
            }
        
        # Load project settings
        with open(self.config_path / "project.yaml") as f:
            project_data = yaml.safe_load(f)
            self.project = ProjectSettings(**project_data["project"])

def script_generation_step(config: ConfigManager):
    """Step 2: Script Generation with Config-based Controls"""
    st.header("Script Generation")
    
    with st.expander("Script Generation Controls", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            topic_focus = st.selectbox(
                "Topic Focus",
                options=config.project.topic_focuses
            )
            
            target_audience = st.selectbox(
                "Target Audience",
                options=config.project.target_audiences
            )
        
        with col2:
            content_style = st.selectbox(
                "Content Style",
                options=config.project.content_styles
            )
    
    # Speaker Settings
    st.subheader("Speaker Settings")
    
    # Host Settings
    host_config = config.speakers["host"]
    with st.expander("Host Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            host_voice = st.selectbox(
                "Host Voice",
                options=host_config.available_voices,
                index=host_config.available_voices.index(host_config.default_voice)
            )
        with col2:
            host_style = st.selectbox(
                "Host Style",
                options=[s.value for s in host_config.available_styles],
                index=[s.value for s in host_config.available_styles].index(
                    host_config.default_style.value
                )
            )
    
    # Guest Settings
    guest_config = config.speakers["guest"]
    with st.expander("Guest Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            guest_voice = st.selectbox(
                "Guest Voice",
                options=guest_config.available_voices,
                index=guest_config.available_voices.index(guest_config.default_voice)
            )
        with col2:
            guest_style = st.selectbox(
                "Guest Style",
                options=[s.value for s in guest_config.available_styles],
                index=[s.value for s in guest_config.available_styles].index(
                    guest_config.default_style.value
                )
            )

def tts_optimization_step(config: ConfigManager):
    """Step 3: TTS Optimization with Config-based Controls"""
    st.header("TTS Optimization")
    
    script_segments = json.loads(st.session_state.generated_script)
    
    for i, (speaker_role, text) in enumerate(script_segments):
        speaker_config = config.speakers[speaker_role.lower()]
        voice_config = config.voices[speaker_config.default_voice]
        
        with st.expander(f"Settings for {speaker_role}", expanded=i==0):
            col1, col2 = st.columns(2)
            
            with col1:
                voice = st.selectbox(
                    "Voice",
                    options=speaker_config.available_voices,
                    index=speaker_config.available_voices.index(
                        speaker_config.default_voice
                    ),
                    key=f"voice_{i}"
                )
                
                speed = st.slider(
                    "Speed",
                    min_value=voice_config.speed_range[0],
                    max_value=voice_config.speed_range[1],
                    value=1.0,
                    step=0.1,
                    key=f"speed_{i}"
                )
            
            with col2:
                style = st.selectbox(
                    "Speaking Style",
                    options=[s.value for s in voice_config.supported_styles],
                    index=[s.value for s in voice_config.supported_styles].index(
                        voice_config.default_style.value
                    ),
                    key=f"style_{i}"
                )
                
                pitch = st.slider(
                    "Pitch",
                    min_value=voice_config.pitch_range[0],
                    max_value=voice_config.pitch_range[1],
                    value=0,
                    step=1,
                    key=f"pitch_{i}"
                )

def main():
    st.set_page_config(
        page_title="Document to Podcast Converter",
        page_icon="🎙️",
        layout="wide"
    )
    
    # Initialize configuration
    config = ConfigManager()
    
    # Store config in session state if not already present
    if 'config' not in st.session_state:
        st.session_state.config = config
    
    init_session_state()
    
    st.title("🎙️ Document to Podcast Converter")
    show_progress_bar()
    
    if st.session_state.step == 1:
        document_upload_step()
    elif st.session_state.step == 2:
        script_generation_step(config)
    elif st.session_state.step == 3:
        tts_optimization_step(config)
    elif st.session_state.step == 4:
        audio_generation_step()
```

This implementation:
1. Uses Pydantic models for type-safe configuration
2. Loads settings from YAML files
3. Dynamically populates UI controls based on configuration
4. Maintains default values from config
5. Validates user inputs against allowed values
6. Provides proper range constraints for numerical inputs

Would you like me to:
1. Add more configuration options?
2. Enhance the validation logic?
3. Add configuration hot-reloading?
4. Include configuration export/import features?