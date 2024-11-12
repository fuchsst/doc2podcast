"""Voice settings configuration step"""

import streamlit as st
import traceback
from pages.components import wizard_ui, status_tracker, script_renderer
from src.config import Settings, PromptManager


def voice_settings_step(pipeline):
    """Step 3: Voice Settings Configuration"""
    st.markdown("## Voice Settings")
    
    if not st.session_state.script_settings or not st.session_state.current_script:
        st.warning("Please generate a script first")
        return

    # Show generated script results
    script_renderer.render_script_output(st.session_state.current_script)

    prompt_manager = PromptManager(settings=Settings())
    
    try:
        # Get voice categories from speakers config
        voices = prompt_manager.speakers_config["voices"]
        
        # Use the stored script
        script = st.session_state.current_script
        
        voice_settings = []
        
        for i, segment in enumerate(script["segments"]):
            # Get speaker name safely
            speaker_name = "Speaker"
            if isinstance(segment, dict):
                if isinstance(segment.get('speaker'), dict):
                    speaker_name = segment['speaker'].get('name', 'Speaker')
                elif isinstance(segment.get('speaker'), str):
                    speaker_name = segment['speaker']
            
            with st.expander(f"Settings for {speaker_name}", expanded=i==0):
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
                
                voice_settings.append({
                    "segment_id": i,
                    "category": category,
                    "voice": voice_name if 'voice_name' in locals() else None,
                    "pace": pace,
                    "energy": energy
                })
        
        if st.button("Apply Voice Settings"):
            try:
                # Store voice settings
                st.session_state.voice_settings = voice_settings
                
                # Initialize progress tracking in sidebar
                with st.sidebar:
                    st.markdown("### Voice Settings Progress")
                    status_tracker.render_substeps([
                        {"name": "Voice Optimization", "status": "in_progress"}
                    ])
                
                # Optimize voice settings
                optimized_script = pipeline.optimize_voice_settings(
                    st.session_state.current_script,
                    {"segments": voice_settings}
                )
                
                # Store optimized script
                st.session_state.optimized_script = optimized_script
                
                st.success("Voice settings applied successfully!")
                
                # Navigate to next step
                st.session_state.wizard_step = 4
                st.rerun()
                
            except Exception as e:
                error_details = traceback.format_exc()
                wizard_ui.show_error(
                    "Failed to apply voice settings",
                    error_details
                )
                    
    except Exception as e:
        error_details = traceback.format_exc()
        wizard_ui.show_error(
            "Failed to load voice configuration",
            error_details
        )
