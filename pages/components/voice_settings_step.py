"""Voice settings configuration step"""

import streamlit as st
import traceback
import yaml
from pathlib import Path
from pages.components import wizard_ui, status_tracker, script_renderer
from src.config import Settings, PromptManager
from src.models.podcast_script import Speaker, VoiceParameters
from src.generators.voice_optimization_tool import VoiceOptimizationTool, VoiceOptimizationContext
from src.generators.schemas import ContentStrategySchema, ScriptSchema, ScriptMetadata, QualityReviewSchema


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
        # Load speakers config
        with open('config/speakers.yaml', 'r') as f:
            speakers_config = yaml.safe_load(f)
        voices = speakers_config.get('voices', {})
        
        # Use the stored script
        script = st.session_state.current_script
        
        voice_settings = []
        
        for i, segment in enumerate(script.segments):
            speaker_name = segment.speaker.name if hasattr(segment, 'speaker') else "Speaker"
            
            with st.expander(f"Settings for {speaker_name}", expanded=i==0):
                # Voice settings
                col1, col2 = st.columns(2)
                with col1:
                    voice_col1, voice_col2, voice_col3 = st.columns(3)
                    category = voice_col1.selectbox(
                        "Voice Category",
                        options=list(voices.keys()),
                        key=f"category_{i}",
                        help="Choose the category of voices"
                    )
                    
                    if category:
                        available_voices = voices[category]
                        voice_name = voice_col2.selectbox(
                            "Voice",
                            options=list(available_voices.keys()),
                            key=f"voice_{i}",
                            help="Choose a specific voice"
                        )
                        
                        if voice_name:
                            voice_profiles = available_voices[voice_name].get('voice_profiles', {})
                            style = voice_col3.selectbox(
                                "Speaking Style",
                                options=list(voice_profiles.keys()),
                                key=f"style_{i}",
                                help="Choose the speaking style"
                            )
                
                with col2:
                    slider_col1, slider_col2, slider_col3 = st.columns(3)
                    pace = slider_col1.slider(
                        "Speaking Pace",
                        min_value=0.5,
                        max_value=2.0,
                        value=1.0,
                        step=0.1,
                        key=f"pace_{i}"
                    )
                    energy = slider_col2.slider(
                        "Energy Level",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.5,
                        step=0.1,
                        key=f"energy_{i}"
                    )
                    variation = slider_col3.slider(
                        "Variation",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.5,
                        step=0.1,
                        key=f"variation_{i}"
                    )
                
                # Get voice profile if selected
                voice_profile = None
                if all([category, voice_name, style]):
                    voice_profile = voices[category][voice_name]['voice_profiles'][style]
                
                voice_settings.append({
                    "segment_id": i,
                    "speaker": {
                        "name": f"{voice_name}" if voice_name else speaker_name,
                        "voice_model": voice_profile['model'] if voice_profile else None,
                        "voice_preset": f"{category}.{voice_name}.{style}" if all([category, voice_name, style]) else None,
                        "style_tags": voice_profile.get('style_tags', []) if voice_profile else []
                    },
                    "voice_parameters": {
                        "pace": pace,
                        "energy": energy,
                        "variation": variation,
                        "pitch": voice_profile['voice_parameters']['pitch'] if voice_profile else 1.0,
                        "emotion": voice_profile['voice_parameters']['emotion'] if voice_profile else "neutral"
                    },
                    "text": segment.text
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
                
                # Initialize voice optimization tool
                voice_tool = VoiceOptimizationTool(
                    name="Voice Optimization Tool",
                    description="Optimizes script for voice synthesis with proper pacing, emphasis, and emotional guidance",
                    llm=Settings().get_llm()
                )
                
                # Create optimization context
                context = VoiceOptimizationContext(
                    content_strategy=ContentStrategySchema(**st.session_state.content_strategy),
                    script=ScriptSchema(**st.session_state.current_script),
                    voice_settings=voice_settings,
                    quality_review=QualityReviewSchema(**st.session_state.quality_review),
                    metadata=ScriptMetadata(**st.session_state.current_script.metadata),
                    settings=st.session_state.script_settings
                )
                
                # Generate optimized script
                optimized_script = voice_tool.analyze(context)
                
                # Generate final structured script
                structured_script = voice_tool.generate_structured_script(optimized_script, context)
                
                # Store both versions
                st.session_state.optimized_script = optimized_script
                st.session_state.structured_script = structured_script
                
                st.success("Voice settings applied successfully!")
                
                # Show preview of structured script
                st.markdown("### Preview of Structured Script")
                for segment in structured_script:
                    st.markdown(f"""
                    **{segment['speaker']} ({segment['style']}):**  
                    {segment['text']}
                    """)
                
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
