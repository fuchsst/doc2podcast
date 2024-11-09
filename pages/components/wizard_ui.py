"""UI components for wizard-style navigation"""
import streamlit as st
from typing import List, Optional, Dict, Any

def show_progress_bar(custom_steps: Optional[List[str]] = None):
    """Display progress bar with current step"""
    steps = custom_steps if custom_steps else [
        "Document Upload",
        "Script Generation", 
        "TTS Optimization",
        "Audio Generation"
    ]
    
    # Ensure wizard_step exists in session state
    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 1
        
    current_step = float(st.session_state.wizard_step - 1)
    
    # Calculate progress percentage
    progress = (current_step / len(steps))
    
    # Display progress bar
    st.progress(progress)
    
    # Display steps horizontally using columns
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if i < current_step:
                st.success(step, icon="✅")
            elif i == current_step:
                st.info(step, icon="🔄")
            else:
                st.text(f"⏳ {step}")

def can_proceed_to_next_step() -> bool:
    """Check if the current step is complete and we can proceed"""
    current_step = st.session_state.wizard_step
    
    if current_step == 1:
        # Can proceed if document is processed
        return st.session_state.get("processed_content") is not None
    elif current_step == 2:
        # Can proceed if script is generated
        return st.session_state.get("current_script") is not None
    elif current_step == 3:
        # Can proceed if voice settings are configured
        return st.session_state.get("voice_settings") is not None
    else:
        return False

def next_step():
    """Increment wizard step if conditions are met"""
    if can_proceed_to_next_step():
        st.session_state.wizard_step += 1
        st.rerun()

def previous_step():
    """Decrement wizard step"""
    if st.session_state.wizard_step > 1:
        st.session_state.wizard_step -= 1
        st.rerun()

def navigation_buttons():
    """Render navigation buttons"""
    cols = st.columns([1, 1, 1])
    
    with cols[0]:
        if st.session_state.wizard_step > 1:
            if st.button("← Previous Step"):
                previous_step()
                
    with cols[2]:
        if st.session_state.wizard_step < 4:  # 4 is the total number of steps
            # Disable next button if step is not complete
            disabled = not can_proceed_to_next_step()
            if st.button("Next Step →", disabled=disabled):
                next_step()

def show_settings_preview(title: str, settings: Dict[str, Any]):
    """Display settings preview in expandable section"""
    with st.expander(f"{title} Preview"):
        st.json(settings)

def show_error(message: str, details: Optional[str] = None):
    """Display error message with optional details"""
    st.error(message)
    if details:
        with st.expander("Error Details"):
            st.code(details)

def render_script_preview(script: Dict[str, Any]):
    """Render script preview with segments
    
    Args:
        script (Dict[str, Any]): Script data containing:
            - metadata (Dict[str, Any]): Script metadata
            - segments (List[Dict[str, Any]]): Script segments
            - settings (Optional[Dict[str, Any]]): Script settings
    """
    # Show metadata
    if "metadata" in script:
        metadata = script["metadata"]
        st.markdown(f"### {metadata.get('title', 'Untitled Podcast')}")
        
        if metadata.get("description"):
            st.markdown(metadata["description"])
            
        if metadata.get("tags"):
            st.markdown("**Tags:** " + ", ".join(metadata["tags"]))
    
    # Show segments
    if "segments" in script:
        for i, segment in enumerate(script["segments"]):
            with st.expander(
                f"Segment {i+1}: {segment['speaker'].name}",
                expanded=i==0
            ):
                # Show segment text
                st.text_area(
                    "Content",
                    segment["text"],
                    height=150,
                    key=f"segment_{i}",
                    disabled=True
                )
                
                # Show voice parameters
                if hasattr(segment["speaker"], "voice_parameters"):
                    params = segment["speaker"].voice_parameters
                    cols = st.columns(5)
                    
                    cols[0].metric("Pace", f"{params.pace:.1f}x")
                    cols[1].metric("Pitch", f"{params.pitch:.1f}")
                    cols[2].metric("Energy", f"{params.energy:.1f}")
                    cols[3].metric("Variation", f"{params.variation:.1f}")
                    cols[4].markdown(f"**Emotion:** {params.emotion}")
                    
                # Show voice settings
                if hasattr(segment["speaker"], "voice_model"):
                    st.markdown(f"""
                    **Voice Settings:**
                    - Model: {segment["speaker"].voice_model}
                    - Preset: {segment["speaker"].voice_preset or 'default'}
                    - Style Tags: {', '.join(segment["speaker"].style_tags)}
                    """)

def render_voice_settings(
    speaker: str,
    voice_categories: Dict[str, Any],
    voice_profiles: Dict[str, Any]
):
    """Render voice settings configuration
    
    Args:
        speaker (str): Speaker name
        voice_categories (Dict[str, Any]): Available voice categories
        voice_profiles (Dict[str, Any]): Voice profiles by category
    """
    # Voice category selection
    category = st.selectbox(
        "Voice Category",
        options=list(voice_categories.keys()),
        help="Choose the category of voices"
    )
    
    if category:
        col1, col2 = st.columns(2)
        
        with col1:
            # Voice selection
            voices = voice_categories[category]
            voice = st.selectbox(
                "Voice",
                options=list(voices.keys()),
                help="Choose a specific voice"
            )
            
            if voice:
                # Show voice profile
                profile = voice_profiles.get(voice, {})
                show_settings_preview("Voice Profile", profile)
                
        with col2:
            # Voice parameters
            st.markdown("### Voice Parameters")
            
            pace = st.slider(
                "Speaking Pace",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1
            )
            
            pitch = st.slider(
                "Pitch",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1
            )
            
            energy = st.slider(
                "Energy Level",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
            
            variation = st.slider(
                "Expression Variation",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
            
            emotion = st.selectbox(
                "Emotional Style",
                options=["neutral", "happy", "sad", "excited", "serious"],
                index=0
            )
            
        return {
            "category": category,
            "voice": voice,
            "parameters": {
                "pace": pace,
                "pitch": pitch,
                "energy": energy,
                "variation": variation,
                "emotion": emotion
            }
        }
    
    return None

def render_content_strategy(strategy: Dict[str, Any]):
    """Render content strategy preview
    
    Args:
        strategy (Dict[str, Any]): Content strategy data
    """
    if not strategy:
        return
        
    st.markdown("### Content Strategy")
    
    # Show episode structure
    if "episode_structure" in strategy:
        with st.expander("Episode Structure", expanded=True):
            structure = strategy["episode_structure"]
            if "introduction" in structure:
                st.markdown("**Introduction**")
                st.markdown(structure["introduction"])
            
            if "segments" in structure:
                st.markdown("**Main Segments**")
                for i, segment in enumerate(structure["segments"], 1):
                    st.markdown(f"{i}. {segment}")
                    
            if "conclusion" in structure:
                st.markdown("**Conclusion**")
                st.markdown(structure["conclusion"])
    
    # Show key points
    if "key_points" in strategy:
        with st.expander("Key Points", expanded=True):
            for point in strategy["key_points"]:
                st.markdown(f"- {point}")
                
    # Show audience adaptation
    if "audience_adaptation" in strategy:
        with st.expander("Audience Adaptation"):
            adaptation = strategy["audience_adaptation"]
            for aspect, details in adaptation.items():
                st.markdown(f"**{aspect}:**")
                st.markdown(details)
                
    # Show technical depth
    if "technical_depth" in strategy:
        with st.expander("Technical Depth"):
            depth = strategy["technical_depth"]
            for category, level in depth.items():
                st.markdown(f"**{category}:** {level}")

def render_quality_review(review: Dict[str, Any]):
    """Render quality review results
    
    Args:
        review (Dict[str, Any]): Quality review data
    """
    if not review:
        return
        
    st.markdown("### Quality Review")
    
    # Show quality metrics
    if "quality_metrics" in review:
        metrics = review["quality_metrics"]
        cols = st.columns(len(metrics))
        
        for col, (metric, score) in zip(cols, metrics.items()):
            col.metric(
                metric.replace("_", " ").title(),
                f"{score:.1f}/10"
            )
    
    # Show improvements
    if "improvements" in review:
        with st.expander("Suggested Improvements", expanded=True):
            for imp in review["improvements"]:
                st.markdown(f"- {imp}")
                
    # Show recommendations
    if "recommendations" in review:
        with st.expander("Recommendations"):
            for category, recs in review["recommendations"].items():
                st.markdown(f"**{category}**")
                for rec in recs:
                    st.markdown(f"- {rec}")
