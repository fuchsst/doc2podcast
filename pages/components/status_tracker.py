"""Status tracking component for the podcast creation wizard"""
import streamlit as st
from typing import Dict, Any, List, Optional

def render_status(status: Dict[str, Any], sidebar: bool = False):
    """Render status information
    
    Args:
        status (Dict[str, Any]): Status information containing:
            - current_step (str): Current step name
            - progress (float): Progress percentage
            - message (str): Status message
            - error (Optional[str]): Error message if any
        sidebar (bool): Whether to render in sidebar
    """
    container = st.sidebar if sidebar else st
    
    with container:
        # Show current step and progress
        if status.get("current_step"):
            st.markdown(f"### Current Status\n**Step:** {status['current_step']}")
            
        if status.get("progress") is not None:
            st.progress(status["progress"] / 100)
            
        # Show status message
        if status.get("message"):
            st.info(status["message"])
            
        # Show error if present
        if status.get("error"):
            st.error(status["error"])

def render_substeps(substeps: List[Dict[str, Any]], expanded: bool = True):
    """Render substeps progress
    
    Args:
        substeps (List[Dict[str, Any]]): List of substeps containing:
            - name (str): Substep name
            - status (str): Status (e.g., "completed", "in_progress")
            - message (Optional[str]): Additional message
        expanded (bool): Whether to show expanded by default
    """
    st.markdown("### Progress Steps")
    for step in substeps:
        # Show step name with status icon
        status_icon = "✅" if step["status"] == "complete" else "⏳" if step["status"] == "in_progress" else "⏸️"
        st.markdown(f"{status_icon} **{step['name']}**")
        
        # Show message if present
        if step.get("message"):
            st.markdown(f"_{step['message']}_")

def render_script_progress(
    current_segment: int,
    total_segments: int,
    segment_info: Optional[Dict[str, Any]] = None
):
    """Render script generation progress
    
    Args:
        current_segment (int): Current segment being processed
        total_segments (int): Total number of segments
        segment_info (Optional[Dict[str, Any]]): Current segment info containing:
            - speaker (str): Speaker name
            - type (str): Segment type
            - length (int): Segment length
    """
    # Show overall progress
    progress = (current_segment / total_segments) * 100
    st.progress(progress / 100)
    
    # Show segment details
    if segment_info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Current Segment",
                f"{current_segment + 1} / {total_segments}"
            )
            
        with col2:
            if segment_info.get("speaker"):
                st.metric(
                    "Speaker",
                    segment_info["speaker"]
                )
                
        if segment_info.get("type"):
            st.caption(f"Type: {segment_info['type']}")
            
        if segment_info.get("length"):
            st.caption(f"Length: {segment_info['length']} words")

def render_voice_progress(
    current_speaker: str,
    voice_info: Dict[str, Any],
    progress: float
):
    """Render voice generation progress
    
    Args:
        current_speaker (str): Current speaker being processed
        voice_info (Dict[str, Any]): Voice settings containing:
            - model (str): Voice model
            - preset (str): Voice preset
            - parameters (Dict[str, Any]): Voice parameters
        progress (float): Progress percentage
    """
    st.markdown(f"### Processing: {current_speaker}")
    st.progress(progress / 100)
    
    # Show voice settings
    with st.expander("Voice Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Model:** {voice_info.get('model', 'default')}")
            st.markdown(f"**Preset:** {voice_info.get('preset', 'default')}")
            
        with col2:
            params = voice_info.get("parameters", {})
            st.markdown(f"**Pace:** {params.get('pace', 1.0)}")
            st.markdown(f"**Energy:** {params.get('energy', 0.5)}")
            st.markdown(f"**Emotion:** {params.get('emotion', 'neutral')}")
