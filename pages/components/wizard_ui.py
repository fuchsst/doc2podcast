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
