"""Status tracker component for displaying pipeline progress"""
import streamlit as st

def render_status(status, sidebar=True):
    """Render the current processing status
    
    Args:
        status (dict): Status dictionary containing:
            - current_step (str): Name of current step
            - progress (float): Progress percentage (0-100)
            - message (str): Status message
            - error (str): Error message if any
        sidebar (bool): Whether to render in sidebar (default: True)
    """
    container = st.sidebar if sidebar else st
    
    with container:
        st.markdown("### Current Status")
        
        # Show progress bar
        if status.get("progress") is not None:
            st.progress(status["progress"] / 100)
            
        # Show current step
        if status.get("current_step"):
            # Format step name for display
            step_name = status["current_step"].replace("_", " ").title()
            st.markdown(f"**Step:** {step_name}")
            
        # Show status message
        if status.get("message"):
            st.markdown(f"**Status:** {status['message']}")
            
        # Show error if present
        if status.get("error"):
            st.error(f"Error: {status['error']}")

def render_substeps(substeps):
    """Render substeps in the sidebar
    
    Args:
        substeps (list): List of substep dictionaries containing:
            - agent_role (str): Role of the agent
            - task_description (str): Description of the task
    """
    if not substeps:
        return
        
    # Create a container for substeps if it doesn't exist
    if "substeps_container" not in st.session_state:
        st.session_state.substeps_container = st.sidebar.empty()
        
    with st.session_state.substeps_container:
        st.markdown("### Current Substeps")
        
        # Group substeps by agent role
        agent_tasks = {}
        for substep in substeps:
            role = substep.get("agent_role", "Unknown")
            task = substep.get("task_description", "")
            if role not in agent_tasks:
                agent_tasks[role] = []
            agent_tasks[role].append(task)
            
        # Display grouped tasks
        for role, tasks in agent_tasks.items():
            with st.expander(f"**{role}**", expanded=True):
                for task in tasks:
                    st.markdown(f"- {task}")
