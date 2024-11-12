"""Podcast creation wizard interface with enhanced configuration management"""

import streamlit as st
import traceback
from pages.components import wizard_ui, file_uploader, script_renderer


def document_upload_step(processing_config, pipeline):
    """Step 1: Document Upload and Processing"""
    
    # File upload component
    doc_path = file_uploader.render_file_uploader()
    
    if doc_path:
        st.session_state.current_file = str(doc_path)
        
        # Show processing settings preview
        wizard_ui.show_settings_preview(
            "Processing Settings",
            processing_config.__dict__
        )
        
        if st.button("Process Document"):
            try:
                # Reset processed content
                st.session_state.processed_content = None
                
                # Create progress placeholders
                st.session_state.progress_placeholder = st.progress(0)
                st.session_state.status_placeholder = st.empty()
                
                # Process document with config
                processed_content = pipeline.process_document(st.session_state.current_file)
                
                # Store processed content
                st.session_state.processed_content = processed_content
                
                # Show preview using shared renderer
                script_renderer.render_document_content(processed_content)
                
                st.success("Document processed successfully!")
                
                # Navigate to next step
                st.session_state.wizard_step = 2
                st.rerun()
                    
            except Exception as e:
                error_details = f"Error: {str(e)}\n\nStacktrace:\n{traceback.format_exc()}"
                st.session_state.processing_status = {
                    "current_step": "Error",
                    "progress": 0,
                    "error": error_details
                }
                
                wizard_ui.show_error("Error processing document", error_details)
