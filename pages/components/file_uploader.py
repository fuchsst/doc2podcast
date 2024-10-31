import streamlit as st
from pathlib import Path
from typing import Optional

def render_file_uploader() -> Optional[Path]:
    """Document upload component with validation"""
    uploaded_file = st.file_uploader(
        "Upload your document",
        type=["pdf"],
        help="Upload a PDF document to convert to podcast"
    )
    
    if uploaded_file:
        # Save uploaded file temporarily
        temp_path = Path("temp") / uploaded_file.name
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
            
        return temp_path
    
    return None
