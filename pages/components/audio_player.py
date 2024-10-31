import streamlit as st
from pathlib import Path
from typing import Optional

def play_audio(
    audio_path: Path,
    title: Optional[str] = None,
    allow_download: bool = True
):
    """Audio player with download option"""
    if not audio_path.exists():
        st.error(f"Audio file not found: {audio_path}")
        return
        
    # Display title if provided
    if title:
        st.subheader(title)
        
    # Audio player
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/mp3")
        
    # Download button
    if allow_download:
        st.download_button(
            label="Download Audio",
            data=audio_bytes,
            file_name=audio_path.name,
            mime="audio/mp3"
        )
