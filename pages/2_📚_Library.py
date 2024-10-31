import streamlit as st
from pathlib import Path
import json
import traceback
from pages.components import audio_player
from src.models.podcast_script import PodcastScript

st.set_page_config(
    page_title="Podcast Library",
    page_icon="📚",
    layout="wide"
)

def main():
    """Render library page"""
    st.title("Podcast Library")
    
    # Get output directory from settings
    output_dir = Path("outputs")
    if not output_dir.exists():
        st.info("No podcasts generated yet. Create your first podcast to see it here!")
        return
        
    # List all generated podcasts
    script_files = list(output_dir.glob("*/script.json"))
    
    if not script_files:
        st.info("No podcasts found. Create your first podcast to see it here!")
        return
        
    st.markdown("""
    ### Your Generated Podcasts
    Browse and play your generated podcasts. Each podcast includes the audio file
    and its associated script.
    """)
    
    # Display podcasts
    for script_file in script_files:
        try:
            # Load script
            script = PodcastScript.load(script_file)
            
            # Create expandable section
            with st.expander(script.metadata.title):
                # Display metadata
                if script.metadata.description:
                    st.markdown(f"**Description:** {script.metadata.description}")
                if script.metadata.source_document:
                    st.markdown(f"**Source:** {script.metadata.source_document}")
                if script.metadata.tags:
                    st.markdown(f"**Tags:** {', '.join(script.metadata.tags)}")
                if script.metadata.created_at:
                    st.markdown(f"**Created:** {script.metadata.created_at}")
                
                # Audio player
                audio_path = script_file.parent / f"{script.metadata.title}.mp3"
                if audio_path.exists():
                    audio_player.play_audio(
                        audio_path,
                        title=script.metadata.title
                    )
                    
                    # Script preview
                    with st.expander("View Script"):
                        st.json(script.to_dict())
                    
                    # Download script button
                    st.download_button(
                        "Download Script",
                        json.dumps(script.to_dict(), indent=2),
                        file_name=script_file.name,
                        mime="application/json"
                    )
                else:
                    st.warning("Audio file not found")
                    
        except Exception as e:
            st.error(f"Error loading podcast {script_file.parent.name}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
