"""Podcast creation wizard interface with enhanced configuration management"""

import streamlit as st
import traceback
import json
from pathlib import Path
from pages.components import wizard_ui, script_renderer
from src.config import Settings, PromptManager
from src.generators.script_generator import ScriptGenerationConfig


def script_generation_step(pipeline):
    """Step 2: Script Generation with Presets"""
    st.markdown("## Script Generation")

    if not st.session_state.processed_content:
        st.warning("Please upload and process a document first")
        return

    # Show processed document content
    with st.expander("Processed Document Content", expanded=False):
        script_renderer.render_document_content(st.session_state.processed_content)

    prompt_manager = PromptManager(settings=Settings())
    
    try:
        # Get available podcast presets
        presets = prompt_manager.get_podcast_presets()
        if not presets:
            st.error("No podcast presets available")
            return
            
        preset_options = list(presets.keys())
        
        # Create form for inputs
        with st.form("script_generation_form"):
            col1, col2, col3 = st.columns(3)
            
            # Podcast format selection with first entry as default
            with col1:
                format_type = st.selectbox(
                    "Select Podcast Format",
                    options=preset_options,
                    index=0,
                    help="Choose the style of podcast you want to create"
                )
            
            # Get format details
            format_config = presets[format_type]
            
            # Get available audiences for this format
            audiences = prompt_manager.get_target_audiences(format_type)
            audience_options = [a.name for a in audiences]
            
            # Target audience selection with first entry as default
            with col2:
                target_audience = st.selectbox(
                    "Target Audience",
                    options=audience_options,
                    index=0,
                    help="Select the primary audience for this podcast"
                )
            
            # Get available expertise levels for this format
            expertise_levels = prompt_manager.get_expertise_levels(format_type)
            level_options = [l.name for l in expertise_levels]
            
            # Expertise level selection with first entry as default
            with col3:
                expertise_level = st.selectbox(
                    "Expertise Level",
                    options=level_options,
                    index=0,
                    help="Select the technical depth of the content"
                )
            
            # Optional guidance prompt
            guidance_prompt = st.text_area(
                "Additional Guidance (Optional)",
                help="Provide any specific instructions or focus areas for the podcast"
            )
            
            # Submit button
            generate_script = st.form_submit_button("Generate Script")
        
        if generate_script:
            try:
                # Create script generation config
                script_config = ScriptGenerationConfig(
                    podcast_preset=format_type,
                    target_audience=target_audience,
                    expertise_level=expertise_level,
                    guidance_prompt=guidance_prompt if guidance_prompt else None
                )
                
                # Create placeholders for each step
                strategy_placeholder = st.empty()
                script_placeholder = st.empty()
                review_placeholder = st.empty()
                
                with st.spinner("Step 1/3: Generating Content Strategy..."):
                    # Step 1: Generate content strategy
                    strategy = pipeline.generate_content_strategy(
                        st.session_state.processed_content,
                        config=script_config
                    )
                    # Show intermediate output
                    with strategy_placeholder.container():
                        script_renderer.render_script_output({"content_strategy": strategy})
                
                with st.spinner("Step 2/3: Writing Script..."):
                    # Step 2: Write script
                    script = pipeline.write_script(
                        st.session_state.processed_content,
                        strategy,
                        config=script_config
                    )
                    # Show intermediate output
                    with script_placeholder.container():
                        script_renderer.render_script_output(script)  # script already contains segments
                
                with st.spinner("Step 3/3: Reviewing Script Quality..."):
                    # Step 3: Review script quality
                    quality_review = pipeline.review_script_quality(
                        script,
                        config=script_config
                    )
                    # Show final output with quality review
                    with review_placeholder.container():
                        script_renderer.render_script_output(quality_review)  # quality_review contains metrics directly
                
                # Create complete script with all components
                complete_script = {
                    "content_strategy": strategy,
                    **script,  # includes segments and metadata
                    "quality_review": quality_review,
                    "settings": {
                        "format_type": format_type,
                        "format_config": format_config.model_dump(),
                        "target_audience": target_audience,
                        "expertise_level": expertise_level,
                        "guidance_prompt": guidance_prompt
                    }
                }
                
                # Store in session state
                st.session_state.current_script = complete_script
                st.session_state.script_settings = complete_script["settings"]
                
                # Cache the complete script
                cache_dir = Path(".cache")
                cache_dir.mkdir(exist_ok=True)
                cache_file = cache_dir / "complete_script.json"
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(complete_script, f, indent=2, ensure_ascii=False)
                
                st.success("Script generated successfully!")
                
                # Navigate to next step
                st.session_state.wizard_step = 3
                st.rerun()
                
            except Exception as e:
                error_details = traceback.format_exc()
                wizard_ui.show_error(
                    "Failed to generate script",
                    error_details
                )
                    
    except Exception as e:
        error_details = traceback.format_exc()
        wizard_ui.show_error(
            "Failed to load podcast presets",
            error_details
        )
