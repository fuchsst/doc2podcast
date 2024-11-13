"""Shared component for rendering script outputs"""

import streamlit as st

def render_script_output(script_data):
    """Render script generation outputs in the UI"""
    if not script_data:
        return
        
    # Content Strategy
    if "content_strategy" in script_data:
        st.markdown("### Content Strategy")
        strategy = script_data["content_strategy"]
        
        with st.expander("Episode Outline", expanded=True):
            outline = strategy["outline"]
            st.markdown(f"**Introduction**\n{outline['introduction']}")
            
            for i, segment in enumerate(outline['main_segments'], 1):
                st.markdown(f"**{i}. {segment['title']}**\n{segment['description']}")
            
            st.markdown(f"**Conclusion**\n{outline['conclusion']}")
        
        with st.expander("Key Points & Adaptations"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Key Points**")
                for point in strategy["key_points"]:
                    st.markdown(f"- {point}")
            
            with col2:
                st.markdown("**Audience Adaptations**")
                adaptations = strategy["audience_adaptations"]
                for key, value in adaptations.items():
                    st.markdown(f"**{key}:** {value}")
                
    # Script Preview
    if "segments" in script_data:
        st.markdown("### Script Preview")
        segments = script_data["segments"]
        for i, segment in enumerate(segments):
            # Get speaker name safely
            speaker_name = "Speaker"
            if isinstance(segment, dict):
                if isinstance(segment.get('speaker'), dict):
                    speaker_name = segment['speaker'].get('name', 'Speaker')
                elif isinstance(segment.get('speaker'), str):
                    speaker_name = segment['speaker']
            
            with st.expander(f"Segment {i+1}: {speaker_name}", expanded=i==0):
                # Show transitions if available
                if isinstance(segment, dict) and "transitions" in segment:
                    st.markdown("**Transitions**")
                    cols = st.columns(2)
                    if segment["transitions"].get("prev"):
                        cols[0].markdown(f"*From previous:* {segment['transitions']['prev']}")
                    if segment["transitions"].get("next"):
                        cols[1].markdown(f"*To next:* {segment['transitions']['next']}")
                
                # Main content
                st.markdown("**Content**")
                text_content = segment.get('text') if isinstance(segment, dict) else str(segment)
                st.text_area(
                    "Segment Content",  # Added proper label
                    text_content,
                    height=150,
                    key=f"segment_{i}",
                    disabled=True,
                    label_visibility="hidden"  # Hide label since we show "Content" in markdown
                )
                
                # Technical terms if available
                if isinstance(segment, dict) and "technical_terms" in segment and segment["technical_terms"]:
                    st.markdown("**Technical Terms**")
                    for term in segment["technical_terms"]:
                        st.markdown(f"- **{term['term']}:** {term['definition']}")
                
                # Voice parameters if available
                if isinstance(segment, dict) and isinstance(segment.get('speaker'), dict) and "voice_parameters" in segment["speaker"]:
                    st.markdown("**Voice Parameters**")
                    params = segment["speaker"]["voice_parameters"]
                    cols = st.columns(5)
                    
                    cols[0].metric("Pace", f"{params.get('pace', 1.0):.1f}x")
                    cols[1].metric("Pitch", f"{params.get('pitch', 1.0):.1f}")
                    cols[2].metric("Energy", f"{params.get('energy', 0.5):.1f}")
                    cols[3].metric("Variation", f"{params.get('variation', 0.5):.1f}")
                    cols[4].markdown(f"**Emotion:** {params.get('emotion', 'neutral')}")
    
    # Quality Review
    if "quality_metrics" in script_data:
        st.markdown("### Quality Review")
        
        # Metrics visualization
        st.markdown("**Quality Metrics**")
        metrics = script_data["quality_metrics"]
        cols = st.columns(5)
        
        cols[0].metric("Content", f"{metrics['content_accuracy']:.0%}")
        cols[1].metric("Flow", f"{metrics['conversation_flow']:.0%}")
        cols[2].metric("Audience Fit", f"{metrics['audience_fit']:.0%}")
        cols[3].metric("Technical", f"{metrics['technical_accuracy']:.0%}")
        cols[4].metric("Engagement", f"{metrics['engagement']:.0%}")
        
        # Improvements and recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Suggested Improvements**")
            for imp in script_data["improvements"]:
                st.markdown(f"- **{imp['type']}:** {imp['description']}")
        
        with col2:
            st.markdown("**Recommendations**")
            recs = script_data["recommendations"]
            st.markdown("*Content:*")
            for rec in recs["content"]:
                st.markdown(f"- {rec}")
            st.markdown("*Delivery:*")
            for rec in recs["delivery"]:
                st.markdown(f"- {rec}")

def render_document_content(processed_content):
    """Render processed document content in the UI"""
    if not processed_content:
        return
        
    # Document Metadata
    st.markdown("### Document Metadata")
    
    # Title
    if "title" in processed_content:
        st.markdown(f"**Title:** {processed_content['title']}")
    
    # Briefing
    if "briefing" in processed_content:
        st.markdown("### Document Briefing")
        st.markdown(processed_content["briefing"])
    
    # Keywords
    if "keywords" in processed_content:
        st.markdown("### Keywords")
        st.markdown(processed_content["keywords"])
    
    # Topics
    if "topics" in processed_content:
        st.markdown("### Topics")
        topics = processed_content["topics"]
        if isinstance(topics, list):
            for topic in topics:
                if isinstance(topic, dict):
                    st.markdown(f"- **{topic.get('name', '')}:** {topic.get('description', '')}")
                else:
                    st.markdown(f"- {topic}")
        else:
            st.markdown(topics)
    
    # Key Insights
    if "key_insights" in processed_content:
        st.markdown("### Key Insights")
        insights = processed_content["key_insights"]
        if isinstance(insights, dict):
            for category, items in insights.items():
                st.markdown(f"**{category}**")
                if isinstance(items, list):
                    for item in items:
                        st.markdown(f"- {item}")
                else:
                    st.markdown(str(items))
        elif isinstance(insights, list):
            for insight in insights:
                st.markdown(f"- {insight}")
        else:
            st.markdown(str(insights))
    
    # Objectives
    if "objectives" in processed_content:
        st.markdown("### Objectives")
        objectives = processed_content["objectives"]
        if isinstance(objectives, dict):
            for obj_type, items in objectives.items():
                st.markdown(f"**{obj_type}**")
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            st.markdown(f"- **{item.get('objective', '')}:** {item.get('description', '')}")
                        else:
                            st.markdown(f"- {item}")
                else:
                    st.markdown(str(items))
        elif isinstance(objectives, list):
            for obj in objectives:
                st.markdown(f"- {obj}")
        else:
            st.markdown(str(objectives))
    
    # Relationships
    if "relationships" in processed_content:
        st.markdown("### Relationships")
        relationships = processed_content["relationships"]
        if isinstance(relationships, list):
            for rel in relationships:
                if isinstance(rel, dict):
                    st.markdown(f"- **{rel.get('source', '')}** → **{rel.get('target', '')}**: {rel.get('relationship', '')}")
                else:
                    st.markdown(f"- {rel}")
        else:
            st.markdown(str(relationships))
    
    # Hierarchy
    if "hierarchy" in processed_content:
        st.markdown("### Content Hierarchy")
        hierarchy = processed_content["hierarchy"]
        if isinstance(hierarchy, dict):
            for level, items in hierarchy.items():
                st.markdown(f"**{level}**")
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            st.markdown(f"- **{item.get('level', '')}:** {item.get('description', '')}")
                        else:
                            st.markdown(f"- {item}")
                else:
                    st.markdown(str(items))
        else:
            st.markdown(str(hierarchy))
    
    # Additional Metadata
    if "metadata" in processed_content:
        st.markdown("### Additional Metadata")
        metadata = processed_content["metadata"]
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                st.markdown(f"**{key}:** {value}")
        else:
            st.markdown(str(metadata))
