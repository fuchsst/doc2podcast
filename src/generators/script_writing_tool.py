"""Tools for podcast script generation"""
import json
from typing import Dict, Any, Optional, List
from pydantic import ConfigDict

from .base import ScriptTool, ScriptContext, ScriptToolArgs
from .schemas import ScriptSchema
from ..utils.cache_manager import cache_manager
from ..utils.callback_handler import PipelineCallback, StepType, ProgressUpdate
from ..utils.text_utils import parse_json_safely

class ScriptWritingToolArgs(ScriptToolArgs):
    """Arguments for script writing tool"""
    pass

class ScriptWritingTool(ScriptTool):
    """Tool for writing podcast scripts"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, name: str, description: str, llm: Any, callback: Optional[PipelineCallback] = None):
        super().__init__(name, description, llm)
        self.args_schema = ScriptWritingToolArgs
        self.callback = callback

    def _generate_metadata(self, context: ScriptContext) -> Dict[str, Any]:
        """Generate script metadata"""
        prompt = f"""Create metadata for a podcast about:
Content: {json.dumps(context.content.get('key_points', []))}
Format: {context.format.get('name')}

You must return a SINGLE valid JSON object with NO additional text. Follow these rules EXACTLY:
1. Use double quotes for all strings
2. No trailing commas
3. NO line breaks or newlines within strings
4. Escape any quotes within strings
5. Keep description under 100 words
6. Include 3-5 relevant tags

Required structure:
{{
    "metadata": {{
        "title": "string",
        "description": "string",
        "tags": ["string", "string", "string"]
    }}
}}

Return ONLY the JSON object, nothing else."""
        response = self.llm.call([{"role": "user", "content": prompt}])
        return parse_json_safely(response)

    def _generate_speakers(self, context: ScriptContext) -> Dict[str, Any]:
        """Generate speaker information"""
        prompt = f"""Define speakers for a podcast in format:
Format: {context.format.get('name')}
Roles: {json.dumps(context.format.get('roles', {}))}

You must return a SINGLE valid JSON object with NO additional text. Follow these rules EXACTLY:
1. Use double quotes for all strings
2. No trailing commas
3. NO line breaks or newlines within strings
4. Escape any quotes within strings
5. Each speaker must have a unique name and role

Required structure:
{{
    "speakers": [
        {{
            "name": "string",
            "role": "string"
        }}
    ]
}}

Return ONLY the JSON object, nothing else."""
        response = self.llm.call([{"role": "user", "content": prompt}])
        return parse_json_safely(response)

    def _generate_segment(self, context: ScriptContext, segment_info: Dict[str, Any], speakers: List[Dict[str, Any]], prev_segment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a single segment"""
        # Include relevant key points for this segment
        relevant_points = []
        if isinstance(context.content, dict):
            # Handle key points from content strategy
            if "key_points" in context.content:
                relevant_points = [
                    point for point in context.content["key_points"]
                    if isinstance(point, str) and any(
                        keyword in segment_info.get('title', '').lower() 
                        for keyword in point.lower().split()
                    )
                ]
            
            # Get audience adaptation if available
            audience_adaptation = (
                context.content.get('audience_adaptations', {}).get(
                    context.audience.get('name', 'General Audience'),
                    {"focus": "", "tone": "professional"}
                )
            )

        prompt = f"""Create a script segment for:
Topic: {segment_info.get('title')}
Description: {segment_info.get('description')}
Key Points: {json.dumps(relevant_points)}
Audience Adaptation: {json.dumps(audience_adaptation)}
Speakers: {json.dumps(speakers)}
Previous Segment: {json.dumps(prev_segment) if prev_segment else "None"}

You must return a SINGLE valid JSON object with NO additional text. Follow these rules EXACTLY:
1. Use double quotes for all strings
2. No trailing commas
3. NO line breaks or newlines within strings - use spaces instead
4. Escape any quotes within strings
5. Keep text under 200 words
6. Speaker must be one of the provided speaker names
7. Include at least one technical term with definition

Required structure:
{{
    "segment": {{
        "speaker": "string",
        "text": "string",
        "style": "string",
        "transitions": {{
            "next": "string",
            "prev": "string"
        }},
        "technical_terms": [
            {{
                "term": "string",
                "definition": "string"
            }}
        ]
    }}
}}

Return ONLY the JSON object, nothing else."""
        response = self.llm.call([{"role": "user", "content": prompt}])
        return parse_json_safely(response)
    
    def analyze(self, context: ScriptContext) -> Dict[str, Any]:
        """Generate podcast script by parts"""
        try:
            # Check cache first
            cache_key = cache_manager.get_content_hash(str(context.content))
            cached_result = cache_manager.load_json_cache(cache_key, "script_writing")
            if cached_result:
                return cached_result

            if self.callback:
                self.callback.on_script_generation(
                    progress=0,
                    message="Writing script...",
                    substeps=[
                        {"name": "Generate Metadata", "status": "in_progress"},
                        {"name": "Generate Speakers", "status": "pending"},
                        {"name": "Generate Segments", "status": "pending"}
                    ]
                )

            # Generate metadata and speakers first
            metadata = self._generate_metadata(context)["metadata"]
            
            if self.callback:
                self.callback.on_script_generation(
                    progress=25,
                    message="Generating speakers...",
                    substeps=[
                        {"name": "Generate Metadata", "status": "completed"},
                        {"name": "Generate Speakers", "status": "in_progress"},
                        {"name": "Generate Segments", "status": "pending"}
                    ]
                )
                
            speakers_data = self._generate_speakers(context)["speakers"]

            # Get segments from content strategy
            segments = []
            prev_segment = None
            
            if self.callback:
                self.callback.on_script_generation(
                    progress=50,
                    message="Generating segments...",
                    substeps=[
                        {"name": "Generate Metadata", "status": "completed"},
                        {"name": "Generate Speakers", "status": "completed"},
                        {"name": "Generate Segments", "status": "in_progress"}
                    ]
                )
            
            if isinstance(context.content, dict) and "outline" in context.content:
                outline = context.content["outline"]
                
                # Generate introduction segment
                intro_info = {
                    "title": "Introduction",
                    "description": outline.get("introduction", "")
                }
                intro_segment = self._generate_segment(context, intro_info, speakers_data)["segment"]
                segments.append(intro_segment)
                prev_segment = intro_segment
                
                # Generate main segments
                for segment_info in outline.get("main_segments", []):
                    segment = self._generate_segment(context, segment_info, speakers_data, prev_segment)["segment"]
                    segments.append(segment)
                    prev_segment = segment
                
                # Generate conclusion segment
                conclusion_info = {
                    "title": "Conclusion",
                    "description": outline.get("conclusion", "")
                }
                conclusion_segment = self._generate_segment(context, conclusion_info, speakers_data, prev_segment)["segment"]
                segments.append(conclusion_segment)

            # Combine all parts
            result = {
                "segments": segments,
                "speakers": speakers_data,
                "metadata": metadata,
                "settings": {
                    "format": context.format.get("name", ""),
                    "style": context.format.get("style", "conversational")
                }
            }

            # Validate against schema
            validated = ScriptSchema(**result)
            
            # Cache the result
            cache_manager.cache_json(cache_key, "script_writing", validated.model_dump())
            
            if self.callback:
                self.callback.on_script_generation(
                    progress=100,
                    message="Script writing complete",
                    substeps=[
                        {"name": "Generate Metadata", "status": "completed"},
                        {"name": "Generate Speakers", "status": "completed"},
                        {"name": "Generate Segments", "status": "completed"}
                    ]
                )
            
            return validated.model_dump()
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate script: {str(e)}")
        
    def enhance_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance results with validation"""
        try:
            return ScriptSchema(**results)
        except Exception as e:
            return {
                "error": f"Failed to validate script: {str(e)}",
                "original_results": results
            }

    def invoke(self, *args, **kwargs) -> Dict[str, Any]:
        """CrewAI compatibility method that maps to run()"""
        # Convert CrewAI's input format to our context format
        if 'input' in kwargs:
            context = {
                'content': kwargs['input'],
                'format': kwargs.get('format', {}),
                'audience': kwargs.get('audience', {}),
                'expertise': kwargs.get('expertise', {}),
                'guidance': kwargs.get('guidance')
            }
            # Pass through any additional kwargs
            return self.run(context, **{k: v for k, v in kwargs.items() if k != 'input'})
        return self.run(*args, **kwargs)
