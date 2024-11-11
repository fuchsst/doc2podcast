"""Tools for podcast script generation"""
import json
import re
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ConfigDict

from .base import ScriptTool, ScriptContext, ScriptToolArgs
from .schemas import (
    ContentStrategySchema,
    ScriptSchema,
    OptimizedScriptSchema,
    QualityReviewSchema
)
from ..utils.cache_manager import cache_manager
from ..utils.callback_handler import PipelineCallback, StepType, ProgressUpdate

def clean_json_output(output: str) -> str:
    """Clean and validate JSON output string"""
    # Convert to string if needed
    if not isinstance(output, str):
        output = str(output)
    
    # Find the first { and last } to extract the main JSON object
    start = output.find('{')
    end = output.rfind('}') + 1
    if start >= 0 and end > 0:
        output = output[start:end]
    
    # Remove any escaped newlines and extra whitespace
    output = output.replace('\\n', ' ').strip()
    
    # Fix common JSON formatting issues
    output = re.sub(r'(?<!\\)"(\w+)":', r'"\1":', output)  # Fix unquoted keys
    output = re.sub(r'\'', '"', output)  # Replace single quotes with double quotes
    output = re.sub(r',(\s*[}\]])', r'\1', output)  # Remove trailing commas
    
    # Handle truncated strings by attempting to complete them
    if output.count('{') > output.count('}'):
        output += '}' * (output.count('{') - output.count('}'))
    if output.count('[') > output.count(']'):
        output += ']' * (output.count('[') - output.count(']'))
    
    # Ensure proper string value formatting
    output = re.sub(r':\s*([^"{}\[\],\s][^{}\[\],\s]*?)([,}\]])', r': "\1"\2', output)
    
    return output

def parse_json_safely(json_str: str) -> Dict[str, Any]:
    """Safely parse JSON with enhanced error handling"""
    try:
        # First attempt: Parse the cleaned JSON directly
        cleaned_json = clean_json_output(json_str)
        try:
            return json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            # If first attempt fails, try more aggressive cleaning
            fixed_json = re.sub(r':\s*([^"{}\[\],\s][^{}\[\],\s]*)', r': "\1"', cleaned_json)
            fixed_json = re.sub(r',(\s*[}\]])', r'\1', fixed_json)
            
            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                # If still failing, try to extract just the essential structure
                match = re.search(r'\{.*\}', cleaned_json, re.DOTALL)
                if match:
                    minimal_json = match.group(0)
                    try:
                        return json.loads(minimal_json)
                    except:
                        raise ValueError(f"Failed to parse JSON after multiple attempts. Original error: {str(e)}\nOutput was: {json_str[:200]}...")
                raise ValueError(f"Failed to parse JSON: {str(e)}\nOutput was: {json_str[:200]}...")
                
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {str(e)}\nOutput was: {json_str[:200]}...")

class ContentStrategyToolArgs(ScriptToolArgs):
    """Arguments for content strategy tool"""
    pass

class ContentStrategyTool(ScriptTool):
    """Tool for generating podcast content strategy"""
    
    def __init__(self, name: str, description: str, llm: Any, callback: Optional[PipelineCallback] = None):
        super().__init__(name, description, llm)
        self.args_schema = ContentStrategyToolArgs
        self.callback = callback
    
    def _generate_outline(self, context: ScriptContext) -> Dict[str, Any]:
        """Generate episode outline"""
        prompt = f"""Create an outline for a podcast about:
Content: {context.content}
Format: {context.format.get('name')}

Return a valid JSON object with this structure:
{{
    "outline": {{
        "introduction": "Brief introduction text",
        "main_segments": [
            {{
                "title": "Segment title",
                "description": "Brief segment description"
            }}
        ],
        "conclusion": "Brief conclusion text"
    }}
}}

Keep each segment description concise (max 2-3 sentences).
Ensure all text fields are properly quoted strings.
Do not use line breaks or special characters in the text.
"""
        response = self.llm.call([{"role": "user", "content": prompt}])
        return parse_json_safely(response)
        
    def _generate_key_points(self, context: ScriptContext) -> Dict[str, Any]:
        """Generate key points"""
        prompt = f"""Extract key points from:
Content: {context.content}

Return a valid JSON object with this structure:
{{
    "key_points": [
        "Key point 1",
        "Key point 2",
        "Key point 3"
    ]
}}

Limit to 3-5 most important points.
Ensure all points are properly quoted strings.
Do not use line breaks or special characters in the text.
"""
        response = self.llm.call([{"role": "user", "content": prompt}])
        return parse_json_safely(response)
        
    def _generate_transitions(self, outline: Dict[str, Any]) -> Dict[str, Any]:
        """Generate transitions between segments"""
        segments = ["Introduction"] + [s["title"] for s in outline["outline"]["main_segments"]] + ["Conclusion"]
        transitions = []
        
        for i in range(len(segments)-1):
            transitions.append(f"Transitioning from {segments[i]} to {segments[i+1]}")
            
        return {"transitions": transitions}
        
    def _generate_adaptations(self, context: ScriptContext) -> Dict[str, Any]:
        """Generate audience adaptations"""
        prompt = f"""Create audience adaptations for:
Content: {context.content}
Audience: {context.audience.get('name')}

Return a valid JSON object with this structure:
{{
    "audience_adaptations": {{
        "technical_level": "beginner/intermediate/advanced",
        "engagement_style": "formal/conversational/interactive",
        "examples_type": "real-world/theoretical/mixed",
        "vocabulary_adjustments": [
            "technical term -> simpler explanation"
        ]
    }}
}}

Ensure all text fields are properly quoted strings.
Do not use line breaks or special characters in the text.
"""
        response = self.llm.call([{"role": "user", "content": prompt}])
        return parse_json_safely(response)
        
    def analyze(self, context: ScriptContext) -> Dict[str, Any]:
        """Generate content strategy"""
        try:
            # Check cache first
            cache_key = cache_manager.get_content_hash(str(context.content))
            cached_result = cache_manager.load_json_cache(cache_key, "content_strategy")
            if cached_result:
                return cached_result
                
            if self.callback:
                self.callback.on_script_generation(
                    progress=0,
                    message="Generating content strategy...",
                    substeps=[
                        {"name": "Generate Outline", "status": "in_progress"},
                        {"name": "Generate Key Points", "status": "pending"},
                        {"name": "Generate Transitions", "status": "pending"},
                        {"name": "Generate Adaptations", "status": "pending"}
                    ]
                )
                
            # Generate each component separately
            outline = self._generate_outline(context)
            
            if self.callback:
                self.callback.on_script_generation(
                    progress=25,
                    message="Generating key points...",
                    substeps=[
                        {"name": "Generate Outline", "status": "completed"},
                        {"name": "Generate Key Points", "status": "in_progress"},
                        {"name": "Generate Transitions", "status": "pending"},
                        {"name": "Generate Adaptations", "status": "pending"}
                    ]
                )
                
            key_points = self._generate_key_points(context)
            
            if self.callback:
                self.callback.on_script_generation(
                    progress=50,
                    message="Generating transitions...",
                    substeps=[
                        {"name": "Generate Outline", "status": "completed"},
                        {"name": "Generate Key Points", "status": "completed"},
                        {"name": "Generate Transitions", "status": "in_progress"},
                        {"name": "Generate Adaptations", "status": "pending"}
                    ]
                )
                
            transitions = self._generate_transitions(outline)
            
            if self.callback:
                self.callback.on_script_generation(
                    progress=75,
                    message="Generating adaptations...",
                    substeps=[
                        {"name": "Generate Outline", "status": "completed"},
                        {"name": "Generate Key Points", "status": "completed"},
                        {"name": "Generate Transitions", "status": "completed"},
                        {"name": "Generate Adaptations", "status": "in_progress"}
                    ]
                )
                
            adaptations = self._generate_adaptations(context)
            
            # Combine results
            result = {
                "outline": outline["outline"],
                "key_points": key_points["key_points"],
                "transitions": transitions["transitions"],
                "audience_adaptations": adaptations["audience_adaptations"],
                "metadata": {
                    "format": context.format.get("name", ""),
                    "target_audience": context.audience.get("name", ""),
                    "expertise_level": context.expertise.get("name", "")
                }
            }
            
            # Validate against schema
            validated = ContentStrategySchema(**result)
            
            # Cache the result
            cache_manager.cache_json(cache_key, "content_strategy", validated.dict())
            
            if self.callback:
                self.callback.on_script_generation(
                    progress=100,
                    message="Content strategy generated",
                    substeps=[
                        {"name": "Generate Outline", "status": "completed"},
                        {"name": "Generate Key Points", "status": "completed"},
                        {"name": "Generate Transitions", "status": "completed"},
                        {"name": "Generate Adaptations", "status": "completed"}
                    ]
                )
            
            return validated.dict()
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate content strategy: {str(e)}")
        
    def enhance_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance results with validation"""
        try:
            return ContentStrategySchema(**results)
        except Exception as e:
            return {
                "error": f"Failed to validate content strategy: {str(e)}",
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

Return a valid JSON object with this structure:
{{
    "metadata": {{
        "title": string,
        "description": string (max 100 words),
        "tags": [string]
    }}
}}
"""
        response = self.llm.call([{"role": "user", "content": prompt}])
        return parse_json_safely(response)

    def _generate_speakers(self, context: ScriptContext) -> Dict[str, Any]:
        """Generate speaker information"""
        prompt = f"""Define speakers for a podcast in format:
Format: {context.format.get('name')}
Roles: {json.dumps(context.format.get('roles', {}))}

Return a valid JSON object with this structure:
{{
    "speakers": [
        {{
            "name": string,
            "role": string
        }}
    ]
}}
"""
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
    Topic: {segment_info.get('title')}  # Using 'title' from OutlineSegment schema
    Description: {segment_info.get('description')}
    Key Points: {json.dumps(relevant_points)}
    Audience Adaptation: {json.dumps(audience_adaptation)}
    Speakers: {json.dumps(speakers)}
    Previous Segment: {json.dumps(prev_segment) if prev_segment else "None"}

    Return a valid JSON object with this structure:
    {{
        "segment": {{
            "speaker": string (must be one of the speaker names),
            "text": string (max 200 words),
            "style": string,
            "transitions": {{"next": string, "prev": string}},
            "technical_terms": [
                {{
                    "term": string,
                    "definition": string
                }}
            ]
        }}
    }}
    """
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

class VoiceOptimizationToolArgs(ScriptToolArgs):
    """Arguments for voice optimization tool"""
    pass

class VoiceOptimizationTool(ScriptTool):
    """Tool for optimizing scripts for voice synthesis"""
    
    def __init__(self, name: str, description: str, llm: Any, callback: Optional[PipelineCallback] = None):
        super().__init__(name, description, llm)
        self.args_schema = VoiceOptimizationToolArgs
        self.callback = callback
    
    def analyze(self, context: ScriptContext) -> Dict[str, Any]:
        """Optimize script for voice synthesis"""
        try:
            # Check cache first
            cache_key = cache_manager.get_content_hash(str(context.content))
            cached_result = cache_manager.load_json_cache(cache_key, "voice_optimization")
            if cached_result:
                return cached_result
                
            if self.callback:
                self.callback.on_script_generation(
                    progress=0,
                    message="Optimizing script for voice synthesis...",
                    substeps=[
                        {"name": "Voice Optimization", "status": "in_progress"}
                    ]
                )
                
            prompt = f"""Optimize this script for voice synthesis:
Script: {json.dumps(context.content)}
Voice Profiles: {json.dumps(context.format.get('voices', {}))}

Return a valid JSON object matching OptimizedScriptSchema:
{{
    "segments": [
        {{
            "speaker": string,
            "text": string,
            "style": string,
            "transitions": {{"next": string, "prev": string}},
            "technical_terms": [{{"term": string, "definition": string}}]
        }}
    ],
    "voice_guidance": {{
        "pronunciation": {{"word": string}},
        "emphasis": [{{"word": string, "level": number}}],
        "pacing": {{"speaker": number}},
        "emotions": {{"speaker": string}}
    }},
    "timing": {{
        "total_duration": number,
        "segment_durations": {{"segment_id": number}}
    }}
}}
"""
            response = self.llm.call([{"role": "user", "content": prompt}])
            result = parse_json_safely(response)
            
            # Validate against schema
            validated = OptimizedScriptSchema(**result)
            
            # Cache the result
            cache_manager.cache_json(cache_key, "voice_optimization", validated.dict())
            
            if self.callback:
                self.callback.on_script_generation(
                    progress=100,
                    message="Voice optimization complete",
                    substeps=[
                        {"name": "Voice Optimization", "status": "completed"}
                    ]
                )
            
            return validated.model_dump()
            
        except Exception as e:
            raise RuntimeError(f"Failed to optimize script for voice: {str(e)}")
        
    def enhance_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance results with validation"""
        try:
            return OptimizedScriptSchema(**results)
        except Exception as e:
            return {
                "error": f"Failed to validate voice optimization: {str(e)}",
                "original_results": results
            }

class QualityControlTool(ScriptTool):
    """Tool for script quality control"""
    
    def __init__(self, name: str, description: str, llm: Any, callback: Optional[PipelineCallback] = None):
        super().__init__(name, description, llm)
        self.args_schema = ScriptToolArgs
        self.callback = callback
    
    def analyze(self, context: ScriptContext) -> Dict[str, Any]:
        """Review script quality and provide feedback"""
        try:
            # Check cache first
            cache_key = cache_manager.get_content_hash(str(context.content))
            cached_result = cache_manager.load_json_cache(cache_key, "quality_control")
            if cached_result:
                return cached_result

            if self.callback:
                self.callback.on_script_generation(
                    progress=0,
                    message="Performing quality review...",
                    substeps=[
                        {"name": "Quality Review", "status": "in_progress"}
                    ]
                )

            prompt = f"""Review this podcast script and provide quality metrics and recommendations:
Script: {json.dumps(context.content)}
Target Audience: {context.audience.get('name')} (Technical Depth: {context.audience.get('technical_depth')})
Expertise Level: {context.expertise.get('name')} (Complexity: {context.expertise.get('complexity')})

Return a valid JSON object with this structure:
{{
    "quality_metrics": {{
        "content_accuracy": number (0-1),
        "conversation_flow": number (0-1),
        "audience_fit": number (0-1),
        "technical_accuracy": number (0-1),
        "engagement": number (0-1)
    }},
    "improvements": [
        {{
            "type": string (e.g., "content", "structure", "language"),
            "description": string
        }}
    ],
    "recommendations": {{
        "content": [string],
        "delivery": [string]
    }}
}}

IMPORTANT:
- All numeric values must be proper numbers between 0 and 1
- Provide specific, actionable improvements and recommendations
- Do not include the script content in the response
- Focus only on quality metrics and suggestions for improvement
"""
            response = self.llm.call([{"role": "user", "content": prompt}])
            result = parse_json_safely(response)
            
            # Validate against schema
            validated = QualityReviewSchema(**result)
            
            # Cache the result
            cache_manager.cache_json(cache_key, "quality_control", validated.dict())
            
            if self.callback:
                self.callback.on_script_generation(
                    progress=100,
                    message="Quality review complete",
                    substeps=[
                        {"name": "Quality Review", "status": "completed"}
                    ]
                )
            
            return validated.model_dump()
            
        except Exception as e:
            raise RuntimeError(f"Failed to perform quality review: {str(e)}")
            
    def enhance_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance results with validation"""
        try:
            return QualityReviewSchema(**results)
        except Exception as e:
            return {
                "error": f"Failed to validate quality review: {str(e)}",
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
            return self.run(context)
        return self.run(*args, **kwargs)
