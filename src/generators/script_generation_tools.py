"""Tools for podcast script generation"""
import json
import re
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ConfigDict

from .base import ScriptTool, ScriptContext, ScriptToolArgs
from .schemas import (
    ContentStrategySchema,
    ScriptSchema,
    ScriptSegmentSchema,
    OptimizedScriptSchema,
    VoiceGuidanceSchema,
    QualityReviewSchema,
    QualityMetricsSchema
)
from ..utils.cache_manager import cache_manager

def clean_json_output(output: str) -> str:
    """Clean and validate JSON output string"""
    # Convert to string if needed
    if not isinstance(output, str):
        output = str(output)
    
    # Remove any non-JSON content before the first { and after the last }
    start = output.find('{')
    end = output.rfind('}') + 1
    if start >= 0 and end > 0:
        output = output[start:end]
    
    # Remove any escaped newlines and extra whitespace
    output = output.replace('\\n', ' ').strip()
    
    # Fix common JSON formatting issues
    output = re.sub(r'(?<!\\)"(\w+)":', r'"\1":', output)  # Fix unquoted keys
    output = re.sub(r'\'', '"', output)  # Replace single quotes with double quotes
    output = re.sub(r',\s*}', '}', output)  # Remove trailing commas
    output = re.sub(r',\s*]', ']', output)  # Remove trailing commas in arrays
    
    # Fix audience_adaptation and technical_depth formatting
    output = re.sub(
        r'"(\w+)":\s*"([^"]+)"(?=,\s*"(?:\w+)":|})',
        r'"\1": {"focus": "\2", "tone": "professional"}',
        output
    )
    
    # Ensure proper string escaping
    output = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', output)
    
    return output

def parse_json_safely(json_str: str) -> Dict[str, Any]:
    """Safely parse JSON with enhanced error handling"""
    try:
        # Handle potential nested JSON strings
        def fix_nested_json(match):
            try:
                # Try to parse the string as JSON
                nested = json.loads(match.group(1))
                return json.dumps(nested)
            except:
                return match.group(1)

        # Clean the JSON string
        cleaned_json = clean_json_output(json_str)
        
        # Fix nested JSON strings that might be escaped
        cleaned_json = re.sub(r'"({[^}]+})"', fix_nested_json, cleaned_json)
        
        # Handle large responses by finding the outermost complete JSON object
        start_brace = cleaned_json.find('{')
        if start_brace >= 0:
            stack = []
            for i, char in enumerate(cleaned_json[start_brace:], start=start_brace):
                if char == '{':
                    stack.append(i)
                elif char == '}':
                    if stack:
                        start = stack.pop()
                        if not stack:  # Found complete outermost object
                            cleaned_json = cleaned_json[start:i+1]
                            break
        
        # Attempt to parse
        try:
            return json.loads(cleaned_json)
        except json.JSONDecodeError:
            # Try to fix common issues and retry
            fixed_json = re.sub(r':\s*([^"{}\[\],\s][^{}\[\],\s]*)', r': "\1"', cleaned_json)
            
            # Fix audience_adaptation and technical_depth if they're strings
            fixed_json = re.sub(
                r'"audience_adaptation":\s*{([^}]+)}',
                lambda m: '"audience_adaptation": {' + re.sub(
                    r'"([^"]+)":\s*"([^"]+)"',
                    r'"\1": {"focus": "\2", "tone": "professional"}',
                    m.group(1)
                ) + '}',
                fixed_json
            )
            fixed_json = re.sub(
                r'"technical_depth":\s*{([^}]+)}',
                lambda m: '"technical_depth": {' + re.sub(
                    r'"([^"]+)":\s*"([^"]+)"',
                    r'"\1": {"focus": "\2", "complexity": 1}',
                    m.group(1)
                ) + '}',
                fixed_json
            )
            
            # Remove any trailing commas in nested structures
            fixed_json = re.sub(r',(\s*[}\]])', r'\1', fixed_json)
            
            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError as e:
                # If still failing, try to extract just the essential structure
                match = re.search(r'\{([^{]*?)\}', cleaned_json)
                if match:
                    minimal_json = '{' + match.group(1) + '}'
                    try:
                        return json.loads(minimal_json)
                    except:
                        raise ValueError(f"Failed to parse JSON: {str(e)}\nOutput was: {json_str[:200]}...")
                raise ValueError(f"Failed to parse JSON: {str(e)}\nOutput was: {json_str[:200]}...")
                
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {str(e)}\nOutput was: {json_str[:200]}...")


class ContentStrategyToolArgs(ScriptToolArgs):
    """Arguments for content strategy tool"""
    pass

class ContentStrategyTool(ScriptTool):
    """Tool for generating podcast content strategy"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, llm):
        super().__init__(
            name="Content Strategy Tool",
            description="""Creates podcast content strategy based on format and audience.""",
            llm=llm
        )
        self.args_schema = ContentStrategyToolArgs
        
    def _generate_structure(self, context: ScriptContext) -> Dict[str, Any]:
        """Generate episode structure"""
        prompt = f"""Create the episode structure for a podcast about:
Content: {context.content}
Format: {context.format.get('name')}

Return a valid JSON object with this structure:
{{
    "episode_structure": {{
        "introduction": string,
        "segments": [
            {{
                "name": string,
                "description": string
            }}
        ],
        "conclusion": string
    }}
}}

Keep each segment description concise (max 2-3 sentences).
"""
        response = self.llm.generate(prompt)
        return parse_json_safely(response)
        
    def _generate_key_points(self, context: ScriptContext) -> Dict[str, Any]:
        """Generate key points"""
        prompt = f"""Extract key points from:
Content: {context.content}

Return a valid JSON object with this structure:
{{
    "key_points": [
        {{
            "description": string,
            "significance": string
        }}
    ]
}}

Limit to 2-3 most important points.
"""
        response = self.llm.generate(prompt)
        return parse_json_safely(response)
        
    def _generate_transitions(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Generate transitions between segments"""
        segments = ["Introduction"] + [s["name"] for s in structure["episode_structure"]["segments"]] + ["Conclusion"]
        transitions = []
        
        for i in range(len(segments)-1):
            transitions.append({
                "from": segments[i],
                "to": segments[i+1],
                "text": f"Moving from {segments[i]} to {segments[i+1]}"
            })
            
        return {"transitions": transitions}
        
    def _generate_adaptations(self, context: ScriptContext) -> Dict[str, Any]:
        """Generate audience adaptations"""
        prompt = f"""Create audience adaptations for:
Content: {context.content}
Audience: {context.audience.get('name')}

Return a valid JSON object with this structure:
{{
    "audience_adaptation": {{
        "Technical Professional": {{"focus": string, "tone": string}},
        "Student": {{"focus": string, "tone": string}},
        "General Audience": {{"focus": string, "tone": string}}
    }},
    "technical_depth": {{
        "Beginner": {{"focus": string, "complexity": number}},
        "Intermediate": {{"focus": string, "complexity": number}},
        "Advanced": {{"focus": string, "complexity": number}}
    }}
}}
"""
        response = self.llm.generate(prompt)
        return parse_json_safely(response)
        
    def analyze(self, context: ScriptContext) -> Dict[str, Any]:
        """Generate content strategy"""
        try:
            # Check cache first
            cache_key = cache_manager.get_content_hash(str(context.content))
            cached_result = cache_manager.load_json_cache(cache_key, "content_strategy")
            if cached_result:
                return cached_result
                
            # Generate each component separately
            structure = self._generate_structure(context)
            key_points = self._generate_key_points(context)
            transitions = self._generate_transitions(structure)
            adaptations = self._generate_adaptations(context)
            
            # Combine results
            result = {
                "episode_structure": structure["episode_structure"],
                "key_points": key_points["key_points"],
                "transitions": transitions["transitions"],
                "audience_adaptation": adaptations["audience_adaptation"],
                "technical_depth": adaptations["technical_depth"]
            }
            
            # Validate against schema
            validated = ContentStrategySchema(**result)
            
            # Cache the result
            cache_manager.cache_json(cache_key, "content_strategy", validated)
            
            return validated
            
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
    
    def __init__(self, llm):
        super().__init__(
            name="Script Writing Tool",
            description="""Creates podcast script from content strategy.
            
            IMPORTANT: Return a valid JSON object matching ScriptSchema:
            {
                "segments": [
                    {
                        "speaker": string,
                        "text": string,
                        "style": string,
                        "transitions": {"next": string, "prev": string},
                        "technical_terms": [{"term": string, "definition": string}]
                    }
                ],
                "speakers": [{"name": string, "role": string}],
                "metadata": {
                    "title": string,
                    "description": string,
                    "tags": [string]
                },
                "settings": {
                    "format": string,
                    "style": string
                }
            }
            """,
            llm=llm
        )
        self.args_schema = ScriptWritingToolArgs

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
        response = self.llm.generate(prompt)
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
        response = self.llm.generate(prompt)
        return parse_json_safely(response)

    def _generate_segment(self, context: ScriptContext, segment_info: Dict[str, Any], speakers: List[Dict[str, Any]], prev_segment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a single segment"""
        # Include relevant key points and adaptations for this segment
        relevant_points = [
            point for point in context.content.get('key_points', [])
            if any(keyword in segment_info.get('description', '').lower() 
                  for keyword in point.get('description', '').lower().split())
        ]
        
        audience_adaptation = context.content.get('audience_adaptation', {}).get(
            context.audience.get('name', 'General Audience'),
            {"focus": "", "tone": "professional"}
        )

        prompt = f"""Create a script segment for:
Topic: {segment_info.get('name')}
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
        response = self.llm.generate(prompt)
        return parse_json_safely(response)
        
    def analyze(self, context: ScriptContext) -> Dict[str, Any]:
        """Generate podcast script by parts"""
        try:
            # Check cache first
            cache_key = cache_manager.get_content_hash(str(context.content))
            cached_result = cache_manager.load_json_cache(cache_key, "script_writing")
            if cached_result:
                return cached_result

            # Generate metadata and speakers first
            metadata = self._generate_metadata(context)["metadata"]
            speakers_data = self._generate_speakers(context)["speakers"]

            # Get segments from content strategy
            segments = []
            prev_segment = None
            
            if isinstance(context.content, dict) and "episode_structure" in context.content:
                structure = context.content["episode_structure"]
                
                # Generate introduction segment
                intro_info = {
                    "name": "Introduction",
                    "description": structure.get("introduction", "")
                }
                intro_segment = self._generate_segment(context, intro_info, speakers_data)["segment"]
                segments.append(intro_segment)
                prev_segment = intro_segment
                
                # Generate main segments
                for segment_info in structure.get("segments", []):
                    segment = self._generate_segment(context, segment_info, speakers_data, prev_segment)["segment"]
                    segments.append(segment)
                    prev_segment = segment
                
                # Generate conclusion segment
                conclusion_info = {
                    "name": "Conclusion",
                    "description": structure.get("conclusion", "")
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
            cache_manager.cache_json(cache_key, "script_writing", validated)
            
            return validated
            
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
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, llm):
        super().__init__(
            name="Voice Optimization Tool",
            description="""Optimizes script for voice synthesis.
            
            IMPORTANT: Return a valid JSON object matching OptimizedScriptSchema:
            {
                "segments": [
                    {
                        "speaker": string,
                        "text": string,
                        "style": string,
                        "transitions": {"next": string, "prev": string},
                        "technical_terms": [{"term": string, "definition": string}]
                    }
                ],
                "voice_guidance": {
                    "pronunciation": {"word": string},
                    "emphasis": [{"word": string, "level": number}],
                    "pacing": {"speaker": number},
                    "emotions": {"speaker": string}
                },
                "timing": {
                    "total_duration": number,
                    "segment_durations": {"segment_id": number}
                }
            }
            """,
            llm=llm
        )
        self.args_schema = VoiceOptimizationToolArgs
        
    def analyze(self, context: ScriptContext) -> Dict[str, Any]:
        """Optimize script for voice synthesis"""
        # Check cache first
        cache_key = cache_manager.get_content_hash(str(context.content))
        cached_result = cache_manager.load_json_cache(cache_key, "voice_optimization")
        if cached_result:
            return cached_result
            
        prompt = f"""Optimize this script for voice synthesis:
Script: {json.dumps(context.content)}
Voice Profiles: {json.dumps(context.format.get('voices', {}))}

IMPORTANT: Return a valid JSON object matching OptimizedScriptSchema.
Ensure all numeric values are proper numbers (not strings) and avoid trailing commas.
"""
        
        response = self.llm.generate(prompt)
        result = parse_json_safely(response)
        
        # Cache the result
        cache_manager.cache_json(cache_key, "voice_optimization", result)
        
        return result
        
    def enhance_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance results with validation"""
        try:
            return OptimizedScriptSchema(**results)
        except Exception as e:
            return {
                "error": f"Failed to validate optimized script: {str(e)}",
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

class QualityControlToolArgs(ScriptToolArgs):
    """Arguments for quality control tool"""
    pass

class QualityControlTool(ScriptTool):
    """Tool for script quality control"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, llm):
        super().__init__(
            name="Quality Control Tool",
            description="""Reviews and improves podcast scripts.
            
            IMPORTANT: Return a valid JSON object matching QualityReviewSchema:
            {
                "final_script": {
                    "segments": [ScriptSegmentSchema],
                    "speakers": [{"name": string, "role": string}],
                    "metadata": {"title": string, "description": string, "tags": [string]},
                    "settings": {"format": string, "style": string}
                },
                "quality_metrics": {
                    "content_accuracy": number,
                    "conversation_flow": number,
                    "audience_fit": number,
                    "technical_accuracy": number,
                    "engagement": number
                },
                "improvements": [{"type": string, "description": string}],
                "recommendations": {
                    "content": [string],
                    "delivery": [string]
                }
            }
            """,
            llm=llm
        )
        self.args_schema = QualityControlToolArgs
        
    def analyze(self, context: ScriptContext) -> Dict[str, Any]:
        """Review and improve script"""
        # Check cache first
        cache_key = cache_manager.get_content_hash(str(context.content))
        cached_result = cache_manager.load_json_cache(cache_key, "quality_control")
        if cached_result:
            return cached_result
            
        prompt = f"""Review this podcast script for:
Script: {json.dumps(context.content)}
Target Audience: {context.audience.get('name')} (Technical Depth: {context.audience.get('technical_depth')})
Expertise Level: {context.expertise.get('name')} (Complexity: {context.expertise.get('complexity')})

Check:
1. Content accuracy and clarity
2. Natural conversation flow
3. Appropriate pacing
4. Technical accuracy
5. Audience engagement

IMPORTANT: Return a valid JSON object matching QualityReviewSchema.
Ensure all numeric values are proper numbers (not strings) and avoid trailing commas.
"""
        
        response = self.llm.generate(prompt)
        result = parse_json_safely(response)
        
        # Cache the result
        cache_manager.cache_json(cache_key, "quality_control", result)
        
        return result
        
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
