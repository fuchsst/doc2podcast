"""Tools for podcast script generation"""
from typing import Dict, Any, Optional

from .base import ScriptTool, ScriptContext, ScriptToolArgs
from .schemas import ContentStrategySchema
from ..utils.cache_manager import cache_manager
from ..utils.callback_handler import PipelineCallback
from ..utils.text_utils import parse_json_safely

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

Return a valid JSON object with this structure. Follow these rules EXACTLY:
1. Use double quotes for all strings
2. No trailing commas
3. No line breaks within strings
4. Escape any quotes within strings
5. Keep descriptions concise (2-3 sentences max)

Example of EXACT expected format:
{{
    "outline": {{
        "introduction": "A brief overview of modern web development practices",
        "main_segments": [
            {{
                "title": "Frontend Frameworks",
                "description": "Overview of React, Vue, and Angular"
            }}
        ],
        "conclusion": "Summary of key takeaways and future trends"
    }}
}}

Your response must be a single JSON object exactly matching this structure.
Do not include any additional text before or after the JSON.
"""
        response = self.llm.call([{"role": "user", "content": prompt}])
        return parse_json_safely(response)
        
    def _generate_key_points(self, context: ScriptContext) -> Dict[str, Any]:
        """Generate key points"""
        prompt = f"""Extract key points from:
Content: {context.content}

Return a valid JSON object with this structure. Follow these rules EXACTLY:
1. Use double quotes for all strings
2. No trailing commas
3. No line breaks within strings
4. Escape any quotes within strings
5. Include 3-5 key points
6. Keep each point concise (1 sentence)

Example of EXACT expected format:
{{
    "key_points": [
        "React is the most popular frontend framework",
        "Vue offers excellent performance and simplicity",
        "Angular provides enterprise-level features"
    ]
}}

Your response must be a single JSON object exactly matching this structure.
Do not include any additional text before or after the JSON.
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

Return a valid JSON object with this structure. Follow these rules EXACTLY:
1. Use double quotes for all strings
2. No trailing commas
3. No line breaks within strings
4. Escape any quotes within strings
5. Use only predefined values for technical_level and engagement_style
6. Keep explanations concise

Example of EXACT expected format:
{{
    "audience_adaptations": {{
        "technical_level": "intermediate",
        "engagement_style": "conversational",
        "examples_type": "real-world",
        "vocabulary_adjustments": [
            "dependency injection -> plugging in different parts of code",
            "middleware -> request processor"
        ]
    }}
}}

Your response must be a single JSON object exactly matching this structure.
Do not include any additional text before or after the JSON.
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

