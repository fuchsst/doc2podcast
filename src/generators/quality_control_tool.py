"""Tools for podcast script generation"""
import json
from typing import Dict, Any, Optional, List

from .base import ScriptTool, ScriptContext, ScriptToolArgs
from .schemas import QualityReviewSchema
from ..utils.cache_manager import cache_manager
from ..utils.callback_handler import PipelineCallback
from ..utils.text_utils import parse_json_safely

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

Return a valid JSON object with this structure. Follow these rules EXACTLY:
1. Use double quotes for all strings
2. No trailing commas
3. No line breaks within strings
4. Escape any quotes within strings
5. All numeric values must be proper numbers between 0 and 1
6. Keep descriptions concise and actionable
7. Include at least 2-3 improvements and recommendations

Example of EXACT expected format:
{{
    "quality_metrics": {{
        "content_accuracy": 0.85,
        "conversation_flow": 0.9,
        "audience_fit": 0.8,
        "technical_accuracy": 0.95,
        "engagement": 0.75
    }},
    "improvements": [
        {{
            "type": "content",
            "description": "Add more real-world examples to illustrate technical concepts"
        }},
        {{
            "type": "structure",
            "description": "Strengthen transitions between framework comparisons"
        }},
        {{
            "type": "language",
            "description": "Simplify technical terminology in the React section"
        }}
    ],
    "recommendations": {{
        "content": [
            "Include code snippets when discussing implementation details",
            "Add performance comparison metrics between frameworks"
        ],
        "delivery": [
            "Maintain consistent pacing during technical explanations",
            "Use more enthusiastic tone when highlighting key features"
        ]
    }}
}}

Your response must be a single JSON object exactly matching this structure.
Do not include any additional text before or after the JSON.
IMPORTANT:
- All numeric values must be proper numbers between 0 and 1 (e.g., 0.85, not 85%)
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
