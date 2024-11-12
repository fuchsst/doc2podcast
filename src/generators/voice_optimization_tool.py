"""Tools for podcast script generation"""

import json
import re
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ConfigDict

from .base import ScriptTool, ScriptContext, ScriptToolArgs
from .schemas import OptimizedScriptSchema
from ..utils.cache_manager import cache_manager
from ..utils.callback_handler import PipelineCallback, StepType, ProgressUpdate
from ..utils.text_utils import parse_json_safely


class VoiceOptimizationToolArgs(ScriptToolArgs):
    """Arguments for voice optimization tool"""

    pass


class VoiceOptimizationTool(ScriptTool):
    """Tool for optimizing scripts for voice synthesis"""

    def __init__(
        self,
        name: str,
        description: str,
        llm: Any,
        callback: Optional[PipelineCallback] = None,
    ):
        super().__init__(name, description, llm)
        self.args_schema = VoiceOptimizationToolArgs
        self.callback = callback

    def analyze(self, context: ScriptContext) -> Dict[str, Any]:
        """Optimize script for voice synthesis"""
        try:
            # Check cache first
            cache_key = cache_manager.get_content_hash(str(context.content))
            cached_result = cache_manager.load_json_cache(
                cache_key, "voice_optimization"
            )
            if cached_result:
                return cached_result

            if self.callback:
                self.callback.on_script_generation(
                    progress=0,
                    message="Optimizing script for voice synthesis...",
                    substeps=[{"name": "Voice Optimization", "status": "in_progress"}],
                )

            prompt = f"""Optimize this script for voice synthesis:
Script: {json.dumps(context.content)}
Voice Profiles: {json.dumps(context.format.get('voices', {}))}

Return a valid JSON object with this structure. Follow these rules EXACTLY:
1. Use double quotes for all strings
2. No trailing commas
3. No line breaks within strings
4. Escape any quotes within strings
5. All numeric values must be proper numbers
6. Keep text content concise and clear
7. Include specific voice guidance for each speaker

Example of EXACT expected format:
{{
    "segments": [
        {{
            "speaker": "Alex Chen",
            "text": "Welcome to our discussion on modern web development",
            "style": "enthusiastic",
            "transitions": {{
                "next": "Let's explore the fundamentals",
                "prev": "As we mentioned in the introduction"
            }},
            "technical_terms": [
                {{
                    "term": "web development",
                    "definition": "The process of building and maintaining websites"
                }}
            ]
        }}
    ],
    "voice_guidance": {{
        "pronunciation": {{
            "React": "ree-act",
            "Vue": "view"
        }},
        "emphasis": [
            {{
                "word": "critical",
                "level": 0.8
            }}
        ],
        "pacing": {{
            "Alex Chen": 1.2,
            "Dr. Sarah Smith": 0.9
        }},
        "emotions": {{
            "Alex Chen": "enthusiastic",
            "Dr. Sarah Smith": "authoritative"
        }}
    }},
    "timing": {{
        "total_duration": 1200,
        "segment_durations": {{
            "introduction": 180,
            "main_content": 840,
            "conclusion": 180
        }}
    }}
}}

Your response must be a single JSON object exactly matching this structure.
Do not include any additional text before or after the JSON.
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
                    substeps=[{"name": "Voice Optimization", "status": "completed"}],
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
                "original_results": results,
            }
