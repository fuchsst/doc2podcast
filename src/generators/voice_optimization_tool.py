"""Voice optimization tool for podcast scripts"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import yaml

from ..config import PromptManager, Settings
from .base import ScriptTool, ToolContext
from .schemas import (
    ContentStrategySchema,
    ScriptSchema,
    VoiceOptimizedScript,
    VoiceSegment,
    VoiceGuidance,
    SpeakerConfig,
    VoiceProfile,
    VoiceParameters,
    ScriptMetadata,
    QualityReviewSchema
)
from ..utils.callback_handler import PipelineCallback, StepType
from ..utils.text_utils import parse_json_safely
from ..utils.cache_manager import cache_manager


@dataclass
class VoiceOptimizationContext(ToolContext):
    """Context for voice optimization"""
    content_strategy: ContentStrategySchema
    script: ScriptSchema
    voice_settings: List[Dict[str, Any]]
    quality_review: QualityReviewSchema
    metadata: ScriptMetadata
    settings: Dict[str, Any]


class VoiceOptimizationTool(ScriptTool):
    """Tool for optimizing script for voice synthesis"""
    
    def __init__(
        self,
        name: str,
        description: str,
        llm: Any,
        callback: Optional[PipelineCallback] = None
    ):
        super().__init__(name, description, llm)
        self.callback = callback
        # Load speakers config
        with open('config/speakers.yaml', 'r') as f:
            self.speakers_config = yaml.safe_load(f)
        
    def analyze(self, context: VoiceOptimizationContext) -> VoiceOptimizedScript:
        """Analyze and optimize script for voice synthesis"""
        try:
            # Check cache first
            cache_key = cache_manager.get_content_hash(str(context.script) + str(context.voice_settings))
            cached_result = cache_manager.load_json_cache(cache_key, "voice_optimization")
            if cached_result:
                return VoiceOptimizedScript(**cached_result)

            if self.callback:
                self.callback.on_substep_start("Starting voice optimization")
                self.callback.on_script_generation(
                    progress=0,
                    message="Optimizing voice settings...",
                    substeps=[
                        {"name": "Segment Analysis", "status": "in_progress"},
                        {"name": "Voice Guidance", "status": "pending"},
                        {"name": "Final Optimization", "status": "pending"}
                    ]
                )
            
            # Create voice segments with optimized parameters
            segments = []
            total_segments = len(context.script.segments)
            
            for i, segment in enumerate(context.script.segments):
                # Get voice settings for this segment
                settings = context.voice_settings[i]
                
                # Analyze text for voice optimization
                optimization_prompt = f"""
                Analyze the following text for voice optimization:
                Text: {segment.text}
                Speaker: {settings["speaker"]["name"]}
                Style: {settings["speaker"]["style_tags"]}
                
                Provide:
                1. Natural emphasis points
                2. Pause locations and durations
                3. Pronunciation guidance for technical terms
                4. Emotional tone variations
                5. Speaking pace adjustments
                """
                
                optimization_result = self.llm.generate(optimization_prompt)
                
                # Create voice profile
                voice_profile = VoiceProfile(
                    model=settings["speaker"]["voice_model"],
                    voice_preset=settings["speaker"]["voice_preset"],
                    style_tags=settings["speaker"]["style_tags"],
                    voice_parameters=VoiceParameters(**settings["voice_parameters"]),
                    reference_audio=None,
                    reference_text=None
                )
                
                # Create speaker config
                speaker_config = SpeakerConfig(
                    name=settings["speaker"]["name"],
                    voice_profile=voice_profile,
                    character_background={}
                )
                
                # Extract optimization details
                emphasis_words = self._extract_emphasis_words(optimization_result)
                pauses = self._extract_pauses(optimization_result)
                pronunciations = self._extract_pronunciations(optimization_result)
                
                # Create voice segment
                voice_segment = VoiceSegment(
                    speaker=speaker_config,
                    text=segment.text,
                    style=settings["speaker"]["style_tags"][0] if settings["speaker"]["style_tags"] else "neutral",
                    voice_parameters=VoiceParameters(**settings["voice_parameters"]),
                    emphasis_words=emphasis_words,
                    pauses=pauses,
                    pronunciation_guide=pronunciations
                )
                
                segments.append(voice_segment)
                
                if self.callback:
                    progress = int((i + 1) / total_segments * 50)  # First 50% for segment analysis
                    self.callback.on_script_generation(
                        progress=progress,
                        message=f"Analyzing segment {i + 1}/{total_segments}...",
                        substeps=[
                            {"name": "Segment Analysis", "status": "in_progress"},
                            {"name": "Voice Guidance", "status": "pending"},
                            {"name": "Final Optimization", "status": "pending"}
                        ]
                    )
            
            if self.callback:
                self.callback.on_script_generation(
                    progress=50,
                    message="Generating voice guidance...",
                    substeps=[
                        {"name": "Segment Analysis", "status": "completed"},
                        {"name": "Voice Guidance", "status": "in_progress"},
                        {"name": "Final Optimization", "status": "pending"}
                    ]
                )
            
            # Generate overall voice guidance
            guidance_prompt = f"""
            Create voice guidance for a podcast script with the following details:
            Title: {context.metadata.title}
            Description: {context.metadata.description}
            Format: {context.settings.get("format")}
            Target Audience: {context.settings.get("target_audience")}
            Quality Review: {context.quality_review}
            
            Consider:
            1. Overall pacing strategy
            2. Emotional progression
            3. Key emphasis points
            4. Technical term pronunciation consistency
            5. Natural conversation flow
            """
            
            guidance_result = self.llm.generate(guidance_prompt)
            
            if self.callback:
                self.callback.on_script_generation(
                    progress=75,
                    message="Finalizing optimization...",
                    substeps=[
                        {"name": "Segment Analysis", "status": "completed"},
                        {"name": "Voice Guidance", "status": "completed"},
                        {"name": "Final Optimization", "status": "in_progress"}
                    ]
                )
            
            # Create voice guidance
            voice_guidance = VoiceGuidance(
                pronunciation=self._extract_pronunciation_guide(guidance_result),
                emphasis=self._extract_emphasis_guide(guidance_result),
                pacing=self._extract_pacing_guide(guidance_result),
                emotions=self._extract_emotion_guide(guidance_result)
            )
            
            # Create optimized script
            optimized_script = VoiceOptimizedScript(
                metadata=context.metadata,
                content_strategy=context.content_strategy,
                segments=segments,
                voice_guidance=voice_guidance,
                quality_review=context.quality_review,
                settings=context.settings
            )
            
            # Cache the result
            cache_manager.cache_json(cache_key, "voice_optimization", optimized_script.model_dump())
            
            if self.callback:
                self.callback.on_script_generation(
                    progress=100,
                    message="Voice optimization completed",
                    substeps=[
                        {"name": "Segment Analysis", "status": "completed"},
                        {"name": "Voice Guidance", "status": "completed"},
                        {"name": "Final Optimization", "status": "completed"}
                    ]
                )
            
            return optimized_script
            
        except Exception as e:
            if self.callback:
                self.callback.on_error(StepType.VOICE_OPTIMIZATION, str(e))
            raise RuntimeError(f"Voice optimization failed: {str(e)}")

    def enhance_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance and validate results"""
        try:
            if isinstance(results, VoiceOptimizedScript):
                return results.model_dump()
            return VoiceOptimizedScript(**results).model_dump()
        except Exception as e:
            return {
                "error": f"Failed to validate voice optimization results: {str(e)}",
                "original_results": results
            }

    def generate_structured_script(
        self,
        optimized_script: VoiceOptimizedScript,
        context: VoiceOptimizationContext
    ) -> List[Dict[str, Any]]:
        """Generate structured script with speaker, style, and text"""
        try:
            # Check cache first
            cache_key = cache_manager.get_content_hash(str(optimized_script))
            cached_result = cache_manager.load_json_cache(cache_key, "structured_script")
            if cached_result:
                return cached_result

            if self.callback:
                self.callback.on_substep_start("Generating structured script")
                self.callback.on_script_generation(
                    progress=0,
                    message="Creating structured script...",
                    substeps=[
                        {"name": "Structured Generation", "status": "in_progress"}
                    ]
                )

            # Build character background context
            character_backgrounds = {}
            for category in self.speakers_config["voices"]:
                for speaker, info in self.speakers_config["voices"][category].items():
                    character_backgrounds[speaker] = info.get("character_background", {})

            # Create prompt for structured script generation
            prompt = f"""
            Generate a natural-sounding podcast script based on the following inputs.

            Content Strategy:
            {context.content_strategy.model_dump()}

            Quality Review Metrics:
            {context.quality_review.model_dump()}

            Available Speakers and Their Backgrounds:
            {character_backgrounds}

            Format: {context.settings.get("format")}
            Target Audience: {context.settings.get("target_audience")}
            Expertise Level: {context.settings.get("expertise_level")}

            Return a valid JSON array where each object has exactly these fields:
            1. "speaker": The speaker's name (one of: {", ".join(character_backgrounds.keys())})
            2. "style": The speaking style from their available profiles
            3. "text": The natural speaking text for that segment

            Rules for natural conversation:
            1. Include realistic speech patterns:
               - Filler words (um, uh, well, you know)
               - Thinking pauses (hmm, let me see)
               - Reactions (ah, oh, wow, really)
               - Discourse markers (actually, basically, honestly)
            2. Show active listening:
               - Brief acknowledgments (right, yes, I see)
               - Encouraging responses (that's interesting, tell me more)
               - Clarifying questions (so you mean...?)
            3. Maintain natural flow:
               - Incomplete sentences
               - Self-corrections
               - Overlapping ideas
               - use dots and dashes to indicate pauses
            4. Express personality:
               - Each speaker's unique speaking style
               - Character-appropriate reactions
               - Natural enthusiasm or thoughtfulness

            Example of EXACT expected format:
            [
                {{
                    "speaker": "alex_rivera",
                    "style": "curious",
                    "text": "Hmm, that's really fascinating!"
                }},
                {{
                    "speaker": "alex_rivera",
                    "style": "excited",
                    "text": "So, um, how exactly does this neural network thing work? Like, what's actually happening under the hood?"
                }},
                {{
                    "speaker": "sarah_chen",
                    "style": "technical",
                    "text": "Well, let me think about how to explain this... You know, it's actually quite interesting. The neural network - uh, think of it as layers of interconnected nodes, right? Each one processing information in its own way."
                }}
            ]

            Your response must be a single JSON array exactly matching this structure.
            Do not include any additional text before or after the JSON.
            """

            # Generate structured script
            response = self.llm.generate(prompt)
            result = parse_json_safely(response)

            # Cache the result
            cache_manager.cache_json(cache_key, "structured_script", result)

            if self.callback:
                self.callback.on_script_generation(
                    progress=100,
                    message="Structured script generation completed",
                    substeps=[
                        {"name": "Structured Generation", "status": "completed"}
                    ]
                )

            return result

        except Exception as e:
            if self.callback:
                self.callback.on_error(StepType.VOICE_OPTIMIZATION, str(e))
            raise RuntimeError(f"Structured script generation failed: {str(e)}")

    def _extract_pronunciations(self, result: str) -> Dict[str, str]:
        """Extract pronunciation guidance from LLM result"""
        pronunciation_prompt = f"""
        From the following optimization result, extract pronunciation guidance for technical terms and proper nouns.
        Text: {result}

        Return a valid JSON object mapping terms to their pronunciation guides.
        Follow these rules EXACTLY:
        1. Use double quotes for all strings
        2. Include phonetic pronunciation and syllable emphasis
        3. Mark stressed syllables with CAPS
        4. Include any specific accent or tone guidance

        Example format:
        {{
            "TensorFlow": "TEN-sor-flow",
            "PyTorch": "PIE-torch",
            "LSTM": "L-S-T-M (each letter spoken separately)",
            "ResNet": "REZ-net"
        }}

        Your response must be a single JSON object exactly matching this structure.
        """
        pronunciation_result = self.llm.generate(pronunciation_prompt)
        return parse_json_safely(pronunciation_result)
    
    def _extract_pronunciation_guide(self, result: str) -> Dict[str, str]:
        """Extract overall pronunciation guidance"""
        guide_prompt = f"""
        Create a comprehensive pronunciation guide from the following guidance:
        {result}

        Return a valid JSON object with pronunciation rules and patterns.
        Include:
        1. Common technical terms
        2. Recurring phrases
        3. Specialized vocabulary
        4. Accent and dialect considerations

        Example format:
        {{
            "technical_terms": {{
                "API": "A-P-I (spell it out)",
                "SQL": "SEEK-well or S-Q-L"
            }},
            "phrases": {{
                "neural network": "NOOR-al NET-work",
                "machine learning": "ma-SHEEN LEARN-ing"
            }}
        }}
        """
        guide_result = self.llm.generate(guide_prompt)
        return parse_json_safely(guide_result)
    
    def _extract_emphasis_guide(self, result: str) -> List[Dict[str, Any]]:
        """Extract overall emphasis guidance"""
        emphasis_prompt = f"""
        Create emphasis guidance from the following:
        {result}

        Return a valid JSON array of emphasis points with:
        1. Words or phrases to emphasize
        2. Type of emphasis (stress, tone, pause)
        3. Purpose of emphasis
        4. Context or trigger

        Example format:
        [
            {{
                "text": "critical",
                "emphasis_type": "stress",
                "purpose": "highlight importance",
                "context": "when explaining key concepts"
            }},
            {{
                "text": "but here's the interesting part",
                "emphasis_type": "tone_shift",
                "purpose": "build anticipation",
                "context": "before revealing key insights"
            }}
        ]
        """
        emphasis_result = self.llm.generate(emphasis_prompt)
        return parse_json_safely(emphasis_result)
    
    def _extract_pacing_guide(self, result: str) -> Dict[str, float]:
        """Extract overall pacing guidance"""
        pacing_prompt = f"""
        Create pacing guidance from the following:
        {result}

        Return a valid JSON object with pacing factors as decimal values between 0.5 and 2.0:
        1. Base speaking rate
        2. Variation factors for different content types
        3. Transition pacing
        4. Emphasis pacing

        Example format:
        {{
            "base_rate": 1.0,
            "technical_explanation": 0.8,
            "excitement": 1.3,
            "transition": 0.9,
            "emphasis": 0.7
        }}
        """
        pacing_result = self.llm.generate(pacing_prompt)
        return parse_json_safely(pacing_result)
    
    def _extract_emotion_guide(self, result: str) -> Dict[str, str]:
        """Extract overall emotion guidance"""
        emotion_prompt = f"""
        Create emotion guidance from the following:
        {result}

        Return a valid JSON object mapping content types to emotional styles.
        Include:
        1. Overall tone
        2. Emotional progression
        3. Response patterns
        4. Transition emotions

        Example format:
        {{
            "introduction": "enthusiastic and welcoming",
            "technical_parts": "confident and clear",
            "challenges": "empathetic and supportive",
            "successes": "excited and celebratory",
            "transitions": "curious and engaging"
        }}
        """
        emotion_result = self.llm.generate(emotion_prompt)
        return parse_json_safely(emotion_result)

    def _extract_emphasis_words(self, result: str) -> List[Dict[str, Any]]:
        """Extract emphasis words from optimization result"""
        emphasis_prompt = f"""
        Extract words and phrases that need emphasis from:
        {result}

        Return a valid JSON array of emphasis points.
        Include:
        1. Word or phrase to emphasize
        2. Type of emphasis
        3. Reason for emphasis

        Example format:
        [
            {{
                "text": "crucial",
                "emphasis_type": "strong",
                "reason": "key concept"
            }},
            {{
                "text": "however",
                "emphasis_type": "pause",
                "reason": "transition"
            }}
        ]
        """
        emphasis_result = self.llm.generate(emphasis_prompt)
        return parse_json_safely(emphasis_result)

    def _extract_pauses(self, result: str) -> List[Dict[str, Any]]:
        """Extract pause locations from optimization result"""
        pause_prompt = f"""
        Extract pause locations and durations from:
        {result}

        Return a valid JSON array of pause points.
        Include:
        1. Location in text
        2. Duration (short, medium, long)
        3. Purpose of pause

        Example format:
        [
            {{
                "location": "before 'however'",
                "duration": "medium",
                "purpose": "emphasis transition"
            }},
            {{
                "location": "after key point",
                "duration": "long",
                "purpose": "allow comprehension"
            }}
        ]
        """
        pause_result = self.llm.generate(pause_prompt)
        return parse_json_safely(pause_result)
