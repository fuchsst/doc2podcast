"""Script generation module using CrewAI agents"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..config import PromptManager, Settings
from ..models.podcast_script import (
    PodcastScript,
    PodcastMetadata,
    Speaker,
    ScriptSegment,
    VoiceParameters,
    Reference
)
from ..utils.callback_handler import PipelineCallback, StepType, ProgressUpdate
from ..utils.cache_manager import cache_manager
from .base import ScriptContext, ScriptGenerator, ResultsFormatter
from .content_strategy_tool import ContentStrategyTool
from .script_writing_tool import ScriptWritingTool
from .voice_optimization_tool import VoiceOptimizationTool
from .quality_control_tool import QualityControlTool


@dataclass
class ScriptGenerationConfig:
    """Configuration for script generation"""
    podcast_preset: str
    target_audience: str
    expertise_level: str
    guidance_prompt: Optional[str] = None

class PodcastScriptGenerator(ScriptGenerator, ResultsFormatter):
    """Generates podcast scripts using CrewAI agents"""
    
    def __init__(self, settings: Settings, callback: Optional[PipelineCallback] = None):
        super().__init__(settings, callback)
        self.prompt_manager = PromptManager(settings)
        
        # Initialize LLM
        self.llm = settings.get_llm()
        
        # Initialize tools with callback
        self.content_tool = ContentStrategyTool(
            name="Content Strategy Tool",
            description="Creates podcast content strategy with episode structure, key points, transitions, and audience adaptations",
            llm=self.llm,
            callback=callback
        )
        self.script_tool = ScriptWritingTool(
            name="Script Writing Tool",
            description="Creates podcast script from content strategy with proper structure and flow",
            llm=self.llm,
            callback=callback
        )
        self.voice_tool = VoiceOptimizationTool(
            name="Voice Optimization Tool",
            description="Optimizes script for voice synthesis with proper pacing, emphasis, and emotional guidance",
            llm=self.llm,
            callback=callback
        )
        self.quality_tool = QualityControlTool(
            name="Quality Control Tool",
            description="Reviews and improves script for content accuracy, flow, and audience fit",
            llm=self.llm,
            callback=callback
        )
        
    def generate(self, content: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        """Implement abstract generate method from Generator base class"""
        config = kwargs.get('config', ScriptGenerationConfig(
            podcast_preset=kwargs.get('podcast_preset', 'default'),
            target_audience=kwargs.get('target_audience', 'general'),
            expertise_level=kwargs.get('expertise_level', 'beginner'),
            guidance_prompt=kwargs.get('guidance_prompt')
        ))
        return self.generate_script(content, config)
        
    def generate_content_strategy(
        self,
        content: Dict[str, Any],
        config: Optional[ScriptGenerationConfig] = None
    ) -> Dict[str, Any]:
        """Generate content strategy"""
        try:
            # Check cache first
            cache_key = cache_manager.get_content_hash(str(content))
            cached_result = cache_manager.load_json_cache(cache_key, "content_strategy")
            if cached_result:
                return cached_result
                
            # Get format configuration
            format_config = self.prompt_manager.get_interview_prompt(config.podcast_preset)
            
            # Get audience and expertise configurations
            audiences = self.prompt_manager.get_target_audiences(config.podcast_preset)
            audience = next(a for a in audiences if a.name == config.target_audience)
            
            expertise_levels = self.prompt_manager.get_expertise_levels(config.podcast_preset)
            expertise = next(l for l in expertise_levels if l.name == config.expertise_level)
            
            # Create script context
            context = ScriptContext(
                content=content,
                format=format_config.model_dump(),
                audience=audience.model_dump(),
                expertise=expertise.model_dump(),
                guidance=config.guidance_prompt
            )
            
            # Generate content strategy
            result = self.content_tool.analyze(context)
            
            # Cache result
            cache_manager.cache_json(cache_key, "content_strategy", result)
            
            return result
                
        except Exception as e:
            raise RuntimeError(f"Failed to generate content strategy: {str(e)}")
            
    def write_script(
        self,
        content: Dict[str, Any],
        strategy: Dict[str, Any],
        config: Optional[ScriptGenerationConfig] = None
    ) -> Dict[str, Any]:
        """Write podcast script"""
        try:
            # Check cache first
            cache_key = cache_manager.get_content_hash(str(content) + str(strategy))
            cached_result = cache_manager.load_json_cache(cache_key, "script_writing")
            if cached_result:
                return cached_result
                
            # Get format configuration
            format_config = self.prompt_manager.get_interview_prompt(config.podcast_preset)
            
            # Get audience and expertise configurations
            audiences = self.prompt_manager.get_target_audiences(config.podcast_preset)
            audience = next(a for a in audiences if a.name == config.target_audience)
            
            expertise_levels = self.prompt_manager.get_expertise_levels(config.podcast_preset)
            expertise = next(l for l in expertise_levels if l.name == config.expertise_level)
            
            # Create script context with strategy as content
            context = ScriptContext(
                content=strategy,
                format=format_config.model_dump(),
                audience=audience.model_dump(),
                expertise=expertise.model_dump(),
                guidance=config.guidance_prompt
            )
            
            # Write script
            result = self.script_tool.analyze(context)
            
            # Cache result
            cache_manager.cache_json(cache_key, "script_writing", result)
            
            return result
                
        except Exception as e:
            raise RuntimeError(f"Failed to write script: {str(e)}")
            
    def optimize_voice_settings(
        self,
        script: Dict[str, Any],
        config: Optional[ScriptGenerationConfig] = None
    ) -> Dict[str, Any]:
        """Optimize voice settings"""
        try:
            # Check cache first
            cache_key = cache_manager.get_content_hash(str(script))
            cached_result = cache_manager.load_json_cache(cache_key, "voice_optimization")
            if cached_result:
                return cached_result
                
            # Get format configuration
            format_config = self.prompt_manager.get_interview_prompt(config.podcast_preset)
            
            # Create script context
            context = ScriptContext(
                content=script,
                format=format_config.model_dump()
            )
            
            # Optimize voice settings
            result = self.voice_tool.analyze(context)
            
            # Cache result
            cache_manager.cache_json(cache_key, "voice_optimization", result)
            
            return result
                
        except Exception as e:
            raise RuntimeError(f"Failed to optimize voice settings: {str(e)}")
            
    def review_script_quality(
        self,
        script: Dict[str, Any],
        config: Optional[ScriptGenerationConfig] = None
    ) -> Dict[str, Any]:
        """Review script quality"""
        try:
            # Check cache first
            cache_key = cache_manager.get_content_hash(str(script))
            cached_result = cache_manager.load_json_cache(cache_key, "quality_control")
            if cached_result:
                return cached_result
                
            # Get format configuration
            format_config = self.prompt_manager.get_interview_prompt(config.podcast_preset)
            
            # Get audience and expertise configurations
            audiences = self.prompt_manager.get_target_audiences(config.podcast_preset)
            audience = next(a for a in audiences if a.name == config.target_audience)
            
            expertise_levels = self.prompt_manager.get_expertise_levels(config.podcast_preset)
            expertise = next(l for l in expertise_levels if l.name == config.expertise_level)
            
            # Create script context
            context = ScriptContext(
                content=script,
                format=format_config.model_dump(),
                audience=audience.model_dump(),
                expertise=expertise.model_dump(),
                guidance=config.guidance_prompt
            )
            
            # Review script
            result = self.quality_tool.analyze(context)
            
            # Cache result
            cache_manager.cache_json(cache_key, "quality_control", result)
            
            return result
                
        except Exception as e:
            raise RuntimeError(f"Failed to review script quality: {str(e)}")
                        
    def generate_script(
        self,
        content: Dict[str, Any],
        config: ScriptGenerationConfig
    ) -> PodcastScript:
        """Generate complete podcast script"""
        try:
            if self.callback:
                self.callback.on_step_start(StepType.SCRIPT_GENERATION, "Starting script generation")
            
            # Check cache first
            cache_key = cache_manager.get_content_hash(str(content))
            cached_result = cache_manager.load_json_cache(cache_key, "complete_script")
            if cached_result:
                return PodcastScript(**cached_result)
            
            # Step 1: Generate content strategy
            strategy = self.generate_content_strategy(content, config)
            
            # Step 2: Write script
            script = self.write_script(content, strategy, config)
            
            # Step 3: Optimize voice settings
            optimized = self.optimize_voice_settings(script, config)
            
            # Step 4: Review script quality
            reviewed = self.review_script_quality(optimized, config)
            
            # Format results
            results = [strategy, script, optimized, reviewed]
            
            # Use the optimized script's segments and voice guidance
            voice_guidance = optimized["voice_guidance"]

            # Get voice configurations from roles
            format_config = self.prompt_manager.get_interview_prompt(config.podcast_preset)
            roles = format_config.roles
            
            # Convert to PodcastScript object
            script = PodcastScript(
                metadata=PodcastMetadata(
                    title=script.get("metadata", {}).get("title", "Untitled Podcast"),
                    description=script.get("metadata", {}).get("description"),
                    source_document=content.get("source", {}).get("path"),
                    tags=script.get("metadata", {}).get("tags", []),
                    duration=None  # Will be set during audio generation
                ),
                segments=[
                    ScriptSegment(
                        speaker=Speaker(
                            name=segment["speaker"],
                            voice_model=self._get_voice_config(roles[segment["speaker"]].voice if segment["speaker"] in roles else None),
                            voice_preset=None,  # Will be set during voice configuration
                            style_tags=[roles[segment["speaker"]].style] if segment["speaker"] in roles and roles[segment["speaker"]].style else [],
                            voice_parameters=VoiceParameters(
                                pace=voice_guidance["pacing"].get(segment["speaker"], 1.0),
                                pitch=voice_guidance.get("pitch", {}).get(segment["speaker"], 1.0),
                                energy=voice_guidance.get("emotions", {}).get(segment["speaker"], {}).get("energy", 0.5),
                                emotion=voice_guidance.get("emotions", {}).get(segment["speaker"], {}).get("type", "neutral"),
                                variation=voice_guidance.get("emphasis", {}).get(segment["speaker"], {}).get("variation", 0.5)
                            ),
                            reference=Reference(
                                audio_path=None,  # Will be set during voice configuration
                                text=segment.get("reference_text", "")
                            ) if segment.get("reference_text") else None
                        ),
                        text=segment["text"],
                        duration=None,  # Will be set during audio generation
                        audio_path=None  # Will be set during audio generation
                    )
                    for segment in optimized["segments"]
                ],
                settings={
                    "format": config.podcast_preset,
                    "target_audience": config.target_audience,
                    "expertise_level": config.expertise_level,
                    "voice_guidance": voice_guidance,
                    "quality_metrics": reviewed.get("quality_metrics", {}),
                    "improvements": reviewed.get("improvements", []),
                    "recommendations": reviewed.get("recommendations", {})
                }
            )
            
            # Cache the complete script
            cache_manager.cache_json(cache_key, "complete_script", script.model_dump())
            
            if self.callback:
                self.callback.on_step_complete(StepType.SCRIPT_GENERATION, "Script generation completed successfully")
            
            return script
            
        except Exception as e:
            error = f"Script generation failed: {str(e)}"
            if self.callback:
                self.callback.on_error(StepType.SCRIPT_GENERATION, error)
            raise RuntimeError(error)

    def _get_voice_config(self, voice_id: str) -> str:
        """Get voice model from voice ID
        
        Args:
            voice_id (str): Voice ID in format "category.voice_name.profile_type"
            
        Returns:
            str: Voice model name
        """
        if not voice_id:
            return "default"
            
        try:
            # Handle both formats: "category.voice_name.profile_type" and "voice_name.profile_type"
            parts = voice_id.split(".")
            if len(parts) == 3:
                category, voice_name, profile_type = parts
            elif len(parts) == 2:
                category = "professional"  # Default category
                voice_name, profile_type = parts
            else:
                return "default"
                
            try:
                profile = self.prompt_manager.get_voice_profile(category, voice_name, profile_type)
                return profile["voice_profile"]["model"]
            except Exception as e:
                print(f"Warning: Failed to get voice profile for {voice_id}: {str(e)}")
                return "default"
                
        except Exception as e:
            print(f"Warning: Invalid voice ID format {voice_id}: {str(e)}")
            return "default"
