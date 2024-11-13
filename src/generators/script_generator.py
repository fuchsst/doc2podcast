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
            
            # Step 3: Review script quality
            reviewed = self.review_script_quality(script, config)
            
            # Format results
            results = [strategy, script, reviewed]
            
            # Convert to PodcastScript object with neutral placeholder speakers
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
                            voice_model=None,  # Will be set during voice configuration
                            voice_preset=None,  # Will be set during voice configuration
                            style_tags=[],  # Will be set during voice configuration
                            voice_parameters=VoiceParameters(
                                pace=1.0,
                                pitch=1.0,
                                energy=0.5,
                                emotion="neutral",
                                variation=0.5
                            ),
                            reference=None
                        ),
                        text=segment["text"],
                        duration=None,  # Will be set during audio generation
                        audio_path=None  # Will be set during audio generation
                    )
                    for segment in script["segments"]
                ],
                settings={
                    "format": config.podcast_preset,
                    "target_audience": config.target_audience,
                    "expertise_level": config.expertise_level,
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
