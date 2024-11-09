"""Script generation module using CrewAI agents"""
import datetime
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from crewai import Agent, Task, Crew, Process

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
from .script_generation_tools import (
    ContentStrategyTool,
    ScriptWritingTool,
    VoiceOptimizationTool,
    QualityControlTool,
    parse_json_safely
)
from .schemas import (
    ConsolidatedScriptSchema,
    ContentStrategySchema,
    ScriptSchema,
    OptimizedScriptSchema,
    QualityReviewSchema
)

@dataclass
class ScriptGenerationConfig:
    """Configuration for script generation"""
    podcast_preset: str
    target_audience: str
    expertise_level: str
    guidance_prompt: Optional[str] = None

class TaskFactory:
    """Factory for creating script generation tasks"""
    
    @staticmethod
    def create_strategy_task(context: ScriptContext, agent: Agent) -> Task:
        """Create content strategy task"""
        return Task(
            description=f"""
            Create a podcast content strategy using:
            Content: {json.dumps(context.content)}
            Format: {json.dumps(context.format)}
            Audience: {json.dumps(context.audience)}
            Expertise: {json.dumps(context.expertise)}
            Guidance: {context.guidance or "None provided"}
            
            IMPORTANT: Return a valid JSON object with the exact structure specified in the tool description.
            """,
            expected_output="""JSON object containing content strategy with proper structure""",
            agent=agent
        )
        
    @staticmethod
    def create_writing_task(context: ScriptContext, agent: Agent, strategy_result: Dict[str, Any]) -> Task:
        """Create script writing task"""
        return Task(
            description=f"""
            Create a podcast script following:
            Strategy: {json.dumps(strategy_result)}
            Format: {json.dumps(context.format)}
            Audience: {json.dumps(context.audience)}
            Expertise: {json.dumps(context.expertise)}
            
            IMPORTANT: Return a valid JSON object with the exact structure specified in the tool description.
            """,
            expected_output="""JSON object containing script with proper structure""",
            agent=agent
        )
        
    @staticmethod
    def create_voice_task(context: ScriptContext, agent: Agent, script_result: Dict[str, Any]) -> Task:
        """Create voice optimization task"""
        return Task(
            description=f"""
            Optimize this script for voice synthesis:
            Script: {json.dumps(script_result)}
            Format: {json.dumps(context.format)}
            
            IMPORTANT: Return a valid JSON object with the exact structure specified in the tool description.
            """,
            expected_output="""JSON object containing voice optimization with proper structure""",
            agent=agent
        )
        
    @staticmethod
    def create_quality_task(context: ScriptContext, agent: Agent, optimized_script: Dict[str, Any]) -> Task:
        """Create quality control task"""
        return Task(
            description=f"""
            Review and improve this script:
            Script: {json.dumps(optimized_script)}
            Audience: {json.dumps(context.audience)}
            Expertise: {json.dumps(context.expertise)}
            
            IMPORTANT: Return a valid JSON object with the exact structure specified in the tool description.
            """,
            expected_output="""JSON object containing quality review with proper structure""",
            agent=agent
        )

class PodcastScriptGenerator(ScriptGenerator, ResultsFormatter):
    """Generates podcast scripts using CrewAI agents"""
    
    def __init__(self, settings: Settings, callback: Optional[PipelineCallback] = None):
        super().__init__(settings, callback)
        self.prompt_manager = PromptManager(settings)
        
        # Initialize LLM
        self.llm = settings.get_llm()
        
        # Initialize tools
        self.content_tool = ContentStrategyTool(self.llm)
        self.script_tool = ScriptWritingTool(self.llm)
        self.voice_tool = VoiceOptimizationTool(self.llm)
        self.quality_tool = QualityControlTool(self.llm)
        
        # Initialize agents
        self.initialize_agents()
        
    def generate(self, content: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        """Implement abstract generate method from Generator base class"""
        config = kwargs.get('config', ScriptGenerationConfig(
            podcast_preset=kwargs.get('podcast_preset', 'default'),
            target_audience=kwargs.get('target_audience', 'general'),
            expertise_level=kwargs.get('expertise_level', 'beginner'),
            guidance_prompt=kwargs.get('guidance_prompt')
        ))
        return self.generate_script(content, config)
        
    def initialize_agents(self):
        """Initialize CrewAI agents"""
        
        # Content Strategist Agent
        content_strategist = self.prompt_manager.get_agent_prompt("content_strategist")
        self.strategist = Agent(
            role=content_strategist["role"],
            goal=content_strategist["goal"],
            backstory=content_strategist["backstory"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.content_tool]
        )
        
        # Script Writer Agent
        script_writer = self.prompt_manager.get_agent_prompt("script_writer")
        self.writer = Agent(
            role=script_writer["role"],
            goal=script_writer["goal"],
            backstory=script_writer["backstory"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.script_tool]
        )
        
        # Voice Style Optimizer Agent
        voice_optimizer = self.prompt_manager.get_agent_prompt("voice_optimizer")
        self.optimizer = Agent(
            role=voice_optimizer["role"],
            goal=voice_optimizer["goal"],
            backstory=voice_optimizer["backstory"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.voice_tool]
        )
        
        # Quality Control Agent
        quality_checker = self.prompt_manager.get_agent_prompt("quality_checker")
        self.quality_checker = Agent(
            role=quality_checker["role"],
            goal=quality_checker["goal"],
            backstory=quality_checker["backstory"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.quality_tool]
        )
        
    def format_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format and validate generation results"""
        try:
            # Validate individual components
            strategy_data = ContentStrategySchema(**results[0])
            script_data = ScriptSchema(**results[1])
            voice_data = OptimizedScriptSchema(**results[2])
            quality_data = QualityReviewSchema(**results[3])
            
            # Create consolidated results
            consolidated = ConsolidatedScriptSchema(
                content_strategy=strategy_data,
                initial_script=script_data,
                optimized_script=voice_data,
                quality_review=quality_data,
                metadata={
                    "version": "1.0",
                    "generated_at": datetime.datetime.now().isoformat()
                }
            )
            
            return consolidated
            
        except Exception as e:
            raise ValueError(f"Failed to format results: {str(e)}")
        
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
            
            # Execute tasks sequentially with progress tracking
            results = []

            # Initialize substeps
            substeps = [
                {"name": "Content Strategy", "status": "pending"},
                {"name": "Script Writing", "status": "pending"},
                {"name": "Voice Optimization", "status": "pending"},
                {"name": "Quality Review", "status": "pending"}
            ]
            
            # Step 1: Content Strategy
            if self.callback:
                substeps[0]["status"] = "in_progress"
                self.callback.on_script_generation(
                    progress=0,
                    message="Generating content strategy...",
                    substeps=substeps
                )
                
            strategy_task = TaskFactory.create_strategy_task(context, self.strategist)
            crew = Crew(
                agents=[self.strategist],
                tasks=[strategy_task],
                verbose=True,
                process=Process.sequential
            )
            strategy_output = crew.kickoff()
            
            try:
                strategy_result = parse_json_safely(str(strategy_output))
                results.append(strategy_result)
            except Exception as e:
                raise RuntimeError(f"Failed to parse strategy output: {str(e)}\nOutput was: {str(strategy_output)}")
            
            if self.callback:
                substeps[0]["status"] = "completed"
                self.callback.on_script_generation(
                    progress=25,
                    message="Content strategy generated",
                    substeps=substeps
                )
            
            # Step 2: Script Writing
            if self.callback:
                substeps[1]["status"] = "in_progress"
                self.callback.on_script_generation(
                    progress=25,
                    message="Writing script...",
                    substeps=substeps
                )
                
            writing_task = TaskFactory.create_writing_task(context, self.writer, strategy_result)
            crew = Crew(
                agents=[self.writer],
                tasks=[writing_task],
                verbose=True,
                process=Process.sequential
            )
            script_output = crew.kickoff()
            
            try:
                script_result = parse_json_safely(str(script_output))
                results.append(script_result)
            except Exception as e:
                raise RuntimeError(f"Failed to parse script output: {str(e)}\nOutput was: {str(script_output)}")
            
            if self.callback:
                substeps[1]["status"] = "completed"
                self.callback.on_script_generation(
                    progress=50,
                    message="Script written",
                    substeps=substeps
                )
            
            # Step 3: Voice Optimization
            if self.callback:
                substeps[2]["status"] = "in_progress"
                self.callback.on_script_generation(
                    progress=50,
                    message="Optimizing for voice...",
                    substeps=substeps
                )
                
            voice_task = TaskFactory.create_voice_task(context, self.optimizer, script_result)
            crew = Crew(
                agents=[self.optimizer],
                tasks=[voice_task],
                verbose=True,
                process=Process.sequential
            )
            voice_output = crew.kickoff()
            
            try:
                voice_result = parse_json_safely(str(voice_output))
                results.append(voice_result)
            except Exception as e:
                raise RuntimeError(f"Failed to parse voice output: {str(e)}\nOutput was: {str(voice_output)}")
            
            if self.callback:
                substeps[2]["status"] = "completed"
                self.callback.on_script_generation(
                    progress=75,
                    message="Voice optimization complete",
                    substeps=substeps
                )
            
            # Step 4: Quality Review
            if self.callback:
                substeps[3]["status"] = "in_progress"
                self.callback.on_script_generation(
                    progress=75,
                    message="Performing quality review...",
                    substeps=substeps
                )
                
            quality_task = TaskFactory.create_quality_task(context, self.quality_checker, voice_result)
            crew = Crew(
                agents=[self.quality_checker],
                tasks=[quality_task],
                verbose=True,
                process=Process.sequential
            )
            quality_output = crew.kickoff()
            
            try:
                quality_result = parse_json_safely(str(quality_output))
                results.append(quality_result)
            except Exception as e:
                raise RuntimeError(f"Failed to parse quality output: {str(e)}\nOutput was: {str(quality_output)}")
            
            if self.callback:
                substeps[3]["status"] = "completed"
                self.callback.on_script_generation(
                    progress=100,
                    message="Quality review complete",
                    substeps=substeps
                )
            
            # Format and validate results
            try:
                consolidated = self.format_results(results)
                
                # Get final script from quality review
                final_script = consolidated["quality_review"]["final_script"]
                voice_guidance = consolidated["optimized_script"]["voice_guidance"]

                # Get voice configurations from roles
                roles = format_config.roles
                
                # Convert to PodcastScript object
                script = PodcastScript(
                    metadata=PodcastMetadata(
                        title=final_script.get("metadata", {}).get("title", "Untitled Podcast"),
                        description=final_script.get("metadata", {}).get("description"),
                        source_document=content.get("source", {}).get("path"),
                        tags=final_script.get("metadata", {}).get("tags", []),
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
                        for segment in final_script["segments"]
                    ],
                    settings={
                        "format": config.podcast_preset,
                        "target_audience": config.target_audience,
                        "expertise_level": config.expertise_level,
                        "voice_guidance": voice_guidance
                    }
                )
                
                # Cache the complete script
                cache_manager.cache_json(cache_key, "complete_script", script.model_dump())
                
                if self.callback:
                    self.callback.on_step_complete(StepType.SCRIPT_GENERATION, "Script generation completed successfully")
                
                return script
                
            except Exception as e:
                error = f"Failed to parse results: {str(e)}"
                if self.callback:
                    self.callback.on_error(StepType.SCRIPT_GENERATION, error)
                raise RuntimeError(error)
                
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
