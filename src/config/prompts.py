"""Prompt management and configuration"""
from typing import Dict, Any, List, Optional
from pathlib import Path

from .settings import Settings
from .podcast_config import (
    SpeakersConfig,
    PodcastFormat,
    AudienceLevel,
    ExpertiseLevel,
    Role,
    PodcastStructure,
    VoiceProfile,
    CharacterBackground
)

class PromptManager:
    """Manages prompts and templates for the podcast generation pipeline"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._speakers_config = None
        self._podcast_formats = None
        
    @property
    def speakers_config(self) -> Dict[str, Any]:
        """Get speakers configuration"""
        if self._speakers_config is None:
            self._speakers_config = self.settings.speakers_config
        return self._speakers_config
    
    def get_agent_prompt(self, agent_name: str) -> Dict[str, Any]:
        """Get agent configuration prompt"""
        return self.settings.get_agent(agent_name)
    
    def get_task_prompt(self, task_name: str) -> Dict[str, Any]:
        """Get task configuration prompt"""
        return self.settings.get_task(task_name)
    
    def get_voice_profile(
        self,
        category: str,
        voice_name: str,
        profile_type: str
    ) -> Dict[str, Any]:
        """Get voice profile configuration"""
        voice = self.settings.get_voice(category, voice_name)
        if profile_type not in voice.get("voice_profiles", {}):
            raise ValueError(f"Unknown profile type: {profile_type}")
            
        profile_data = voice["voice_profiles"][profile_type]
        character_data = voice["character_background"]
        
        try:
            # Create validated objects
            profile = VoiceProfile(**profile_data)
            background = CharacterBackground(**character_data)
            
            return {
                "name": voice["name"],
                "character_background": background.model_dump(),
                "voice_profile": profile.model_dump()
            }
        except Exception as e:
            raise ValueError(f"Invalid voice profile configuration: {str(e)}")
    
    def get_podcast_presets(self) -> Dict[str, PodcastFormat]:
        """Get available podcast format presets
        
        Returns:
            Dict[str, PodcastFormat]: Dictionary of preset name to PodcastFormat
        """
        if self._podcast_formats is None:
            self._podcast_formats = {}
            for preset_name, preset_data in self.speakers_config.get("prompts", {}).get("interview", {}).items():
                try:
                    self._podcast_formats[preset_name] = PodcastFormat(**preset_data)
                except Exception as e:
                    print(f"Error loading preset {preset_name}: {str(e)}")
                    continue
                    
        return self._podcast_formats
    
    def get_interview_prompt(self, format_type: str) -> PodcastFormat:
        """Get interview prompt configuration
        
        Args:
            format_type (str): Name of the podcast format
            
        Returns:
            PodcastFormat: Validated podcast format configuration
        """
        presets = self.get_podcast_presets()
        if format_type not in presets:
            raise ValueError(f"Unknown podcast format: {format_type}")
        return presets[format_type]
    
    def get_target_audiences(self, preset_name: str) -> List[AudienceLevel]:
        """Get target audiences for a preset
        
        Args:
            preset_name (str): Name of the podcast preset
            
        Returns:
            List[AudienceLevel]: List of available target audiences
        """
        format_config = self.get_interview_prompt(preset_name)
        return format_config.target_audiences
    
    def get_expertise_levels(self, preset_name: str) -> List[ExpertiseLevel]:
        """Get expertise levels for a preset
        
        Args:
            preset_name (str): Name of the podcast preset
            
        Returns:
            List[ExpertiseLevel]: List of available expertise levels
        """
        format_config = self.get_interview_prompt(preset_name)
        return format_config.expertise_levels
    
    def get_format_roles(self, preset_name: str) -> Dict[str, Role]:
        """Get roles for a podcast format
        
        Args:
            preset_name (str): Name of the podcast preset
            
        Returns:
            Dict[str, Role]: Dictionary of role name to Role configuration
        """
        format_config = self.get_interview_prompt(preset_name)
        return format_config.roles
    
    def get_format_structure(self, preset_name: str) -> PodcastStructure:
        """Get structure for a podcast format
        
        Args:
            preset_name (str): Name of the podcast preset
            
        Returns:
            PodcastStructure: Podcast format structure configuration
        """
        format_config = self.get_interview_prompt(preset_name)
        return format_config.structure
    
    def format_system_prompt(self, agent_name: str, task_name: str) -> str:
        """Format system prompt for an agent and task"""
        agent = self.get_agent_prompt(agent_name)
        task = self.get_task_prompt(task_name)
        
        return f"""You are {agent['name']}, {agent['role']}.
Your goal is to {agent['goal']}.
Background: {agent['backstory']}

Your current task:
{task['description']}

Expected output:
{task['expected_output']}

Please follow these validation requirements:
Required fields: {', '.join(task['validation']['required_fields'])}
Output format: {task['validation']['output_format']}
"""
    
    def format_interview_prompt(
        self,
        format_type: str,
        target_audience: str,
        expertise_level: str,
        guidance: Optional[str] = None
    ) -> str:
        """Format interview prompt template
        
        Args:
            format_type (str): Name of the podcast format
            target_audience (str): Target audience name
            expertise_level (str): Expertise level name
            guidance (Optional[str]): Additional guidance prompt
            
        Returns:
            str: Formatted interview prompt
        """
        format_config = self.get_interview_prompt(format_type)
        
        # Get audience and expertise details
        audience = next((a for a in format_config.target_audiences if a.name == target_audience), None)
        if not audience:
            raise ValueError(f"Unknown target audience: {target_audience}")
            
        level = next((l for l in format_config.expertise_levels if l.name == expertise_level), None)
        if not level:
            raise ValueError(f"Unknown expertise level: {expertise_level}")
        
        prompt = f"""You are a professional podcast scriptwriter.

Format: {format_config.name}
Description: {format_config.description}

Target Audience:
- {audience.name}: {audience.description}
- Technical Depth: {audience.technical_depth}
- Assumed Knowledge: {', '.join(audience.assumed_knowledge)}

Expertise Level:
- {level.name}: {level.description}
- Complexity: {level.complexity}
- Focus Areas: {', '.join(level.focus_areas)}

Roles:
"""
        
        # Add roles
        for role_name, role in format_config.roles.items():
            prompt += f"""
{role_name} ({role.voice}):
- Objective: {role.objective}
- Style: {role.style}
"""
        
        # Add structure
        prompt += f"""
Structure:
1. Introduction:
{format_config.structure.introduction.template}

2. Main Discussion:
Segments: {', '.join(format_config.structure.main_discussion['segments'])}

3. Conclusion:
{format_config.structure.conclusion.template}
"""
        
        # Add guidance if provided
        if guidance:
            prompt += f"""
Additional Guidance:
{guidance}
"""
        
        return prompt
