"""Podcast configuration models"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict

class VoiceParameters(BaseModel):
    """Voice synthesis parameters"""
    pace: float = Field(default=1.0, ge=0.5, le=2.0)
    pitch: Optional[float] = Field(default=1.0, ge=0.1, le=2.0)
    energy: float = Field(default=1.0, ge=0.1, le=1.0)
    emotion: str = Field(default="neutral")
    variation: float = Field(default=0.1, ge=0.0, le=1.0)
    
    model_config = ConfigDict(extra='allow')

class VoiceProfile(BaseModel):
    """Voice profile configuration"""
    model: str
    reference_audio: str
    reference_text: str
    style_tags: List[str]
    voice_parameters: VoiceParameters
    
    model_config = ConfigDict(extra='allow')

class CharacterBackground(BaseModel):
    """Speaker character background"""
    profession: str
    expertise: List[str]
    personality: List[str]
    speaking_style: str
    
    model_config = ConfigDict(extra='allow')

class Speaker(BaseModel):
    """Speaker configuration"""
    name: str
    character_background: CharacterBackground
    voice_profiles: Dict[str, VoiceProfile]
    
    model_config = ConfigDict(extra='allow')

class Role(BaseModel):
    """Role in podcast format"""
    voice: str  # Format: "speaker_name.profile_name"
    objective: str
    style: str
    
    model_config = ConfigDict(extra='allow')

class SegmentTemplate(BaseModel):
    """Template for podcast segment"""
    template: str
    
    model_config = ConfigDict(extra='allow')

class PodcastStructure(BaseModel):
    """Podcast format structure"""
    introduction: SegmentTemplate
    main_discussion: Dict[str, List[str]]
    conclusion: SegmentTemplate
    
    model_config = ConfigDict(extra='allow')

class AudienceLevel(BaseModel):
    """Target audience configuration"""
    name: str
    description: str
    technical_depth: int = Field(ge=1, le=5)
    assumed_knowledge: List[str]
    
    model_config = ConfigDict(extra='allow')

class ExpertiseLevel(BaseModel):
    """Expertise level configuration"""
    name: str
    description: str
    complexity: int = Field(ge=1, le=5)
    focus_areas: List[str]
    
    model_config = ConfigDict(extra='allow')

class PodcastFormat(BaseModel):
    """Podcast format configuration"""
    name: str
    description: str
    roles: Dict[str, Role]
    structure: PodcastStructure
    target_audiences: List[AudienceLevel]
    expertise_levels: List[ExpertiseLevel]
    
    model_config = ConfigDict(extra='allow')

class VoicesConfig(BaseModel):
    """Voice configurations"""
    professional: Dict[str, Speaker]
    casual: Dict[str, Speaker]
    
    model_config = ConfigDict(extra='allow')

class PromptsConfig(BaseModel):
    """Prompt configurations"""
    interview: Dict[str, PodcastFormat]
    
    model_config = ConfigDict(extra='allow')

class SpeakersConfig(BaseModel):
    """Root configuration"""
    voices: VoicesConfig
    prompts: PromptsConfig
    
    model_config = ConfigDict(extra='allow')
