```python
# doc2podcast/config/models.py
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from enum import Enum

class SpeakerStyle(str, Enum):
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    ENERGETIC = "energetic"
    ACADEMIC = "academic"
    FRIENDLY = "friendly"

class SpeakerRole(str, Enum):
    HOST = "host"
    GUEST = "guest"
    EXPERT = "expert"
    INTERVIEWER = "interviewer"

class VoiceConfig(BaseModel):
    voice_id: str
    name: str
    gender: str
    default_style: SpeakerStyle
    supported_styles: List[SpeakerStyle]
    pitch_range: tuple[float, float] = (-20.0, 20.0)
    speed_range: tuple[float, float] = (0.5, 2.0)

class SpeakerConfig(BaseModel):
    role: SpeakerRole
    default_voice: str
    available_voices: List[str]
    default_style: SpeakerStyle
    available_styles: List[SpeakerStyle]

class ProjectSettings(BaseModel):
    target_audiences: List[str]
    topic_focuses: List[str]
    content_styles: List[str]
    max_duration: int = Field(default=3600, description="Maximum duration in seconds")
    chunk_size: int = Field(default=1000, description="Text chunk size for processing")
```
