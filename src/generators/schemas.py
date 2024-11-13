"""Schemas for script generation results"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class OutlineSegment(BaseModel):
    """Schema for outline segment"""
    title: str
    description: str

class OutlineSchema(BaseModel):
    """Schema for outline structure"""
    introduction: str
    main_segments: List[OutlineSegment]
    conclusion: str

class ContentStrategySchema(BaseModel):
    """Schema for content strategy results"""
    outline: OutlineSchema
    key_points: List[str]
    transitions: List[str]
    audience_adaptations: Dict[str, Any]
    metadata: Dict[str, str]

class TechnicalTerm(BaseModel):
    """Schema for technical terms"""
    term: str
    definition: str

class Transitions(BaseModel):
    """Schema for segment transitions"""
    next: str
    prev: str

class ScriptSegment(BaseModel):
    """Schema for script segment"""
    speaker: str
    text: str
    style: str
    transitions: Transitions
    technical_terms: List[TechnicalTerm]

class Speaker(BaseModel):
    """Schema for speaker information"""
    name: str
    role: str

class ScriptMetadata(BaseModel):
    """Schema for script metadata"""
    title: str
    description: str
    tags: List[str]

class ScriptSettings(BaseModel):
    """Schema for script settings"""
    format: str
    style: str

class ScriptSchema(BaseModel):
    """Schema for script results"""
    segments: List[ScriptSegment]
    speakers: List[Speaker]
    metadata: ScriptMetadata
    settings: ScriptSettings

class VoiceParameters(BaseModel):
    """Schema for voice parameters"""
    pace: float
    energy: float
    variation: float
    pitch: float
    emotion: str

class VoiceProfile(BaseModel):
    """Schema for voice profile"""
    model: str
    voice_preset: Optional[str]
    style_tags: List[str]
    voice_parameters: VoiceParameters
    reference_audio: Optional[str]
    reference_text: Optional[str]

class SpeakerConfig(BaseModel):
    """Schema for speaker configuration"""
    name: str
    voice_profile: VoiceProfile
    character_background: Dict[str, Any]

class VoiceSegment(BaseModel):
    """Schema for voice-optimized segment"""
    speaker: SpeakerConfig
    text: str
    style: str
    voice_parameters: VoiceParameters
    emphasis_words: List[str]
    pauses: List[Dict[str, Any]]
    pronunciation_guide: Dict[str, str]

class VoiceGuidance(BaseModel):
    """Schema for voice guidance"""
    pronunciation: Dict[str, str]
    emphasis: List[Dict[str, Any]]
    pacing: Dict[str, float]
    emotions: Dict[str, str]

class Timing(BaseModel):
    """Schema for timing information"""
    total_duration: float
    segment_durations: Dict[str, float]

class OptimizedScriptSchema(BaseModel):
    """Schema for voice-optimized script"""
    segments: List[VoiceSegment]
    voice_guidance: VoiceGuidance
    timing: Timing

class QualityMetrics(BaseModel):
    """Schema for quality metrics"""
    content_accuracy: float
    conversation_flow: float
    audience_fit: float
    technical_accuracy: float
    engagement: float

class Improvement(BaseModel):
    """Schema for improvement suggestions"""
    type: str
    description: str

class Recommendations(BaseModel):
    """Schema for recommendations"""
    content: List[str]
    delivery: List[str]

class QualityReviewSchema(BaseModel):
    """Schema for quality review results"""
    quality_metrics: QualityMetrics
    improvements: List[Improvement]
    recommendations: Recommendations

class VoiceOptimizedScript(BaseModel):
    """Schema for final voice-optimized script"""
    metadata: ScriptMetadata
    content_strategy: ContentStrategySchema
    segments: List[VoiceSegment]
    voice_guidance: VoiceGuidance
    quality_review: QualityReviewSchema
    settings: Dict[str, Any]

class ConsolidatedScriptSchema(BaseModel):
    """Schema for consolidated script generation results"""
    content_strategy: ContentStrategySchema
    initial_script: ScriptSchema
    optimized_script: OptimizedScriptSchema
    quality_review: QualityReviewSchema
    metadata: Dict[str, Any]
