"""JSON schemas for script generation"""

from typing import Dict, Any, List, TypedDict, Optional

class ContentStrategySchema(TypedDict):
    """Schema for content strategy"""
    episode_structure: Dict[str, Any]
    key_points: List[Dict[str, str]]
    transitions: List[Dict[str, str]]
    audience_adaptation: Dict[str, Any]
    technical_depth: Dict[str, Any]

class ScriptSegmentSchema(TypedDict):
    """Schema for script segment"""
    speaker: str
    text: str
    style: str
    transitions: Dict[str, str]
    technical_terms: List[Dict[str, str]]

class ScriptSchema(TypedDict):
    """Schema for complete script"""
    segments: List[ScriptSegmentSchema]
    speakers: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    settings: Dict[str, Any]

class VoiceGuidanceSchema(TypedDict):
    """Schema for voice guidance"""
    pronunciation: Dict[str, str]
    emphasis: List[Dict[str, Any]]
    pacing: Dict[str, float]
    emotions: Dict[str, str]

class OptimizedScriptSchema(TypedDict):
    """Schema for voice-optimized script"""
    segments: List[ScriptSegmentSchema]
    voice_guidance: VoiceGuidanceSchema
    timing: Dict[str, Any]

class QualityMetricsSchema(TypedDict):
    """Schema for quality metrics"""
    content_accuracy: float
    conversation_flow: float
    audience_fit: float
    technical_accuracy: float
    engagement: float

class QualityReviewSchema(TypedDict):
    """Schema for quality review"""
    final_script: ScriptSchema
    quality_metrics: QualityMetricsSchema
    improvements: List[Dict[str, str]]
    recommendations: Dict[str, Any]

class ConsolidatedScriptSchema(TypedDict):
    """Schema for consolidated script results"""
    content_strategy: ContentStrategySchema
    initial_script: ScriptSchema
    optimized_script: OptimizedScriptSchema
    quality_review: QualityReviewSchema
    metadata: Dict[str, Any]
