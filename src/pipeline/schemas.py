"""JSON schemas for analysis results"""

from typing import Dict, Any, List, TypedDict, Optional

class TopicSchema(TypedDict):
    name: str
    description: str
    significance: str
    subtopics: List[Dict[str, str]]
    key_terms: List[Dict[str, Any]]
    evidence: List[Dict[str, str]]

class RelationshipSchema(TypedDict):
    source: str
    target: str
    relationship: str
    strength: float
    evidence: str

class HierarchyLevelSchema(TypedDict):
    level: int
    topics: List[str]
    description: str

class TopicAnalysisSchema(TypedDict):
    main_topics: List[TopicSchema]
    topic_relationships: List[RelationshipSchema]
    topic_hierarchy: Dict[str, List[HierarchyLevelSchema]]

class ApproachSchema(TypedDict):
    name: str
    description: str
    context: str
    advantages: List[str]
    considerations: List[str]

class KeyPointSchema(TypedDict):
    description: str
    support: Dict[str, str]
    applications: Dict[str, List[str]]
    importance: str
    related_points: List[str]

class NotableElementSchema(TypedDict):
    description: str
    uniqueness: Dict[str, str]
    applications: Dict[str, List[str]]
    examples: List[str]

class InsightAnalysisSchema(TypedDict):
    approaches: Dict[str, List[ApproachSchema]]
    key_points: List[KeyPointSchema]
    notable_elements: List[NotableElementSchema]
    synthesis: Dict[str, Any]

class ObjectiveSchema(TypedDict):
    objective: str
    type: str
    scope: str
    importance: str
    related_concepts: List[str]
    prerequisites: List[str]

class ConsiderationSchema(TypedDict):
    point: str
    related_objective: str
    factors: Dict[str, List[str]]
    implementation: Dict[str, Any]

class LearningGoalSchema(TypedDict):
    goal: str
    type: str
    related_objectives: List[str]
    success_criteria: List[str]

class QuestionAnalysisSchema(TypedDict):
    main_objectives: Dict[str, List[ObjectiveSchema]]
    key_considerations: List[ConsiderationSchema]
    learning_goals: List[LearningGoalSchema]
    next_steps: List[Dict[str, Any]]
    content_structure: Dict[str, Any]

class ConsolidatedResultsSchema(TypedDict):
    document_summary: Dict[str, Any]
    structure: Dict[str, List[Dict[str, Any]]]
    topic_hierarchy: Dict[str, Any]
    key_insights: Dict[str, Any]
    learning_landscape: Dict[str, Any]
