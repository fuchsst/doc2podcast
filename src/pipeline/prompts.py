"""Prompt templates for analysis tasks"""

from typing import Dict, Any

class PromptTemplates:
    """Collection of prompt templates"""
    
    @staticmethod
    def get_topic_analysis_prompt(context: Dict[str, Any]) -> str:
        """Get topic analysis prompt"""
        return f"""
        Analyze these topic analysis results and provide a comprehensive topic structure.
        Focus on the actual content and themes present in the document, adapting the analysis to the document type (technical, tutorial, research, etc).
        Return the analysis in this exact JSON format:
        {{
            "main_topics": [
                {{
                    "name": "topic name",
                    "description": "detailed topic description",
                    "significance": "topic significance in the document",
                    "subtopics": [
                        {{
                            "name": "subtopic name",
                            "description": "subtopic description",
                            "parent_relationship": "how it relates to main topic"
                        }}
                    ],
                    "key_terms": [
                        {{
                            "term": "term text",
                            "context": "how the term is used",
                            "importance": "term significance"
                        }}
                    ],
                    "evidence": [
                        {{
                            "type": "term frequency|entity|topic weight",
                            "description": "evidence description"
                        }}
                    ]
                }}
            ],
            "topic_relationships": [
                {{
                    "source": "source topic",
                    "target": "target topic",
                    "relationship": "relationship description",
                    "strength": 0.0-1.0,
                    "evidence": "evidence for relationship"
                }}
            ],
            "topic_hierarchy": {{
                "levels": [
                    {{
                        "level": 1-n,
                        "topics": ["topic1", "topic2"],
                        "description": "level description"
                    }}
                ],
                "cross_level_relationships": [
                    {{
                        "from_level": 1-n,
                        "to_level": 1-n,
                        "relationship": "relationship description"
                    }}
                ]
            }}
        }}
        
        Context: {context}
        """
    
    @staticmethod
    def get_insight_analysis_prompt(context: Dict[str, Any]) -> str:
        """Get insight analysis prompt"""
        return f"""
        Analyze the document's key insights and provide a comprehensive analysis.
        Adapt the analysis to the document type (technical guide, tutorial, research paper, etc).
        For technical documents or tutorials, focus on technical approaches, implementation details, and practical insights.
        For research papers, focus on methodologies and findings.
        Return the analysis in this exact JSON format:
        {{
            "approaches": {{
                "main_approaches": [
                    {{
                        "name": "approach name",
                        "description": "detailed description",
                        "context": "usage context",
                        "advantages": ["advantage1", "advantage2"],
                        "considerations": ["consideration1", "consideration2"]
                    }}
                ],
                "implementation": [
                    {{
                        "technique": "implementation technique",
                        "description": "detailed description",
                        "key_points": ["point1", "point2"],
                        "practical_tips": ["tip1", "tip2"]
                    }}
                ],
                "framework": {{
                    "description": "overall framework or approach",
                    "components": ["component1", "component2"],
                    "integration": "how components work together"
                }}
            }},
            "key_points": [
                {{
                    "description": "point description",
                    "support": {{
                        "type": "example|explanation|demonstration",
                        "details": "supporting details"
                    }},
                    "applications": {{
                        "immediate": ["application1", "application2"],
                        "extended": ["application1", "application2"]
                    }},
                    "importance": "significance level",
                    "related_points": ["related1", "related2"]
                }}
            ],
            "notable_elements": [
                {{
                    "description": "element description",
                    "uniqueness": {{
                        "aspect": "what's notable",
                        "context": "why it matters"
                    }},
                    "applications": {{
                        "current": ["application1", "application2"],
                        "potential": ["potential1", "potential2"]
                    }},
                    "examples": ["example1", "example2"]
                }}
            ],
            "synthesis": {{
                "main_themes": ["theme1", "theme2"],
                "connections": [
                    {{
                        "elements": ["element1", "element2"],
                        "relationship": "how they relate",
                        "significance": "why it matters"
                    }}
                ],
                "overall_assessment": "comprehensive assessment"
            }}
        }}
        
        Context: {context}
        """
    
    @staticmethod
    def get_question_analysis_prompt(context: Dict[str, Any]) -> str:
        """Get question analysis prompt"""
        return f"""
        Analyze the document's key questions, objectives, and learning points.
        Adapt the analysis to the document type (technical guide, tutorial, research paper, etc).
        For technical documents or tutorials, focus on learning objectives, implementation steps, and practical considerations.
        For research papers, focus on research questions and hypotheses.
        Return the analysis in this exact JSON format:
        {{
            "main_objectives": {{
                "primary": [
                    {{
                        "objective": "objective text",
                        "type": "learning|implementation|understanding",
                        "scope": "objective scope",
                        "importance": "why it matters",
                        "related_concepts": ["concept1", "concept2"],
                        "prerequisites": ["prerequisite1", "prerequisite2"]
                    }}
                ],
                "secondary": [
                    {{
                        "objective": "objective text",
                        "relationship": "relation to primary objective",
                        "contribution": "how it supports primary",
                        "dependencies": ["dependency1", "dependency2"]
                    }}
                ]
            }},
            "key_considerations": [
                {{
                    "point": "consideration point",
                    "related_objective": "linked objective",
                    "factors": {{
                        "technical": ["factor1", "factor2"],
                        "practical": ["factor1", "factor2"],
                        "optional": ["factor1", "factor2"]
                    }},
                    "implementation": {{
                        "approach": "implementation approach",
                        "challenges": ["challenge1", "challenge2"]
                    }}
                }}
            ],
            "learning_goals": [
                {{
                    "goal": "goal statement",
                    "type": "conceptual|practical|technical",
                    "related_objectives": ["objective1", "objective2"],
                    "success_criteria": ["criterion1", "criterion2"]
                }}
            ],
            "next_steps": [
                {{
                    "step": "next step",
                    "rationale": "why important",
                    "prerequisites": ["prerequisite1", "prerequisite2"],
                    "outcomes": {{
                        "technical": ["outcome1", "outcome2"],
                        "practical": ["outcome1", "outcome2"]
                    }}
                }}
            ],
            "content_structure": {{
                "organization": {{
                    "levels": ["level1", "level2"],
                    "connections": ["connection1", "connection2"]
                }},
                "flow": {{
                    "sequence": ["step1", "step2"],
                    "dependencies": ["dependency1", "dependency2"]
                }},
                "concepts": {{
                    "core": ["concept1", "concept2"],
                    "supporting": ["concept1", "concept2"]
                }}
            }}
        }}
        
        Context: {context}
        """
    
    @staticmethod
    def get_consolidation_prompt(context: Dict[str, Any]) -> str:
        """Get consolidation prompt"""
        return f"""
        Consolidate these analysis results into a comprehensive document analysis.
        Adapt the consolidation to the document type (technical guide, tutorial, research paper, etc).
        Return results in this exact JSON format:
        {{
            "document_summary": {{
                "title": "document title",
                "type": "document type",
                "overview": "brief overview",
                "key_points": ["point1", "point2"]
            }},
            "structure": {{
                "sections": [
                    {{
                        "title": "section title",
                        "type": "section type",
                        "content_summary": "section summary",
                        "subsections": ["subsection1", "subsection2"]
                    }}
                ],
                "flow": [
                    {{
                        "from": "section1",
                        "to": "section2",
                        "relationship": "how they connect"
                    }}
                ]
            }},
            "topic_hierarchy": {{
                "main_topics": [
                    {{
                        "name": "topic name",
                        "description": "topic description",
                        "subtopics": ["subtopic1", "subtopic2"],
                        "importance": 0.0-1.0
                    }}
                ],
                "relationships": [
                    {{
                        "source": "topic1",
                        "target": "topic2",
                        "type": "relationship type"
                    }}
                ]
            }},
            "key_insights": {{
                "approaches": [
                    {{
                        "name": "approach name",
                        "description": "detailed description",
                        "context": "usage context"
                    }}
                ],
                "main_points": [
                    {{
                        "description": "point description",
                        "support": "supporting details",
                        "significance": "importance"
                    }}
                ],
                "notable_elements": [
                    {{
                        "description": "element description",
                        "uniqueness": "what's notable",
                        "applications": "potential applications"
                    }}
                ]
            }},
            "learning_landscape": {{
                "objectives": {{
                    "primary": ["objective1", "objective2"],
                    "secondary": ["objective3", "objective4"]
                }},
                "considerations": [
                    {{
                        "point": "consideration text",
                        "context": "supporting context"
                    }}
                ],
                "next_steps": [
                    {{
                        "area": "focus area",
                        "steps": ["step1", "step2"],
                        "outcome": "expected result"
                    }}
                ]
            }}
        }}
        
        Context: {context}
        """
