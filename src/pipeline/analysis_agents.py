"""Analysis agents for document processing using CrewAI"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from crewai import Agent, Task, Crew, Process
import json

from .base import DocumentAnalyzer, ResultsConsolidator, AnalysisContext
from .analysis_tools import (
    TopicAnalysisTool,
    InsightAnalysisTool,
    ObjectiveAnalysisTool,
    AnalysisConfig
)
from .config import AgentConfig
from .prompts import PromptTemplates
from ..config import Settings

@dataclass
class AnalysisResult:
    """Container for analysis results"""
    title: str
    briefing: Dict[str, Any]
    keywords: List[str]
    topics: Dict[str, Any]
    key_insights: Dict[str, Any]
    objectives: Dict[str, Any]
    relationships: List[Dict[str, Any]]
    hierarchy: Dict[str, Any]

class AnalysisAgents(DocumentAnalyzer, ResultsConsolidator):
    """Manages document analysis agents and tasks"""
    
    def __init__(self, llm, config: Optional[AgentConfig] = None):
        self.llm = llm
        self.config = config or AgentConfig()
        
        # Get settings for chunk_size
        settings = Settings()
        chunk_size = settings.project_config.processing.chunk_size
        
        # Initialize tools with analysis config using chunk_size from settings
        analysis_config = AnalysisConfig(chunk_size=chunk_size)
        self.topic_tool = TopicAnalysisTool(llm, analysis_config)
        self.insight_tool = InsightAnalysisTool(llm, analysis_config)
        self.objective_tool = ObjectiveAnalysisTool(llm, analysis_config)
        
        # Initialize agents
        self.initialize_agents()
        
    def initialize_agents(self):
        """Initialize specialized analysis agents"""
        
        # Title & Briefing Agent
        self.briefing_agent = Agent(
            role="Content Summarizer",
            goal="Create engaging titles and comprehensive briefings",
            backstory="""Expert in distilling complex content into compelling summaries.
            Skilled at identifying key themes and creating clear, structured overviews.""",
            verbose=self.config.verbose,
            allow_delegation=self.config.allow_delegation,
            llm=self.llm
        )
        
        # Topic Analysis Agent
        self.topic_agent = Agent(
            role="Topic Analyst",
            goal="Extract and organize document topics",
            backstory="""Expert in identifying core topics, themes and concepts.
            Creates detailed topic hierarchies and relationship maps.
            Specializes in understanding complex topic interactions.""",
            verbose=self.config.verbose,
            allow_delegation=self.config.allow_delegation,
            llm=self.llm,
            tools=[self.topic_tool]
        )
        
        # Key Insights Agent
        self.insights_agent = Agent(
            role="Content Analyst",
            goal="Extract key insights and approaches",
            backstory="""Expert in analyzing content to identify key approaches,
            implementation details, and notable elements. Skilled at understanding
            technical details and practical applications.""",
            verbose=self.config.verbose,
            allow_delegation=self.config.allow_delegation,
            llm=self.llm,
            tools=[self.insight_tool]
        )
        
        # Objective Analysis Agent
        self.objective_agent = Agent(
            role="Learning Objectives Analyst",
            goal="Analyze objectives and learning goals",
            backstory="""Expert in identifying learning objectives, implementation steps,
            and practical considerations. Specializes in understanding educational
            goals and technical requirements.""",
            verbose=self.config.verbose,
            allow_delegation=self.config.allow_delegation,
            llm=self.llm,
            tools=[self.objective_tool]
        )
        
        # Integration Agent
        self.integration_agent = Agent(
            role="Content Integrator",
            goal="Synthesize and integrate analysis results",
            backstory="""Expert in combining and synthesizing multiple analyses
            into coherent, comprehensive summaries. Skilled at identifying
            patterns and relationships across different analyses.""",
            verbose=self.config.verbose,
            allow_delegation=self.config.allow_delegation,
            llm=self.llm
        )
        
    def analyze(self, context: AnalysisContext) -> Dict[str, Any]:
        """Analyze document content"""
        try:
            # Create tasks
            briefing_task = TaskFactory.create_briefing_task(context.text, self.briefing_agent)
            topic_task = TaskFactory.create_topic_task(context.text, self.topic_agent)
            insight_task = TaskFactory.create_insight_task(context.text, self.insights_agent)
            objective_task = TaskFactory.create_objective_task(context.text, self.objective_agent)
            
            # Create crew
            crew = Crew(
                agents=[
                    self.briefing_agent,
                    self.topic_agent,
                    self.insights_agent,
                    self.objective_agent
                ],
                tasks=[
                    briefing_task,
                    topic_task,
                    insight_task,
                    objective_task
                ],
                verbose=self.config.verbose,
                process=self.config.process
            )
            
            # Execute tasks
            results = crew.kickoff()
            
            # Parse results
            try:
                briefing_data = json.loads(str(results.tasks_output[0]))
                topics_data = json.loads(str(results.tasks_output[1]))
                insights_data = json.loads(str(results.tasks_output[2]))
                objectives_data = json.loads(str(results.tasks_output[3]))
                
                return {
                    "briefing": briefing_data,
                    "topics": topics_data,
                    "insights": insights_data,
                    "objectives": objectives_data
                }
                
            except (json.JSONDecodeError, AttributeError, IndexError) as e:
                return {
                    "error": f"Failed to parse results: {str(e)}",
                    "raw_output": str(results)
                }
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
            
    def consolidate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidate multiple analysis results"""
        try:
            # Get integration tasks
            integration_tasks = TaskFactory.create_integration_task(
                results,
                self.integration_agent
            )
            
            # Process each task and collect results
            integrated_data = {}
            for task in integration_tasks:
                # Create crew for this task
                crew = Crew(
                    agents=[self.integration_agent],
                    tasks=[task],
                    verbose=self.config.verbose,
                    process=self.config.process
                )
                
                # Execute task
                result = crew.kickoff()
                
                try:
                    # Parse result and merge into integrated data
                    task_result = json.loads(str(result))
                    integrated_data.update(task_result)
                except json.JSONDecodeError as e:
                    return {
                        "error": f"Failed to parse task result: {str(e)}",
                        "raw_output": str(result)
                    }
            
            # Extract components for AnalysisResult
            return {
                "title": integrated_data.get("document_overview", {}).get("title", ""),
                "briefing": {
                    "themes": integrated_data.get("document_overview", {}).get("main_themes", []),
                    "significance": integrated_data.get("document_overview", {}).get("significance", "")
                },
                "keywords": self._extract_keywords(integrated_data),
                "topics": integrated_data.get("topic_analysis", {}),
                "key_insights": integrated_data.get("key_insights", {}),
                "objectives": integrated_data.get("learning_landscape", {}),
                "relationships": integrated_data.get("relationships", []),
                "hierarchy": integrated_data.get("topic_analysis", {}).get("topic_hierarchy", {})
            }
                
        except Exception as e:
            return {"error": f"Consolidation failed: {str(e)}"}
            
    def _extract_keywords(self, data: Dict[str, Any]) -> List[str]:
        """Extract keywords from integrated results"""
        keywords = set()
        
        def safe_add_to_keywords(item: Any) -> None:
            """Safely add an item to the keywords set"""
            if isinstance(item, str):
                keywords.add(item)
            elif isinstance(item, (list, tuple)):
                for subitem in item:
                    if isinstance(subitem, str):
                        keywords.add(subitem)
                    elif isinstance(subitem, dict) and "name" in subitem:
                        keywords.add(str(subitem["name"]))
            elif isinstance(item, dict):
                if "name" in item:
                    keywords.add(str(item["name"]))
                if "title" in item:
                    keywords.add(str(item["title"]))
                if "description" in item:
                    # Split description into words and take first two
                    words = str(item["description"]).split()[:2]
                    keywords.update(words)
        
        # Extract from document overview
        if "document_overview" in data:
            overview = data["document_overview"]
            safe_add_to_keywords(overview.get("title"))
            safe_add_to_keywords(overview.get("main_themes", []))
        
        # Extract from topic analysis
        if "topic_analysis" in data:
            topic_analysis = data["topic_analysis"]
            if "main_topics" in topic_analysis:
                safe_add_to_keywords(topic_analysis["main_topics"])
            
            # Extract from topic hierarchy
            if "topic_hierarchy" in topic_analysis:
                hierarchy = topic_analysis["topic_hierarchy"]
                if "levels" in hierarchy:
                    for level in hierarchy["levels"]:
                        safe_add_to_keywords(level.get("topics", []))
        
        # Extract from key insights
        if "key_insights" in data:
            insights = data["key_insights"]
            if "approaches" in insights:
                approaches = insights["approaches"]
                if "main_approaches" in approaches:
                    safe_add_to_keywords(approaches["main_approaches"])
            
            if "key_points" in insights:
                safe_add_to_keywords(insights["key_points"])
        
        # Extract from learning landscape
        if "learning_landscape" in data:
            landscape = data["learning_landscape"]
            if "objectives" in landscape:
                objectives = landscape["objectives"]
                safe_add_to_keywords(objectives.get("primary", []))
                safe_add_to_keywords(objectives.get("secondary", []))
        
        # Process collected strings into proper keywords using LLM
        prompt = f"""Given this list of strings extracted from a document analysis:
        {list(keywords)}
        
        Convert these into a focused list of keywords that:
        1. Are concise and specific
        2. Capture key concepts and themes
        3. Are normalized (e.g. consistent casing, no duplicates)
        4. Remove generic or non-descriptive terms
        5. Prioritize domain-specific terminology
        
        Return only the list of keywords, one per line."""
        
        try:
            response = self.llm.predict(prompt)
            # Split response into lines and clean up
            processed_keywords = [
                keyword.strip() 
                for keyword in response.split('\n') 
                if keyword.strip()
            ]
            return processed_keywords
        except Exception as e:
            # Fallback to original keywords if LLM processing fails
            return list(keywords)

class TaskFactory:
    """Factory for creating analysis tasks"""
    
    @staticmethod
    def create_briefing_task(text: str, agent: Agent) -> Task:
        """Create briefing task"""
        return Task(
            description=f"""
            Create a comprehensive briefing for this content:
            1. Identify key themes and main points
            2. Create a structured summary
            3. Extract significant takeaways
            4. Return results in this JSON format:
            {{
                "summary": {{
                    "main_points": ["point1", "point2"],
                    "themes": ["theme1", "theme2"],
                    "context": "content context"
                }},
                "key_takeaways": [
                    {{
                        "point": "takeaway point",
                        "significance": "why it matters",
                        "evidence": "supporting evidence"
                    }}
                ],
                "relevance": {{
                    "technical": ["relevance1", "relevance2"],
                    "practical": ["relevance1", "relevance2"]
                }}
            }}
            
            Text: {text}
            """,
            expected_output="JSON with briefing content",
            agent=agent
        )

    @staticmethod
    def create_topic_task(text: str, agent: Agent) -> Task:
        """Create topic analysis task"""
        return Task(
            description=f"""
            Use the Topic Analysis tool to analyze this text.
            The tool will:
            1. Extract core topics and concepts
            2. Create topic hierarchy
            3. Map topic relationships
            4. Provide context and significance
            
            Text: {text}
            """,
            expected_output="JSON with topic analysis",
            agent=agent
        )
        
    @staticmethod
    def create_insight_task(text: str, agent: Agent) -> Task:
        """Create insight analysis task"""
        return Task(
            description=f"""
            Use the Insight Analysis tool to analyze this text.
            The tool will:
            1. Extract key approaches and techniques
            2. Identify implementation details
            3. Highlight important points
            4. Note notable elements
            5. Suggest practical applications
            
            Text: {text}
            """,
            expected_output="JSON with insight analysis",
            agent=agent
        )
        
    @staticmethod
    def create_objective_task(text: str, agent: Agent) -> Task:
        """Create objective analysis task"""
        return Task(
            description=f"""
            Use the Objective Analysis tool to analyze this text.
            The tool will:
            1. Identify primary and secondary objectives
            2. Extract key considerations
            3. Determine learning goals
            4. Generate next steps
            
            Text: {text}
            """,
            expected_output="JSON with objective analysis",
            agent=agent
        )
        
    @staticmethod
    def create_integration_task(results: List[Dict[str, Any]], agent: Agent) -> List[Task]:
        """Create integration tasks split into manageable chunks"""
        
        # Part 1: Overview and Topics
        overview_task = Task(
            description=f"""
            Create document overview and topic analysis from these results:
            1. Extract title and main themes
            2. Analyze core topics and their relationships
            3. Return results in this JSON format:
            {{
                "document_overview": {{
                    "title": "document title",
                    "main_themes": ["theme1", "theme2"],
                    "significance": "overall significance"
                }},
                "topic_analysis": {{
                    "main_topics": [
                        {{
                            "name": "topic name",
                            "description": "topic description",
                            "subtopics": ["subtopic1", "subtopic2"]
                        }}
                    ],
                    "topic_hierarchy": {{
                        "levels": ["level1", "level2"]
                    }}
                }}
            }}
            
            Analysis results: {json.dumps(results)}
            """,
            expected_output="JSON with overview and topics",
            agent=agent
        )
        
        # Part 2: Insights and Objectives
        insights_task = Task(
            description=f"""
            Extract key insights and objectives from these results:
            1. Identify key insights and approaches
            2. Extract learning objectives and next steps
            3. Return results in this JSON format:
            {{
                "key_insights": {{
                    "approaches": [
                        {{
                            "name": "approach name",
                            "description": "description"
                        }}
                    ],
                    "key_points": [
                        {{
                            "description": "point description",
                            "significance": "importance"
                        }}
                    ]
                }},
                "learning_landscape": {{
                    "objectives": {{
                        "primary": ["objective1", "objective2"],
                        "secondary": ["objective3", "objective4"]
                    }}
                }}
            }}
            
            Analysis results: {json.dumps(results)}
            """,
            expected_output="JSON with insights and objectives",
            agent=agent
        )
        
        # Part 3: Relationships
        relationships_task = Task(
            description=f"""
            Analyze relationships between elements:
            1. Map connections between topics
            2. Identify cross-references
            3. Return results in this JSON format:
            {{
                "relationships": [
                    {{
                        "elements": ["element1", "element2"],
                        "type": "relationship type"
                    }}
                ]
            }}
            
            Analysis results: {json.dumps(results)}
            """,
            expected_output="JSON with relationships",
            agent=agent
        )
        
        return [overview_task, insights_task, relationships_task]
