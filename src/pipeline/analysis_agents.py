"""Analysis agents for document processing using CrewAI"""

from typing import List, Dict, Any
from dataclasses import dataclass
from crewai import Agent, Task, Crew
from .analysis_tools import (
    TopicAnalysisTool,
    InsightAnalysisTool,
    QuestionAnalysisTool
)
import json

@dataclass
class AnalysisResult:
    """Container for analysis results"""
    title: str
    briefing: Dict[str, Any]
    keywords: List[str]
    topics: Dict[str, Any]
    key_insights: Dict[str, Any]
    questions: Dict[str, Any]

class AnalysisAgents:
    """Manages document analysis agents and tasks"""
    
    def __init__(self, llm):
        self.llm = llm
        # Initialize tools
        self.topic_tool = TopicAnalysisTool(llm)
        self.insight_tool = InsightAnalysisTool(llm)
        self.question_tool = QuestionAnalysisTool(llm)
        self.initialize_agents()
        
    def initialize_agents(self):
        """Initialize specialized analysis agents"""
        
        # Title & Briefing Agent
        self.briefing_agent = Agent(
            role="Content Summarizer",
            goal="Create engaging titles and briefings",
            backstory="""Expert in distilling complex content into compelling summaries
            and creating attention-grabbing titles.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        )
        
        # Topic Analysis Agent
        self.topic_agent = Agent(
            role="Topic Analyst",
            goal="Extract and organize document topics",
            backstory="""Expert in identifying core topics, themes and concepts. 
            Skilled at creating topic hierarchies and relationship mapping.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            tools=[self.topic_tool]
        )
        
        # Key Insights Agent
        self.insights_agent = Agent(
            role="Research Analyst",
            goal="Extract key research insights and findings",
            backstory="""Expert in analyzing academic content to identify methodologies,
            findings, contributions and implications.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            tools=[self.insight_tool]
        )
        
        # Question Analysis Agent
        self.question_agent = Agent(
            role="Question Analyst",
            goal="Analyze research questions and generate follow-ups",
            backstory="""Expert in identifying research questions, hypotheses,
            and generating insightful follow-up questions.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            tools=[self.question_tool]
        )
        
        # Results Integration Agent
        self.integration_agent = Agent(
            role="Research Integrator",
            goal="Synthesize and integrate analysis results",
            backstory="""Expert in combining and synthesizing multiple analyses
            into coherent, comprehensive summaries.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        )
        
    def create_briefing_task(self, chunk: str) -> Task:
        """Create briefing task for a chunk"""
        return Task(
            description=f"""
            Create a briefing for this content:
            1. Create a concise executive summary
            2. Extract key takeaways and main points
            3. Return results in JSON format with:
               - briefing: object with summary and takeaways
            
            Text chunk: {chunk}
            """,
            expected_output="JSON with briefing content",
            agent=self.briefing_agent
        )
        
    def create_topic_task(self, chunk: str) -> Task:
        """Create topic extraction task for a chunk"""
        return Task(
            description=f"""
            Use the Topic Analysis tool to analyze this text chunk.
            The tool combines TF-IDF, LDA, and LLM reasoning to:
            1. Identify key topics and terms
            2. Create topic hierarchy
            3. Map relationships between topics
            4. Provide context and significance
            
            Text chunk: {chunk}
            """,
            expected_output="JSON with topic analysis results",
            agent=self.topic_agent
        )
        
    def create_insights_task(self, chunk: str) -> Task:
        """Create key insights extraction task for a chunk"""
        return Task(
            description=f"""
            Use the Insight Analysis tool to analyze this text chunk.
            The tool combines sentence importance scoring with LLM reasoning to:
            1. Identify key research insights
            2. Extract methodological approaches
            3. Highlight significant findings
            4. Note novel contributions
            5. Suggest future directions
            
            Text chunk: {chunk}
            """,
            expected_output="JSON with key insights analysis",
            agent=self.insights_agent
        )
        
    def create_questions_task(self, chunk: str) -> Task:
        """Create question analysis task for a chunk"""
        return Task(
            description=f"""
            Use the Question Analysis tool to analyze this text chunk.
            The tool combines pattern matching with LLM reasoning to:
            1. Identify primary and secondary research questions
            2. Analyze hypotheses and their relationships
            3. Generate relevant follow-up questions
            4. Map questions to potential answers
            
            Text chunk: {chunk}
            """,
            expected_output="JSON with question analysis",
            agent=self.question_agent
        )
        
    def create_integration_task(self, results: List[Dict[str, Any]]) -> Task:
        """Create task to integrate multiple analysis results"""
        return Task(
            description=f"""
            Synthesize and integrate these analysis results:
            1. Identify common themes and patterns
            2. Resolve any contradictions or inconsistencies
            3. Create a coherent narrative from the findings
            4. Highlight the most significant insights
            5. Generate a comprehensive summary
            6. Return results in JSON format with:
               - integrated_topics
               - integrated_insights
               - integrated_questions
               - overall_summary
            
            Analysis results: {json.dumps(results, indent=2)}
            """,
            expected_output="JSON with integrated analysis",
            agent=self.integration_agent
        )
        
    def create_title_task(self, briefing: Dict[str, Any]) -> Task:
        """Create task to generate final title from briefing"""
        return Task(
            description=f"""
            Generate an engaging title based on this briefing:
            1. Consider the main themes and key findings
            2. Make it attention-grabbing but accurate
            3. Ensure it reflects the content's depth
            4. Keep it concise but descriptive
            5. Return results in JSON format with:
               - title: string
               - subtitle: string (optional)
            
            Briefing content: 
            {json.dumps(briefing, indent=2)}
            """,
            expected_output="JSON with title",
            agent=self.briefing_agent
        )
        
    def analyze_chunk(self, chunk: str) -> AnalysisResult:
        """Run all analysis tasks on a single chunk"""
        
        # Create tasks
        briefing_task = self.create_briefing_task(chunk)
        topic_task = self.create_topic_task(chunk)
        insights_task = self.create_insights_task(chunk)
        questions_task = self.create_questions_task(chunk)
        
        # Create crew for sequential processing
        crew = Crew(
            agents=[self.briefing_agent, self.topic_agent, 
                   self.insights_agent, self.question_agent],
            tasks=[briefing_task, topic_task, 
                  insights_task, questions_task],
            verbose=True
        )
        
        # Execute tasks and get results
        crew_output = crew.kickoff()
        
        # Parse JSON results
        try:
            # Get task outputs from CrewOutput
            task_outputs = crew_output.tasks_output
            
            # Parse JSON from task outputs
            briefing_data = json.loads(task_outputs[0].raw)
            topics = json.loads(task_outputs[1].raw)
            insights = json.loads(task_outputs[2].raw)
            questions = json.loads(task_outputs[3].raw)
            
            # Extract keywords from topics
            keywords = []
            if "main_topics" in topics:
                for topic in topics["main_topics"]:
                    keywords.extend(topic.get("key_terms", []))
            
        except (json.JSONDecodeError, AttributeError, IndexError) as e:
            # Fallback to raw output if parsing fails
            briefing_data = {"briefing": {"summary": str(crew_output)}}
            topics = {"raw": str(crew_output)}
            insights = {"raw": str(crew_output)}
            questions = {"raw": str(crew_output)}
            keywords = []
            
        return AnalysisResult(
            title="",  # Title will be generated after combining results
            briefing=briefing_data.get("briefing", {}),
            keywords=list(set(keywords)),
            topics=topics,
            key_insights=insights,
            questions=questions
        )
        
    def combine_results(self, chunk_results: List[AnalysisResult]) -> Dict[str, Any]:
        """Combine results from multiple chunks using LLM"""
        
        # Convert results to list of dicts for integration
        results_for_integration = []
        for result in chunk_results:
            results_for_integration.append({
                "briefing": result.briefing,
                "keywords": result.keywords,
                "topics": result.topics,
                "key_insights": result.key_insights,
                "questions": result.questions
            })
            
        # Create and execute integration task
        integration_task = self.create_integration_task(results_for_integration)
        integration_crew = Crew(
            agents=[self.integration_agent],
            tasks=[integration_task],
            verbose=True
        )
        
        # Get integration results
        integration_output = integration_crew.kickoff()
        integrated_results = integration_output.tasks_output[0].raw
        
        try:
            integrated_data = json.loads(integrated_results)
        except json.JSONDecodeError:
            integrated_data = {
                "integrated_topics": {},
                "integrated_insights": {},
                "integrated_questions": {},
                "overall_summary": str(integrated_results)
            }
            
        # Create briefing from integrated results
        briefing = {
            "summary": integrated_data.get("overall_summary", ""),
            "takeaways": integrated_data.get("key_points", [])
        }
        
        # Generate final title based on briefing
        title_task = self.create_title_task(briefing)
        title_crew = Crew(
            agents=[self.briefing_agent],
            tasks=[title_task],
            verbose=True
        )
        
        # Get title results
        title_output = title_crew.kickoff()
        title_result = title_output.tasks_output[0].raw
        
        try:
            title_data = json.loads(title_result)
            title = title_data.get("title", "Untitled")
        except json.JSONDecodeError:
            title = "Untitled"
            
        return {
            "title": title,
            "briefing": briefing,
            "keywords": list(set(
                kw for result in chunk_results 
                for kw in result.keywords
            )),
            "topics": integrated_data.get("integrated_topics", {}),
            "key_insights": integrated_data.get("integrated_insights", {}),
            "questions": integrated_data.get("integrated_questions", {})
        }
