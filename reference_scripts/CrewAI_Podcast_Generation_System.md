```python
from pathlib import Path
import yaml
from crewai import Agent, Task, Crew, Process
from typing import List, Dict
import json

class PodcastCrewSystem:
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.load_configurations()
        self.initialize_agents()

    def load_configurations(self):
        """Load all configuration files"""
        with open(self.config_dir / "agents.yaml", "r") as f:
            self.agents_config = yaml.safe_load(f)
        with open(self.config_dir / "tasks.yaml", "r") as f:
            self.tasks_config = yaml.safe_load(f)

    def initialize_agents(self):
        """Initialize all agents with their configurations"""
        # Document Analysis Agents
        self.document_processor = Agent(
            config=self.agents_config["document_processor"],
            allow_delegation=False,
            verbose=True
        )
        
        self.topic_analyzer = Agent(
            config=self.agents_config["topic_analyzer"],
            allow_delegation=False,
            verbose=True
        )
        
        self.insight_extractor = Agent(
            config=self.agents_config["insight_extractor"],
            allow_delegation=False,
            verbose=True
        )
        
        self.qa_analyzer = Agent(
            config=self.agents_config["qa_analyzer"],
            allow_delegation=False,
            verbose=True
        )

        # Content Creation Agents
        self.briefing_creator = Agent(
            config=self.agents_config["briefing_creator"],
            allow_delegation=False,
            verbose=True
        )
        
        self.script_writer = Agent(
            config=self.agents_config["script_writer"],
            allow_delegation=False,
            verbose=True
        )
        
        self.dialogue_enhancer = Agent(
            config=self.agents_config["dialogue_enhancer"],
            allow_delegation=False,
            verbose=True
        )

        # Quality Control Agents
        self.fact_checker = Agent(
            config=self.agents_config["fact_checker"],
            allow_delegation=False,
            verbose=True
        )
        
        self.quality_scorer = Agent(
            config=self.agents_config["quality_scorer"],
            allow_delegation=False,
            verbose=True
        )

    def create_document_analysis_crew(self) -> Crew:
        """Create crew for document analysis phase"""
        tasks = [
            Task(
                description=self.tasks_config["document_processing"]["description"],
                expected_output=self.tasks_config["document_processing"]["expected_output"],
                agent=self.document_processor
            ),
            Task(
                description=self.tasks_config["topic_analysis"]["description"],
                expected_output=self.tasks_config["topic_analysis"]["expected_output"],
                agent=self.topic_analyzer
            ),
            Task(
                description=self.tasks_config["insight_extraction"]["description"],
                expected_output=self.tasks_config["insight_extraction"]["expected_output"],
                agent=self.insight_extractor
            ),
            Task(
                description=self.tasks_config["qa_analysis"]["description"],
                expected_output=self.tasks_config["qa_analysis"]["expected_output"],
                agent=self.qa_analyzer
            )
        ]
        
        return Crew(
            agents=[self.document_processor, self.topic_analyzer, 
                   self.insight_extractor, self.qa_analyzer],
            tasks=tasks,
            verbose=2,
            process=Process.sequential
        )

    def create_content_generation_crew(self) -> Crew:
        """Create crew for content generation phase"""
        tasks = [
            Task(
                description=self.tasks_config["briefing_creation"]["description"],
                expected_output=self.tasks_config["briefing_creation"]["expected_output"],
                agent=self.briefing_creator
            ),
            Task(
                description=self.tasks_config["script_writing"]["description"],
                expected_output=self.tasks_config["script_writing"]["expected_output"],
                agent=self.script_writer
            ),
            Task(
                description=self.tasks_config["dialogue_enhancement"]["description"],
                expected_output=self.tasks_config["dialogue_enhancement"]["expected_output"],
                agent=self.dialogue_enhancer
            )
        ]
        
        return Crew(
            agents=[self.briefing_creator, self.script_writer, self.dialogue_enhancer],
            tasks=tasks,
            verbose=2,
            process=Process.sequential
        )

    def create_quality_control_crew(self) -> Crew:
        """Create crew for quality control phase"""
        tasks = [
            Task(
                description=self.tasks_config["fact_checking"]["description"],
                expected_output=self.tasks_config["fact_checking"]["expected_output"],
                agent=self.fact_checker
            ),
            Task(
                description=self.tasks_config["quality_scoring"]["description"],
                expected_output=self.tasks_config["quality_scoring"]["expected_output"],
                agent=self.quality_scorer
            )
        ]
        
        return Crew(
            agents=[self.fact_checker, self.quality_scorer],
            tasks=tasks,
            verbose=2,
            process=Process.sequential
        )

    def process_documents(self, documents: List[Dict]) -> Dict:
        """Process documents through all phases"""
        results = {}
        
        # Document Analysis Phase
        analysis_crew = self.create_document_analysis_crew()
        analysis_results = analysis_crew.kickoff()
        results['analysis'] = json.loads(analysis_results)
        
        # Content Generation Phase
        generation_crew = self.create_content_generation_crew()
        generation_results = generation_crew.kickoff()
        results['content'] = json.loads(generation_results)
        
        # Quality Control Phase
        qc_crew = self.create_quality_control_crew()
        qc_results = qc_crew.kickoff()
        results['quality'] = json.loads(qc_results)
        
        return results

def main():
    # Initialize the system
    podcast_system = PodcastCrewSystem()
    
    # Example documents
    documents = [
        {
            "path": "path/to/document1.pdf",
            "type": "research_paper",
            "content": "..."
        },
        {
            "path": "path/to/document2.pdf",
            "type": "article",
            "content": "..."
        }
    ]
    
    # Process documents
    results = podcast_system.process_documents(documents)
    
    # Save results
    with open("podcast_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
```