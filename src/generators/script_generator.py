"""Script generation module"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from crewai import Agent, Task, Crew, Process, LLM

from ..config import PromptManager
from ..config.settings import Settings
from ..models.podcast_script import (
    PodcastScript,
    PodcastMetadata,
    Speaker,
    ScriptSegment,
    VoiceParameters,
    Reference
)
from ..utils.callback_handler import PipelineCallback, StepType

class ScriptGenerator:
    """Generates podcast scripts from processed documents"""
    
    def __init__(self, settings: Settings, callback: Optional[PipelineCallback] = None):
        self.settings = settings
        self.prompt_manager = PromptManager(settings)
        self.max_chunk_tokens = 50000  # Leave room for system prompts
        self.callback = callback
        
        # Initialize LLM with Anthropic model
        self.llm = LLM(
            model=settings.text_generation_config.default,
            temperature=settings.text_generation_config.temperature,
            max_tokens=settings.text_generation_config.max_new_tokens,
            api_key=settings.ANTHROPIC_API_KEY
        )
        
        self.initialize_agents()
                
    def initialize_agents(self):
        """Initialize CrewAI agents"""
        # Document Analysis Agent
        doc_processor = self.prompt_manager.get_agent_prompt("document_processor")
        self.analyst = Agent(
            role=doc_processor["role"],
            goal=doc_processor["goal"],
            backstory=doc_processor["backstory"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            step_callback=self._on_agent_step
        )
        
        # Script Writing Agent
        script_writer = self.prompt_manager.get_agent_prompt("script_writer")
        self.writer = Agent(
            role=script_writer["role"],
            goal=script_writer["goal"],
            backstory=script_writer["backstory"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            step_callback=self._on_agent_step
        )
        
        # Quality Control Agent
        quality_checker = self.prompt_manager.get_agent_prompt("quality_checker")
        self.quality_checker = Agent(
            role=quality_checker["role"],
            goal=quality_checker["goal"],
            backstory=quality_checker["backstory"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            step_callback=self._on_agent_step
        )

    def _on_agent_step(self, step_output: Dict[str, Any]):
        """Handle agent step callback"""
        if self.callback:
            # Extract agent role and current task
            agent_role = step_output.get("agent_role", "Unknown")
            task_desc = step_output.get("task_description", "")
            
            # Map agent roles to progress percentages
            progress_map = {
                "Document Processor": 25,
                "Script Writer": 50,
                "Quality Checker": 75
            }
            
            self.callback.on_script_generation(
                progress=progress_map.get(agent_role, 0),
                message=f"{agent_role}: {task_desc}",
                substeps=[step_output]
            )

    def create_analysis_task(self, content: Dict[str, Any]) -> Task:
        """Create content analysis task"""
        task_config = self.prompt_manager.get_task_prompt("document_analysis")
        # Ensure content text fits within limits
        if isinstance(content, dict) and "text" in content:
            content["text"] = self._truncate_text(content["text"])
        return Task(
            description=task_config["description"].format(content=json.dumps(content)),
            expected_output=task_config["expected_output"],
            agent=self.analyst
        )
        
    def create_writing_task(self, analysis_result: str) -> Task:
        """Create script writing task"""
        task_config = self.prompt_manager.get_task_prompt("content_generation")
        return Task(
            description=task_config["description"].format(analysis_result=analysis_result),
            expected_output=task_config["expected_output"],
            agent=self.writer
        )
        
    def create_quality_task(self, script: str) -> Task:
        """Create quality control task"""
        task_config = self.prompt_manager.get_task_prompt("quality_control")
        return Task(
            description=task_config["description"].format(script=script),
            expected_output=task_config["expected_output"],
            agent=self.quality_checker
        )
        
    def _create_default_speakers(self) -> Dict[str, Speaker]:
        """Create default speaker configurations"""
        host = Speaker(
            name="Host",
            voice_model="f5-tts",
            voice_preset="sarah_chen",
            style_tags=["professional", "technical"],
            voice_parameters=VoiceParameters(
                pace=1.0,
                pitch=1.0,
                energy=1.0,
                emotion="neutral",
                variation=0.1
            )
        )
        
        guest = Speaker(
            name="Guest",
            voice_model="f5-tts",
            voice_preset="alex_rivera",
            style_tags=["casual", "curious"],
            voice_parameters=VoiceParameters(
                pace=1.0,
                pitch=1.0,
                energy=1.0,
                emotion="neutral",
                variation=0.2
            )
        )
        
        return {"Host": host, "Guest": guest}
        
    def _convert_to_podcast_script(
        self,
        initial_script: str,
        tts_script: str,
        document_title: str,
        source_path: Optional[str] = None
    ) -> PodcastScript:
        """Convert generated scripts to PodcastScript format"""
        speakers = self._create_default_speakers()
        
        # Create metadata
        metadata = PodcastMetadata(
            title=document_title,
            source_document=source_path,
            tags=["technical", "research"],
            duration=None  # Will be set when audio is generated
        )
        
        # Parse TTS script into segments
        segments = []
        current_text = []
        current_speaker = None
        
        for line in tts_script.split('\n'):
            line = line.strip()
            if not line:
                if current_speaker and current_text:
                    segments.append(ScriptSegment(
                        speaker=speakers[current_speaker],
                        text=' '.join(current_text),
                        duration=None,
                        audio_path=None
                    ))
                    current_text = []
                continue
                
            if ': ' in line:
                if current_speaker and current_text:
                    segments.append(ScriptSegment(
                        speaker=speakers[current_speaker],
                        text=' '.join(current_text),
                        duration=None,
                        audio_path=None
                    ))
                    current_text = []
                    
                speaker, text = line.split(': ', 1)
                current_speaker = "Host" if "Speaker 1" in speaker else "Guest"
                current_text = [text]
            else:
                current_text.append(line)
                
        # Add final segment
        if current_speaker and current_text:
            segments.append(ScriptSegment(
                speaker=speakers[current_speaker],
                text=' '.join(current_text),
                duration=None,
                audio_path=None
            ))
            
        return PodcastScript(
            metadata=metadata,
            segments=segments,
            settings={"original_script": initial_script}
        )
        
    def _truncate_text(self, text: str, max_tokens: Optional[int] = None) -> str:
        """Truncate text to fit within token limit"""
        max_tokens = max_tokens or self.max_chunk_tokens
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens])
        
    def _split_content(self, content: Union[Dict[str, Any], str]) -> List[Dict[str, Any]]:
        """Split content into manageable chunks for processing"""
        chunks = []
        
        # Handle string content
        if isinstance(content, str):
            text = content
            while text:
                chunk_text = self._truncate_text(text)
                chunks.append({"text": chunk_text, "metadata": {}})
                remaining_words = text.split()[self.max_chunk_tokens:]
                text = " ".join(remaining_words) if remaining_words else ""
            return chunks
            
        # Handle dictionary content
        metadata = content.get("metadata", {})
        text = ""
        
        # Process chunks if available
        if "chunks" in content:
            for chunk in content["chunks"]:
                # Handle both string and dict chunks
                chunk_text = chunk["text"] if isinstance(chunk, dict) else str(chunk)
                text += "\n\n" + chunk_text
        else:
            # Handle content without chunks
            text = content.get("text", str(content))
            
        # Split the combined text
        while text:
            chunk_text = self._truncate_text(text)
            chunks.append({"text": chunk_text, "metadata": metadata})
            remaining_words = text.split()[self.max_chunk_tokens:]
            text = " ".join(remaining_words) if remaining_words else ""
            
        return chunks

    def generate_initial_script(self, content: str) -> str:
        """Generate initial podcast script from analyzed content"""
        try:
            # Create a task for script generation using content_generation task
            task = self.create_writing_task(content)
            
            # Execute the task
            crew = Crew(
                agents=[self.writer],
                tasks=[task],
                verbose=True,
                process=Process.sequential
            )
            
            result = crew.kickoff()
            return result if isinstance(result, str) else str(result)
            
        except Exception as e:
            print(f"Error in generate_initial_script: {str(e)}")
            if self.callback:
                self.callback.on_error(StepType.SCRIPT_GENERATION, str(e))
            # Fallback to direct LLM call
            try:
                task_config = self.prompt_manager.get_task_prompt("content_generation")
                response = self.llm.generate(
                    task_config["description"].format(analysis_result=content)
                )
                return str(response)
            except Exception as e:
                print(f"Fallback script generation failed: {str(e)}")
                raise

    def rewrite_for_tts(self, script: str) -> str:
        """Optimize script for text-to-speech processing"""
        try:
            # Create task for TTS optimization using quality_control task
            task = self.create_quality_task(script)
            
            # Execute the task
            crew = Crew(
                agents=[self.quality_checker],
                tasks=[task],
                verbose=True,
                process=Process.sequential
            )
            
            result = crew.kickoff()
            return result if isinstance(result, str) else str(result)
            
        except Exception as e:
            print(f"Error in rewrite_for_tts: {str(e)}")
            if self.callback:
                self.callback.on_error(StepType.SCRIPT_GENERATION, str(e))
            # Fallback to direct LLM call
            try:
                task_config = self.prompt_manager.get_task_prompt("quality_control")
                response = self.llm.generate(
                    task_config["description"].format(script=script)
                )
                return str(response)
            except Exception as e:
                print(f"Fallback TTS optimization failed: {str(e)}")
                raise
        
    def _prepare_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Chunk objects and PDF text objects to dictionaries for JSON serialization"""
        def convert_chunk(chunk):
            if hasattr(chunk, '__dict__'):
                # Handle Chunk objects
                return {
                    "text": str(getattr(chunk, 'text', '')),
                    "start_idx": getattr(chunk, 'start_idx', 0),
                    "end_idx": getattr(chunk, 'end_idx', 0),
                    "importance_score": getattr(chunk, 'importance_score', 0),
                    "topics": getattr(chunk, 'topics', []),
                    "metadata": getattr(chunk, 'metadata', {}),
                    "references": getattr(chunk, 'references', {})
                }
            # Convert any PDF text objects to string
            return str(chunk)

        if isinstance(content, str):
            return {"text": content}

        processed = {}
        for key, value in content.items():
            if key == "chunks":
                processed[key] = [convert_chunk(chunk) for chunk in value]
            elif isinstance(value, dict):
                processed[key] = self._prepare_content(value)
            elif isinstance(value, list):
                processed[key] = [
                    convert_chunk(item) if hasattr(item, '__dict__') else str(item)
                    for item in value
                ]
            else:
                # Convert any PDF text objects to string
                processed[key] = str(value)
        return processed
        
    def generate_script(
        self,
        content: Dict[str, Any]
    ) -> PodcastScript:
        """Generate complete podcast script"""
        try:
            # Prepare content for JSON serialization
            processed_content = self._prepare_content(content)
            
            # Split content into manageable chunks
            content_chunks = self._split_content(processed_content)
            
            # Process each chunk
            results = []
            for chunk in content_chunks:
                try:
                    # Create CrewAI tasks for this chunk
                    analysis_task = self.create_analysis_task(chunk)
                    writing_task = self.create_writing_task("{analysis_result}")
                    quality_task = self.create_quality_task("{script}")
                    
                    # Create and run crew
                    crew = Crew(
                        agents=[self.analyst, self.writer, self.quality_checker],
                        tasks=[analysis_task, writing_task, quality_task],
                        verbose=True,
                        process=Process.sequential
                    )
                    
                    results.append(crew.kickoff())
                except Exception as e:
                    print(f"Error processing chunk with CrewAI: {str(e)}")
                    if self.callback:
                        self.callback.on_error(StepType.SCRIPT_GENERATION, str(e))
                    continue
                
            # Combine results
            combined_result = "\n\n".join(results)
            
            # Generate and optimize script
            initial_script = self.generate_initial_script(combined_result)
            tts_script = self.rewrite_for_tts(initial_script)
            
            # Convert to PodcastScript format
            document_title = content.get("metadata", {}).get("title", "Untitled Document")
            source_path = content.get("metadata", {}).get("source_path")
            
            script = self._convert_to_podcast_script(
                initial_script=initial_script,
                tts_script=tts_script,
                document_title=document_title,
                source_path=source_path
            )
            
            # Save scripts (for reference)
            output_dir = Path(self.settings.project_config.output.script_dir)
            output_dir.mkdir(exist_ok=True)
            
            with open(output_dir / "initial_script.txt", 'w', encoding='utf-8') as f:
                f.write(initial_script)
                
            with open(output_dir / "tts_script.txt", 'w', encoding='utf-8') as f:
                f.write(tts_script)
                
            return script
            
        except Exception as e:
            print(f"Error generating script: {str(e)}")
            if self.callback:
                self.callback.on_error(StepType.SCRIPT_GENERATION, str(e))
            try:
                # Fallback to simpler processing
                if not isinstance(content, str):
                    content = self._prepare_content(content)
                    content_chunks = self._split_content(content)
                    content = "\n\n".join(chunk["text"] for chunk in content_chunks)
                    
                initial_script = self.generate_initial_script(content)
                tts_script = self.rewrite_for_tts(initial_script)
                
                document_title = (
                    content.get("metadata", {}).get("title", "Untitled Document")
                    if isinstance(content, dict) else "Untitled Document"
                )
                
                return self._convert_to_podcast_script(
                    initial_script=initial_script,
                    tts_script=tts_script,
                    document_title=document_title
                )
            except Exception as e:
                print(f"Fallback generation failed: {str(e)}")
                if self.callback:
                    self.callback.on_error(StepType.SCRIPT_GENERATION, str(e))
                raise
