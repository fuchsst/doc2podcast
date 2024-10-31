from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
from pydantic_settings import BaseSettings
from pydantic import Field, BaseModel, ConfigDict

class ProcessingConfig(BaseModel):
    chunk_size: int = Field(default=1000)
    max_chars: int = Field(default=100000)
    overlap: int = Field(default=200)
    model_config = ConfigDict(extra='allow')

class TextGenerationConfig(BaseModel):
    provider: str = Field(default="anthropic")
    default: str = Field(default="claude-3-5-sonnet-20241022")
    fallback: str = Field(default="claude-3-sonnet-20240229")
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.9)
    max_new_tokens: int = Field(default=8126)
    model_config = ConfigDict(extra='allow')

class VoiceSynthesisModelConfig(BaseModel):
    name: str
    type: str
    sampling_rate: int
    temperature: Optional[float] = None
    semantic_temperature: Optional[float] = None
    model_config = ConfigDict(extra='allow')

class VoiceSynthesisConfig(BaseModel):
    primary: VoiceSynthesisModelConfig
    secondary: VoiceSynthesisModelConfig
    model_config = ConfigDict(extra='allow')

class AudioOutputConfig(BaseModel):
    format: str = Field(default="mp3")
    bitrate: str = Field(default="192k")
    parameters: List[str] = Field(default_factory=list)
    model_config = ConfigDict(extra='allow')

class TranscriptOutputConfig(BaseModel):
    format: str = Field(default="txt")
    include_timestamps: bool = Field(default=True)
    include_speaker_labels: bool = Field(default=True)
    model_config = ConfigDict(extra='allow')

class OutputConfig(BaseModel):
    audio: AudioOutputConfig
    transcript: TranscriptOutputConfig
    script_dir: str = Field(default="outputs/scripts")
    audio_dir: str = Field(default="outputs/audio")
    segments_dir: str = Field(default="outputs/segments")
    model_config = ConfigDict(extra='allow')

class WorkflowStep(BaseModel):
    name: str
    enabled: bool = Field(default=True)
    timeout: int
    model_config = ConfigDict(extra='allow')

class WorkflowConfig(BaseModel):
    steps: List[WorkflowStep]
    model_config = ConfigDict(extra='allow')

class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")
    format: str
    handlers: List[Dict[str, Any]]
    model_config = ConfigDict(extra='allow')

class ModelsConfig(BaseModel):
    text_generation: TextGenerationConfig
    voice_synthesis: VoiceSynthesisConfig
    model_config = ConfigDict(extra='allow')

class ProjectConfig(BaseModel):
    name: str
    version: str
    description: str
    processing: ProcessingConfig
    models: ModelsConfig
    output: OutputConfig
    workflow: WorkflowConfig
    logging: LoggingConfig
    model_config = ConfigDict(extra='allow')

class Settings(BaseSettings):
    # API Keys
    ANTHROPIC_API_KEY: str = Field(default="")
    
    # Model Settings
    CLAUDE_MODEL: str = Field(default="claude-3-5-sonnet-20241022")
    MAX_TOKENS: int = Field(default=4096)
    TEMPERATURE: float = Field(default=0.7)
    
    # Project paths
    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    CONFIG_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "config")
    
    # Processing settings
    chunk_size: int = Field(default=1000)
    max_chars: int = Field(default=100000)
    overlap: int = Field(default=200)
    max_retries: int = Field(default=3)
    
    model_config = ConfigDict(
        env_file=".env",
        arbitrary_types_allowed=True,
        extra='allow'
    )
    
    # Load YAML configs
    def load_yaml_config(self, filename: str) -> Dict[str, Any]:
        config_path = self.CONFIG_DIR / filename
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    # Configuration properties
    @property
    def project_config(self) -> ProjectConfig:
        data = self.load_yaml_config("project.yaml")
        return ProjectConfig(**data)
    
    @property
    def speakers_config(self) -> Dict[str, Any]:
        return self.load_yaml_config("speakers.yaml")
    
    @property
    def tasks_config(self) -> Dict[str, Any]:
        return self.load_yaml_config("tasks.yaml")
    
    @property
    def agents_config(self) -> Dict[str, Any]:
        return self.load_yaml_config("agents.yaml")
    
    # Project settings
    @property
    def project_name(self) -> str:
        return self.project_config.name
    
    @property
    def project_version(self) -> str:
        return self.project_config.version
    
    # Model settings
    @property
    def text_generation_config(self) -> TextGenerationConfig:
        return self.project_config.models.text_generation
    
    @property
    def voice_synthesis_config(self) -> VoiceSynthesisConfig:
        return self.project_config.models.voice_synthesis
    
    # Output settings
    @property
    def audio_output_config(self) -> AudioOutputConfig:
        return self.project_config.output.audio
    
    @property
    def transcript_output_config(self) -> TranscriptOutputConfig:
        return self.project_config.output.transcript
    
    # Workflow settings
    @property
    def workflow_steps(self) -> List[WorkflowStep]:
        return self.project_config.workflow.steps
    
    # Logging settings
    @property
    def logging_config(self) -> LoggingConfig:
        return self.project_config.logging
    
    # Agent settings
    @property
    def get_agent_config(self) -> Dict[str, Dict[str, Any]]:
        return self.agents_config["agents"]
    
    def get_agent(self, agent_name: str) -> Dict[str, Any]:
        return self.get_agent_config.get(agent_name)
    
    # Task settings
    @property
    def get_task_config(self) -> Dict[str, Dict[str, Any]]:
        return self.tasks_config["tasks"]
    
    def get_task(self, task_name: str) -> Dict[str, Any]:
        return self.get_task_config.get(task_name)
    
    # Speaker settings
    @property
    def get_voice_config(self) -> Dict[str, Dict[str, Any]]:
        return self.speakers_config["voices"]
    
    def get_voice(self, category: str, voice_name: str) -> Dict[str, Any]:
        return self.get_voice_config.get(category, {}).get(voice_name)
    
    @property
    def get_prompt_config(self) -> Dict[str, Dict[str, Any]]:
        return self.speakers_config["prompts"]
    
    def get_prompt(self, category: str, prompt_name: str) -> Dict[str, Any]:
        return self.get_prompt_config.get(category, {}).get(prompt_name)
