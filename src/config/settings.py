from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
from pydantic_settings import BaseSettings
from pydantic import Field, BaseModel, ConfigDict, validator

class ProcessingConfig(BaseModel):
    chunk_size: int = Field(default=1000)
    max_chars: int = Field(default=100000)
    overlap: int = Field(default=200)
    validation: Dict[str, Any] = Field(default_factory=lambda: {
        "min_chunk_size": 1000,
        "max_chunk_size": 1000000,
        "min_overlap": 100,
        "max_overlap_ratio": 0.5
    })
    
    @validator('chunk_size')
    def validate_chunk_size(cls, v, values):
        validation = values.get('validation', {})
        min_size = validation.get('min_chunk_size', 1000)
        max_size = validation.get('max_chunk_size', 1000000)
        if not min_size <= v <= max_size:
            raise ValueError(f"chunk_size must be between {min_size} and {max_size}")
        return v
    
    @validator('overlap')
    def validate_overlap(cls, v, values):
        validation = values.get('validation', {})
        min_overlap = validation.get('min_overlap', 100)
        max_ratio = validation.get('max_overlap_ratio', 0.5)
        chunk_size = values.get('chunk_size', 1000)
        
        if v < min_overlap:
            raise ValueError(f"overlap must be at least {min_overlap}")
        if v > chunk_size * max_ratio:
            raise ValueError(f"overlap must be less than {max_ratio * 100}% of chunk_size")
        return v
    
    model_config = ConfigDict(extra='allow')

class TextGenerationConfig(BaseModel):
    provider: str = Field(default="anthropic")
    default: str = Field(default="claude-3-5-sonnet-20241022")
    fallback: str = Field(default="claude-3-sonnet-20240229")
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.9)
    max_new_tokens: int = Field(default=8126)
    validation: Dict[str, Any] = Field(default_factory=lambda: {
        "min_temperature": 0.1,
        "max_temperature": 1.0,
        "min_top_p": 0.1,
        "max_top_p": 1.0
    })
    
    @validator('temperature')
    def validate_temperature(cls, v, values):
        validation = values.get('validation', {})
        min_temp = validation.get('min_temperature', 0.1)
        max_temp = validation.get('max_temperature', 1.0)
        if not min_temp <= v <= max_temp:
            raise ValueError(f"temperature must be between {min_temp} and {max_temp}")
        return v
    
    @validator('top_p')
    def validate_top_p(cls, v, values):
        validation = values.get('validation', {})
        min_p = validation.get('min_top_p', 0.1)
        max_p = validation.get('max_top_p', 1.0)
        if not min_p <= v <= max_p:
            raise ValueError(f"top_p must be between {min_p} and {max_p}")
        return v
    
    model_config = ConfigDict(extra='allow')

class VoiceSynthesisModelConfig(BaseModel):
    name: str
    type: str
    sampling_rate: int
    temperature: Optional[float] = None
    semantic_temperature: Optional[float] = None
    validation: Dict[str, Any] = Field(default_factory=lambda: {
        "min_sampling_rate": 16000,
        "max_sampling_rate": 48000,
        "min_temperature": 0.1,
        "max_temperature": 1.0
    })
    
    @validator('sampling_rate')
    def validate_sampling_rate(cls, v, values):
        validation = values.get('validation', {})
        min_rate = validation.get('min_sampling_rate', 16000)
        max_rate = validation.get('max_sampling_rate', 48000)
        if not min_rate <= v <= max_rate:
            raise ValueError(f"sampling_rate must be between {min_rate} and {max_rate}")
        return v
    
    model_config = ConfigDict(extra='allow')

class VoiceSynthesisConfig(BaseModel):
    primary: VoiceSynthesisModelConfig
    secondary: VoiceSynthesisModelConfig
    model_config = ConfigDict(extra='allow')

class AudioOutputConfig(BaseModel):
    format: str = Field(default="mp3")
    bitrate: str = Field(default="192k")
    parameters: List[str] = Field(default_factory=list)
    validation: Dict[str, Any] = Field(default_factory=lambda: {
        "allowed_formats": ["mp3", "wav", "ogg"],
        "allowed_bitrates": ["128k", "192k", "256k", "320k"]
    })
    
    @validator('format')
    def validate_format(cls, v, values):
        validation = values.get('validation', {})
        allowed = validation.get('allowed_formats', ["mp3", "wav", "ogg"])
        if v not in allowed:
            raise ValueError(f"format must be one of {allowed}")
        return v
    
    @validator('bitrate')
    def validate_bitrate(cls, v, values):
        validation = values.get('validation', {})
        allowed = validation.get('allowed_bitrates', ["128k", "192k", "256k", "320k"])
        if v not in allowed:
            raise ValueError(f"bitrate must be one of {allowed}")
        return v
    
    model_config = ConfigDict(extra='allow')

class TranscriptOutputConfig(BaseModel):
    format: str = Field(default="txt")
    include_timestamps: bool = Field(default=True)
    include_speaker_labels: bool = Field(default=True)
    validation: Dict[str, Any] = Field(default_factory=lambda: {
        "allowed_formats": ["txt", "srt", "vtt"]
    })
    
    @validator('format')
    def validate_format(cls, v, values):
        validation = values.get('validation', {})
        allowed = validation.get('allowed_formats', ["txt", "srt", "vtt"])
        if v not in allowed:
            raise ValueError(f"format must be one of {allowed}")
        return v
    
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
    validation: Dict[str, Any] = Field(default_factory=lambda: {
        "min_timeout": 60,
        "max_timeout": 3600
    })
    
    @validator('timeout')
    def validate_timeout(cls, v, values):
        validation = values.get('validation', {})
        min_timeout = validation.get('min_timeout', 60)
        max_timeout = validation.get('max_timeout', 3600)
        if not min_timeout <= v <= max_timeout:
            raise ValueError(f"timeout must be between {min_timeout} and {max_timeout}")
        return v
    
    model_config = ConfigDict(extra='allow')

class WorkflowConfig(BaseModel):
    steps: List[WorkflowStep]
    model_config = ConfigDict(extra='allow')

class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")
    format: str
    handlers: List[Dict[str, Any]]
    validation: Dict[str, Any] = Field(default_factory=lambda: {
        "allowed_levels": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    })
    
    @validator('level')
    def validate_level(cls, v, values):
        validation = values.get('validation', {})
        allowed = validation.get('allowed_levels', ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        if v not in allowed:
            raise ValueError(f"level must be one of {allowed}")
        return v
    
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
    
    def _validate_config_file(self, filename: str) -> None:
        """Validate that a config file exists"""
        config_path = self.CONFIG_DIR / filename
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {filename}")
    
    # Load YAML configs with validation
    def load_yaml_config(self, filename: str) -> Dict[str, Any]:
        self._validate_config_file(filename)
        config_path = self.CONFIG_DIR / filename
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {filename}: {str(e)}")
    
    # Configuration properties with validation
    @property
    def project_config(self) -> ProjectConfig:
        try:
            data = self.load_yaml_config("project.yaml")
            return ProjectConfig(**data)
        except Exception as e:
            raise ValueError(f"Error loading project config: {str(e)}")
    
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
        agents = self.get_agent_config
        if agent_name not in agents:
            raise ValueError(f"Unknown agent: {agent_name}")
        return agents[agent_name]
    
    # Task settings
    @property
    def get_task_config(self) -> Dict[str, Dict[str, Any]]:
        return self.tasks_config["tasks"]
    
    def get_task(self, task_name: str) -> Dict[str, Any]:
        tasks = self.get_task_config
        if task_name not in tasks:
            raise ValueError(f"Unknown task: {task_name}")
        return tasks[task_name]
    
    # Speaker settings
    @property
    def get_voice_config(self) -> Dict[str, Dict[str, Any]]:
        return self.speakers_config["voices"]
    
    def get_voice(self, category: str, voice_name: str) -> Dict[str, Any]:
        voices = self.get_voice_config
        if category not in voices or voice_name not in voices.get(category, {}):
            raise ValueError(f"Unknown voice: {category}/{voice_name}")
        return voices[category][voice_name]
    
    @property
    def get_prompt_config(self) -> Dict[str, Dict[str, Any]]:
        return self.speakers_config["prompts"]
    
    def get_prompt(self, category: str, prompt_name: str) -> Dict[str, Any]:
        prompts = self.get_prompt_config
        if category not in prompts or prompt_name not in prompts.get(category, {}):
            raise ValueError(f"Unknown prompt: {category}/{prompt_name}")
        return prompts[category][prompt_name]
