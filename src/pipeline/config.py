"""Configuration management for analysis pipeline"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from ..config.settings import Settings

@dataclass
class AnalysisConfig:
    """Configuration for analysis tools"""
    chunk_size: int
    max_features: int = 100
    num_topics: int = 5
    context_window: int = 2
    min_importance: float = 0.3
    
    @classmethod
    def from_settings(cls, settings: Settings) -> 'AnalysisConfig':
        """Create config from settings"""
        processing = settings.project_config.processing
        return cls(
            chunk_size=processing.chunk_size,
            max_features=100,  # Default
            num_topics=5,  # Default
            context_window=2,  # Default
            min_importance=0.3  # Default
        )

@dataclass
class AgentConfig:
    """Configuration for agent execution"""
    verbose: bool = True
    allow_delegation: bool = True
    process: str = "sequential"
    max_iterations: int = 3
    
    @classmethod
    def from_settings(cls, settings: Settings) -> 'AgentConfig':
        """Create config from settings"""
        return cls(
            verbose=True,  # Default
            allow_delegation=True,  # Default
            process="sequential",  # Default
            max_iterations=3  # Default
        )

@dataclass
class ProcessingConfig:
    """Configuration for document processing"""
    chunk_size: int
    overlap: int
    cache_enabled: bool = True
    analysis_config: Optional[AnalysisConfig] = None
    agent_config: Optional[AgentConfig] = None
    
    @classmethod
    def from_settings(cls, settings: Settings) -> 'ProcessingConfig':
        """Create config from settings"""
        processing = settings.project_config.processing
        return cls(
            chunk_size=processing.chunk_size,
            overlap=processing.overlap,
            cache_enabled=True,  # Default
            analysis_config=AnalysisConfig.from_settings(settings),
            agent_config=AgentConfig.from_settings(settings)
        )
    
    def with_overrides(self, overrides: Dict[str, Any]) -> 'ProcessingConfig':
        """Create new config with overrides"""
        config_dict = {
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "cache_enabled": self.cache_enabled,
            "analysis_config": self.analysis_config,
            "agent_config": self.agent_config
        }
        
        # Apply overrides
        if "chunk_size" in overrides:
            config_dict["chunk_size"] = overrides["chunk_size"]
        if "overlap" in overrides:
            config_dict["overlap"] = overrides["overlap"]
        if "cache_enabled" in overrides:
            config_dict["cache_enabled"] = overrides["cache_enabled"]
            
        # Handle nested configs
        if "analysis_config" in overrides and self.analysis_config:
            analysis_dict = overrides["analysis_config"]
            config_dict["analysis_config"] = AnalysisConfig(
                chunk_size=analysis_dict.get("chunk_size", self.analysis_config.chunk_size),
                max_features=analysis_dict.get("max_features", self.analysis_config.max_features),
                num_topics=analysis_dict.get("num_topics", self.analysis_config.num_topics),
                context_window=analysis_dict.get("context_window", self.analysis_config.context_window),
                min_importance=analysis_dict.get("min_importance", self.analysis_config.min_importance)
            )
            
        if "agent_config" in overrides and self.agent_config:
            agent_dict = overrides["agent_config"]
            config_dict["agent_config"] = AgentConfig(
                verbose=agent_dict.get("verbose", self.agent_config.verbose),
                allow_delegation=agent_dict.get("allow_delegation", self.agent_config.allow_delegation),
                process=agent_dict.get("process", self.agent_config.process),
                max_iterations=agent_dict.get("max_iterations", self.agent_config.max_iterations)
            )
            
        return ProcessingConfig(**config_dict)

class ConfigurationManager:
    """Manages configuration for analysis pipeline"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.processing_config = ProcessingConfig.from_settings(settings)
        
    def get_processing_config(self, overrides: Optional[Dict[str, Any]] = None) -> ProcessingConfig:
        """Get processing configuration with optional overrides"""
        if overrides:
            return self.processing_config.with_overrides(overrides)
        return self.processing_config
        
    def get_analysis_config(self, overrides: Optional[Dict[str, Any]] = None) -> AnalysisConfig:
        """Get analysis configuration with optional overrides"""
        config = self.processing_config.analysis_config or AnalysisConfig.from_settings(self.settings)
        if overrides:
            return AnalysisConfig(
                chunk_size=overrides.get("chunk_size", config.chunk_size),
                max_features=overrides.get("max_features", config.max_features),
                num_topics=overrides.get("num_topics", config.num_topics),
                context_window=overrides.get("context_window", config.context_window),
                min_importance=overrides.get("min_importance", config.min_importance)
            )
        return config
        
    def get_agent_config(self, overrides: Optional[Dict[str, Any]] = None) -> AgentConfig:
        """Get agent configuration with optional overrides"""
        config = self.processing_config.agent_config or AgentConfig.from_settings(self.settings)
        if overrides:
            return AgentConfig(
                verbose=overrides.get("verbose", config.verbose),
                allow_delegation=overrides.get("allow_delegation", config.allow_delegation),
                process=overrides.get("process", config.process),
                max_iterations=overrides.get("max_iterations", config.max_iterations)
            )
        return config
