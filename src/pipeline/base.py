"""Base classes for analysis components"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable
from dataclasses import dataclass
import json
from llama_index.core import Document
from llama_index.core.schema import TextNode
from crewai_tools import BaseTool

@dataclass
class AnalysisContext:
    """Context for analysis operations"""
    text: str
    metadata: Dict[str, Any]
    settings: Any
    cache_key: str = None

class TextProcessor(ABC):
    """Base class for text processing operations"""
    
    @abstractmethod
    def process(self, text: str) -> str:
        """Process text content"""
        pass

class DocumentAnalyzer(ABC):
    """Base class for document analysis"""
    
    @abstractmethod
    def analyze(self, context: AnalysisContext) -> Dict[str, Any]:
        """Analyze document content"""
        pass

class ChunkStrategy(ABC):
    """Base class for chunking strategies"""
    
    @abstractmethod
    def create_chunks(self, text: str, settings: Any) -> List[TextNode]:
        """Create chunks from text"""
        pass

class AnalysisTool(BaseTool):
    """Base class for analysis tools"""
    
    def __init__(self, name: str, description: str):
        """Initialize the tool with required CrewAI attributes"""
        super().__init__(
            name=name,
            description=description,
            func=self._run
        )
    
    @abstractmethod
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text content"""
        pass
    
    @abstractmethod
    def enhance_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance analysis results"""
        pass
    
    def _run(self, text: str = None, **kwargs) -> str:
        """Run the tool (required by CrewAI BaseTool class)
        
        Args:
            text (str): The text content to analyze
            **kwargs: Additional arguments passed by the agent
            
        Returns:
            str: JSON string containing the analysis results
            
        Raises:
            ValueError: If text input is missing or invalid
        """
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("Text input is required and must be a non-empty string")
            
        try:
            results = self.analyze(text)
            enhanced = self.enhance_results(results)
            return json.dumps(enhanced, indent=2)
        except Exception as e:
            error_response = {
                "error": f"Analysis failed: {str(e)}",
                "input_preview": text[:100] + "..." if len(text) > 100 else text,
                "tool_name": self.name
            }
            return json.dumps(error_response, indent=2)

class ResultsConsolidator(ABC):
    """Base class for results consolidation"""
    
    @abstractmethod
    def consolidate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidate multiple analysis results"""
        pass

class CacheStrategy(ABC):
    """Base class for caching strategies"""
    
    @abstractmethod
    def get(self, key: str) -> Any:
        """Get cached value"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set cache value"""
        pass
    
    @abstractmethod
    def has(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass
