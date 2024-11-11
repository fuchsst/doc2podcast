"""Base classes for generators"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json
from pydantic import BaseModel, Field

class ScriptContext:
    """Context for script generation"""
    def __init__(
        self,
        content: Dict[str, Any],
        format: Dict[str, Any],
        audience: Dict[str, Any],
        expertise: Dict[str, Any],
        guidance: Optional[str] = None
    ):
        self.content = content
        self.format = format
        self.audience = audience
        self.expertise = expertise
        self.guidance = guidance

class Generator(ABC):
    """Base generator class"""
    def __init__(self, settings: Any, callback: Optional[Any] = None):
        """Initialize generator with settings and optional callback"""
        self.settings = settings
        self.callback = callback
        
    @abstractmethod
    def generate(self, *args, **kwargs):
        """Generate content"""
        pass

class ScriptGenerator(Generator):
    """Base script generator class"""
    @abstractmethod
    def generate(self, content: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        """Generate script from content
        
        Args:
            content: Content to generate script from
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments including:
                - config: Optional script generation configuration
            
        Returns:
            Generated script data
        """
        pass

class ResultsFormatter:
    """Base results formatter class"""
    @abstractmethod
    def format_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format generation results"""
        try:
            return json.loads(json.dumps(results))
        except Exception as e:
            return {"error": f"Failed to format results: {str(e)}"}

class ScriptToolArgs(BaseModel):
    """Base schema for script tool arguments"""
    context: Dict[str, Any] = Field(
        description="Script generation context containing content, format, audience, and expertise information"
    )

class ScriptTool(ABC):
    """Base class for script generation tools"""
    
    def __init__(self, name: str, description: str, llm: Any):
        self.name = name
        self.description = description
        self.llm = llm
        self.args_schema = ScriptToolArgs
        
    @abstractmethod
    def analyze(self, context: ScriptContext) -> Dict[str, Any]:
        """Analyze input and generate results"""
        pass
        
    @abstractmethod
    def enhance_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance and validate results"""
        pass
        
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        try:
            return json.loads(response)
        except Exception as e:
            raise ValueError(f"Failed to parse response: {str(e)}")
            
    def run(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Run tool with provided context
        
        This method is called by CrewAI's Agent
        
        Args:
            context: Context dictionary containing content, format, audience, and expertise info
            **kwargs: Additional keyword arguments
            
        Returns:
            Generated results
        """
        script_context = ScriptContext(**context)
        results = self.analyze(script_context)
        return self.enhance_results(results)
