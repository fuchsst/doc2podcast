from typing import Dict, Any
import anthropic
from ..config.settings import Settings

class ClaudeClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = anthropic.Anthropic(api_key=self.settings.ANTHROPIC_API_KEY)
        
    def generate_script(
        self,
        content: Dict[str, Any],
        temperature: float = None,
        max_tokens: int = None
    ) -> Dict[str, Any]:
        """Generate podcast script using Claude"""
        try:
            # Get model settings
            model_config = self.settings.text_generation_config
            temperature = temperature or model_config.temperature
            max_tokens = max_tokens or model_config.max_new_tokens
            
            # Create message
            response = self.client.messages.create(
                model=model_config.default,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{
                    "role": "user",
                    "content": self._format_content(content)
                }]
            )
            
            return self._parse_response(response)
            
        except Exception as e:
            raise Exception(f"Claude API error: {str(e)}")
            
    def _format_content(self, content: Dict[str, Any]) -> str:
        """Format content for Claude prompt"""
        # Format based on content structure
        return f"""
        Title: {content.get('title', 'Untitled')}
        
        Content:
        {content.get('text', '')}
        
        Please generate a podcast script following these guidelines:
        1. Create an engaging conversation between two speakers
        2. Break down complex topics into understandable segments
        3. Include natural transitions and explanations
        4. Maintain a balance of technical accuracy and accessibility
        """
        
    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """Parse Claude response into script format"""
        # Extract and structure the response
        text = response.content[0].text
        
        # Basic parsing - this should be enhanced based on actual response format
        segments = []
        current_speaker = None
        current_text = []
        
        for line in text.split('\n'):
            if line.startswith('Speaker 1:') or line.startswith('Speaker 2:'):
                if current_speaker:
                    segments.append({
                        'speaker': current_speaker,
                        'text': ' '.join(current_text)
                    })
                current_speaker = line.split(':')[0]
                current_text = [line.split(':', 1)[1].strip()]
            elif line.strip():
                current_text.append(line.strip())
                
        if current_speaker and current_text:
            segments.append({
                'speaker': current_speaker,
                'text': ' '.join(current_text)
            })
            
        return {
            'segments': segments
        }
