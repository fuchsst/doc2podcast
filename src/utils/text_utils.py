"""Text processing utilities"""

import json
import re
from typing import Any, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

def extract_sections(text: str) -> List[Dict[str, str]]:
    """Extract document sections with headers"""
    sections = []
    current_section = {"header": "", "content": []}
    
    for line in text.split('\n'):
        if re.match(r'^#+\s', line) or line.isupper():  # Markdown headers or all caps
            if current_section["content"]:
                sections.append({
                    "header": current_section["header"],
                    "content": '\n'.join(current_section["content"])
                })
            current_section = {"header": line.strip('#').strip(), "content": []}
        else:
            current_section["content"].append(line)
            
    if current_section["content"]:
        sections.append({
            "header": current_section["header"],
            "content": '\n'.join(current_section["content"])
        })
        
    return sections

def format_dialogue(script: List[Tuple[str, str]], style: str = "natural") -> str:
    """Format script into dialogue format"""
    formatted = []
    
    for speaker, text in script:
        if style == "natural":
            # Add speech patterns and pauses
            text = add_speech_patterns(text)
        elif style == "tts":
            # Format specifically for TTS
            text = format_for_tts(text)
            
        formatted.append(f"{speaker}: {text}")
        
    return '\n\n'.join(formatted)

def add_speech_patterns(text: str) -> str:
    """Add natural speech patterns"""
    # Add fillers and pauses
    patterns = {
        r'([.!?])\s+': r'\1 [pause] ',
        r'(However|Moreover|Furthermore)': r'[pause] \1',
        r'(\w+),(\w+)': r'\1, [brief_pause] \2'
    }
    
    for pattern, replacement in patterns.items():
        text = re.sub(pattern, replacement, text)
        
    return text

def format_for_tts(text: str) -> str:
    """Format text specifically for TTS engines"""
    # Add TTS-specific markers
    text = re.sub(r'([A-Z]{2,})', r'[emphasis]\1[/emphasis]', text)  # Emphasize acronyms
    text = re.sub(r'([.!?])\s+', r'\1 [500ms] ', text)  # Add millisecond pauses
    text = re.sub(r'([,;])\s+', r'\1 [200ms] ', text)  # Shorter pauses for commas
    
    return text

def clean_transcript(text: str) -> str:
    """Clean transcript text"""
    # Remove unwanted characters and normalize spacing
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'\[.*?\]', '', text)  # Remove square brackets
    text = re.sub(r'[^\w\s.,!?;:\-\'"]', '', text)  # Remove special characters
    
    return text.strip()

def extract_key_points(text: str, max_points: int = 5) -> List[str]:
    """Extract key points from text"""
    # Simple extraction based on sentence importance
    sentences = text.split('.')
    scores = []
    
    for sentence in sentences:
        score = 0
        # Score based on keywords
        score += len(re.findall(r'\b(important|key|significant|main|critical)\b', sentence, re.I)) * 2
        # Score based on sentence position
        if sentence == sentences[0]:
            score += 2
        elif sentence == sentences[-1]:
            score += 1
        # Score based on length
        score += min(len(sentence.split()) / 20, 1)
        
        scores.append((score, sentence))
        
    # Get top scoring sentences
    top_points = sorted(scores, reverse=True)[:max_points]
    return [point[1].strip() for point in top_points]

def clean_json_output(output: str) -> str:
    """Clean and validate JSON output string"""
    # Convert to string if needed
    if not isinstance(output, str):
        output = str(output)
    
    # Find the first { and last } to extract the main JSON object
    start = output.find('{')
    end = output.rfind('}') + 1
    if start >= 0 and end > 0:
        output = output[start:end]
    
    # Handle line breaks within strings
    in_string = False
    cleaned = []
    i = 0
    while i < len(output):
        char = output[i]
        
        if char == '"' and (i == 0 or output[i-1] != '\\'):
            in_string = not in_string
            cleaned.append(char)
        elif in_string and char in '\n\r':
            # Replace line breaks in strings with space
            cleaned.append(' ')
        else:
            cleaned.append(char)
        i += 1
    
    output = ''.join(cleaned)
    
    # Fix common JSON formatting issues
    output = re.sub(r'(?<!\\)"(\w+)":', r'"\1":', output)  # Fix unquoted keys
    output = re.sub(r'\'', '"', output)  # Replace single quotes with double quotes
    output = re.sub(r',(\s*[}\]])', r'\1', output)  # Remove trailing commas
    
    # Handle truncated strings by attempting to complete them
    if output.count('{') > output.count('}'):
        output += '}' * (output.count('{') - output.count('}'))
    if output.count('[') > output.count(']'):
        output += ']' * (output.count('[') - output.count(']'))
    
    # Ensure proper string value formatting
    output = re.sub(r':\s*([^"{}\[\],\s][^{}\[\],\s]*?)([,}\]])', r': "\1"\2', output)
    
    # Remove any remaining whitespace between values
    output = re.sub(r'\s+', ' ', output)
    
    return output.strip()

def parse_json_safely(json_str: str) -> Dict[str, Any]:
    """Safely parse JSON with enhanced error handling"""
    try:
        # First attempt: Parse the cleaned JSON directly
        cleaned_json = clean_json_output(json_str)
        try:
            return json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            # If first attempt fails, try more aggressive cleaning
            fixed_json = re.sub(r':\s*([^"{}\[\],\s][^{}\[\],\s]*)', r': "\1"', cleaned_json)
            fixed_json = re.sub(r',(\s*[}\]])', r'\1', fixed_json)
            
            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                # If still failing, try to extract just the essential structure
                match = re.search(r'\{.*\}', cleaned_json, re.DOTALL)
                if match:
                    minimal_json = match.group(0)
                    try:
                        return json.loads(minimal_json)
                    except json.JSONDecodeError as e:
                        # One final attempt - try to fix any remaining issues
                        final_attempt = re.sub(r'([^\\])"([^"]*?)\n([^"]*?)"', r'\1"\2 \3"', minimal_json)
                        try:
                            return json.loads(final_attempt)
                        except:
                            raise ValueError(f"Failed to parse JSON after multiple attempts. Original error: {str(e)}\nOutput was: {json_str[:200]}...")
                raise ValueError(f"Failed to parse JSON: {str(e)}\nOutput was: {json_str[:200]}...")
                
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {str(e)}\nOutput was: {json_str[:200]}...")
