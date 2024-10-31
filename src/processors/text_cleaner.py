"""Text cleaning and preprocessing module"""

import re
from typing import Dict, List, Any, Generator
import num2words
from dataclasses import dataclass
import logging
from collections import deque
import unicodedata

logger = logging.getLogger(__name__)

@dataclass
class CleaningStats:
    """Statistics about the cleaning process"""
    original_length: int
    cleaned_length: int
    num_references: int
    num_equations: int
    num_figures: int
    num_tables: int
    special_chars_removed: int

class TextCleaner:
    def __init__(self):
        # Core patterns
        self.reference_pattern = re.compile(r'\[\d+\]|\(\d+\)')
        self.latex_pattern = re.compile(r'\$.*?\$')
        self.figure_pattern = re.compile(r'(fig\.|figure|tab\.|table)\s*\d+', re.I)
        self.equation_pattern = re.compile(r'equation\s*\d+|eq\.\s*\d+', re.I)
        
        # Technical content patterns
        self.variable_pattern = re.compile(r'\b[a-zA-Z](?:_[a-zA-Z0-9]+)?\b')
        self.unit_pattern = re.compile(r'\b\d+\.?\d*\s*[a-zA-Z]+\b')
        
        # Initialize cleaning stats
        self.stats = None
        
        # Buffer for streaming large documents
        self.buffer_size = 1024 * 1024  # 1MB buffer
        
    def clean_text(self, text: str, preserve_references: bool = True, 
                  stream_mode: bool = False) -> Dict[str, Any]:
        """Clean and prepare text for TTS with optional streaming for large documents"""
        try:
            if stream_mode:
                return self._clean_text_streaming(text, preserve_references)
            
            # Initialize stats
            self.stats = CleaningStats(
                original_length=len(text),
                cleaned_length=0,
                num_references=0,
                num_equations=0,
                num_figures=0,
                num_tables=0,
                special_chars_removed=0
            )
            
            # Extract references if needed
            references = {}
            if preserve_references:
                references = self._extract_references(text)
                self.stats.num_references = len(references.get("citations", []))
            
            # Clean text in stages
            text = self._normalize_text(text)
            text = self._clean_technical_content(text)
            text = self._format_for_speech(text)
            
            # Update stats
            self.stats.cleaned_length = len(text)
            
            return {
                "text": text,
                "references": references,
                "stats": self.stats.__dict__
            }
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return {"text": text, "error": str(e)}
            
    def _clean_text_streaming(self, text: str, preserve_references: bool) -> Generator[Dict, None, None]:
        """Stream clean text in chunks for memory efficiency"""
        buffer = deque()
        current_size = 0
        
        for char in text:
            buffer.append(char)
            current_size += 1
            
            if current_size >= self.buffer_size:
                chunk = ''.join(buffer)
                cleaned_chunk = self.clean_text(chunk, preserve_references, stream_mode=False)
                yield cleaned_chunk
                buffer.clear()
                current_size = 0
                
        # Process remaining text
        if buffer:
            chunk = ''.join(buffer)
            cleaned_chunk = self.clean_text(chunk, preserve_references, stream_mode=False)
            yield cleaned_chunk
            
    def _normalize_text(self, text: str) -> str:
        """Normalize text encoding and characters"""
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Remove control characters
        text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
        
    def _extract_references(self, text: str) -> Dict[str, List[str]]:
        """Extract references from text with context"""
        references = {
            "citations": [],
            "footnotes": [],
            "bibliography": []
        }
        
        # Find citations [1] or (1)
        citations = self.reference_pattern.finditer(text)
        for match in citations:
            # Get surrounding context
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end].strip()
            
            references["citations"].append({
                "citation": match.group(),
                "context": context
            })
            
        # Find footnotes and bibliography entries
        lines = text.split('\n')
        in_bibliography = False
        
        for line in lines:
            if re.match(r'references|bibliography', line, re.I):
                in_bibliography = True
                continue
                
            if in_bibliography and line.strip():
                references["bibliography"].append(line.strip())
            elif self.reference_pattern.match(line):
                references["footnotes"].append(line.strip())
                
        return references
        
    def _clean_technical_content(self, text: str) -> str:
        """Clean technical content while preserving meaning"""
        # Track equations and figures
        self.stats.num_equations = len(self.equation_pattern.findall(text))
        self.stats.num_figures = len(self.figure_pattern.findall(text))
        
        # Convert LaTeX to readable text
        text = self.latex_pattern.sub(self._convert_latex, text)
        
        # Format variables and units
        text = self.variable_pattern.sub(self._format_variable, text)
        text = self.unit_pattern.sub(self._format_unit, text)
        
        return text
        
    def _convert_latex(self, match: re.Match) -> str:
        """Convert LaTeX expressions to readable text"""
        latex = match.group(0).replace('$', '')
        
        # Basic mathematical symbols
        latex = latex.replace('\\alpha', 'alpha')
        latex = latex.replace('\\beta', 'beta')
        latex = latex.replace('\\gamma', 'gamma')
        latex = latex.replace('\\delta', 'delta')
        latex = latex.replace('\\sum', 'sum of')
        latex = latex.replace('\\int', 'integral of')
        latex = latex.replace('\\infty', 'infinity')
        
        # Superscripts and subscripts
        latex = re.sub(r'\^(\d+)', r' to the power of \1', latex)
        latex = re.sub(r'_(\d+)', r' subscript \1', latex)
        
        # Fractions
        latex = re.sub(r'\\frac{(.+?)}{(.+?)}', r'\1 divided by \2', latex)
        
        return latex
        
    def _format_variable(self, match: re.Match) -> str:
        """Format variable names for speech"""
        var = match.group(0)
        if '_' in var:
            base, sub = var.split('_')
            return f"{base} subscript {sub}"
        return var
        
    def _format_unit(self, match: re.Match) -> str:
        """Format units for speech"""
        unit_text = match.group(0)
        number, unit = re.match(r'(\d+\.?\d*)\s*([a-zA-Z]+)', unit_text).groups()
        
        # Convert number to words if needed
        try:
            number_text = num2words.num2words(float(number))
        except:
            number_text = number
            
        # Common unit expansions
        units = {
            'm': 'meters',
            'km': 'kilometers',
            'cm': 'centimeters',
            'mm': 'millimeters',
            'g': 'grams',
            'kg': 'kilograms',
            's': 'seconds',
            'ms': 'milliseconds',
            'h': 'hours',
            'min': 'minutes'
        }
        
        unit_expanded = units.get(unit, unit)
        return f"{number_text} {unit_expanded}"
        
    def _format_for_speech(self, text: str) -> str:
        """Format text for natural speech"""
        # Convert numbers to words
        text = re.sub(
            r'\b\d+\.?\d*\b',
            lambda m: self._number_to_words(float(m.group())),
            text
        )
        
        # Add pauses for readability
        text = text.replace(';', ',')
        text = re.sub(r'([.!?])\s+', r'\1\n\n', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
        
    def _number_to_words(self, num: float) -> str:
        """Convert number to words, handling special cases"""
        try:
            if num.is_integer():
                if num > 1000000:  # Large numbers
                    return f"{num:,.0f}"  # Keep as digits with commas
                return num2words.num2words(int(num))
            
            if abs(num) < 0.01:  # Very small numbers
                return f"{num:.2e}"  # Scientific notation
            return num2words.num2words(num)
        except:
            return str(num)

text_cleaner = TextCleaner()
