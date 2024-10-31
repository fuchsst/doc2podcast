"""Manage text chunking for optimal processing"""

from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass
import logging
import re
from nltk.tokenize import sent_tokenize
import nltk

from ..config.settings import Settings

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    text: str
    start_idx: int
    end_idx: int
    references: Dict = None
    metadata: Dict = None
    importance_score: float = 0.0
    topics: List[str] = None

class ChunkManager:
    """Manage text chunking and processing"""
    
    def __init__(self, settings: Settings):
        self.chunk_size = settings.chunk_size
        self.overlap = settings.overlap or int(self.chunk_size * 0.2)  # 20% overlap by default
        try:
            nltk.download('punkt', quiet=True)
        except:
            logger.warning("NLTK punkt download failed - falling back to basic sentence splitting")
            
    def create_chunks(self, text: str, preserve_sentences: bool = True) -> List[Chunk]:
        """Split text into overlapping chunks with semantic awareness"""
        try:
            # Split into sentences using NLTK for better accuracy
            try:
                sentences = sent_tokenize(text)
            except:
                sentences = self._split_into_sentences(text)
                
            chunks = []
            current_chunk = []
            current_length = 0
            start_idx = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                # Check if adding this sentence exceeds chunk size
                if current_length + sentence_length > self.chunk_size and current_chunk:
                    # Create chunk with semantic metadata
                    chunk_text = ' '.join(current_chunk)
                    chunk = Chunk(
                        text=chunk_text,
                        start_idx=start_idx,
                        end_idx=start_idx + len(chunk_text),
                        metadata=self._extract_semantic_metadata(chunk_text),
                        importance_score=self._calculate_importance(chunk_text),
                        topics=self._extract_topics(chunk_text)
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk with overlap
                    overlap_tokens = self._get_overlap_tokens(current_chunk)
                    current_chunk = overlap_tokens + [sentence]
                    current_length = sum(len(t) for t in current_chunk)
                    start_idx = start_idx + len(chunk_text) - len(' '.join(overlap_tokens))
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
                    
            # Add final chunk if exists
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk = Chunk(
                    text=chunk_text,
                    start_idx=start_idx,
                    end_idx=start_idx + len(chunk_text),
                    metadata=self._extract_semantic_metadata(chunk_text),
                    importance_score=self._calculate_importance(chunk_text),
                    topics=self._extract_topics(chunk_text)
                )
                chunks.append(chunk)
                
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}")
            return []
            
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences - fallback method"""
        delimiters = '.!?'
        sentences = []
        current_sentence = []
        
        for char in text:
            current_sentence.append(char)
            if char in delimiters:
                sentences.append(''.join(current_sentence).strip())
                current_sentence = []
                
        if current_sentence:
            sentences.append(''.join(current_sentence).strip())
            
        return sentences
        
    def _extract_semantic_metadata(self, text: str) -> Dict:
        """Extract semantic metadata from chunk text"""
        metadata = {
            "has_citations": bool(re.search(r'\[\d+\]|\(\d+\)', text)),
            "has_figures": bool(re.search(r'fig\.|figure|tab\.|table', text, re.I)),
            "has_equations": bool(re.search(r'\$.*?\$', text)),
            "section_indicators": self._identify_section_indicators(text)
        }
        return metadata
        
    def _identify_section_indicators(self, text: str) -> List[str]:
        """Identify potential section headings or markers"""
        indicators = []
        
        # Common section patterns
        patterns = [
            r'^(introduction|methodology|results|discussion|conclusion)',
            r'^\d+\.\s+\w+',  # Numbered sections
            r'^[A-Z][a-z]+\s*\n'  # Capitalized lines
        ]
        
        for pattern in patterns:
            if matches := re.findall(pattern, text, re.I | re.M):
                indicators.extend(matches)
                
        return indicators
        
    def _calculate_importance(self, text: str) -> float:
        """Calculate importance score for chunk"""
        score = 0.0
        
        # Key indicators of importance
        if re.search(r'(significant|important|key|main|critical|essential)', text, re.I):
            score += 0.2
            
        if re.search(r'(conclude|conclude that|we found|results show)', text, re.I):
            score += 0.3
            
        if re.search(r'(in summary|therefore|thus|hence|consequently)', text, re.I):
            score += 0.2
            
        # Presence of numerical data/statistics
        if re.search(r'\d+\.?\d*%|\d+\.?\d*\s*\+/-', text):
            score += 0.15
            
        # Normalize score
        return min(score, 1.0)
        
    def _extract_topics(self, text: str) -> List[str]:
        """Extract potential topics from chunk text"""
        topics = []
        
        # Extract noun phrases (basic approach)
        noun_phrase_pattern = r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'
        if matches := re.findall(noun_phrase_pattern, text):
            topics.extend(matches[:5])  # Limit to top 5
            
        return topics
        
    def _get_overlap_tokens(self, chunk: List[str], num_tokens: Optional[int] = None) -> List[str]:
        """Get tokens for chunk overlap"""
        if num_tokens is None:
            num_tokens = max(1, int(self.overlap / 10))  # Approximate tokens from overlap size
            
        return chunk[-num_tokens:] if len(chunk) > num_tokens else chunk
        
    def optimize_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Optimize chunks for better processing"""
        optimized_chunks = []
        
        # Sort chunks by importance score
        chunks.sort(key=lambda x: x.importance_score, reverse=True)
        
        for chunk in chunks:
            # Analyze chunk complexity
            complexity = self._analyze_complexity(chunk.text)
            
            # Split complex chunks if needed
            if complexity > 0.8 and len(chunk.text) > self.chunk_size / 2:
                sub_chunks = self.create_chunks(chunk.text, preserve_sentences=True)
                optimized_chunks.extend(sub_chunks)
            else:
                optimized_chunks.append(chunk)
                
        return optimized_chunks
        
    def _analyze_complexity(self, text: str) -> float:
        """Analyze text complexity"""
        # Simple complexity score based on sentence length and word length
        words = text.split()
        if not words:
            return 0
            
        avg_word_length = np.mean([len(word) for word in words])
        sentences = self._split_into_sentences(text)
        avg_sentence_length = np.mean([len(sentence.split()) for sentence in sentences]) if sentences else 0
        
        # Consider technical indicators
        technical_terms = len(re.findall(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b', text))
        equation_count = len(re.findall(r'\$.*?\$', text))
        
        # Normalize scores
        word_score = min(avg_word_length / 10, 1)  # Assume max avg word length of 10
        sentence_score = min(avg_sentence_length / 30, 1)  # Assume max avg sentence length of 30
        technical_score = min(technical_terms / 10, 1)  # Normalize technical term count
        equation_score = min(equation_count / 5, 1)  # Normalize equation count
        
        # Weighted average
        return (word_score * 0.3 + sentence_score * 0.3 + 
                technical_score * 0.2 + equation_score * 0.2)
        
    def merge_chunks(self, chunks: List[Chunk], max_size: int = None) -> List[Chunk]:
        """Merge small chunks while respecting max size and semantic coherence"""
        if not chunks:
            return []
            
        max_size = max_size or self.chunk_size * 2
        merged = []
        current = chunks[0]
        
        for chunk in chunks[1:]:
            combined_length = len(current.text) + len(chunk.text)
            
            # Check semantic compatibility
            topic_overlap = bool(set(current.topics or []) & set(chunk.topics or []))
            similar_importance = abs(current.importance_score - chunk.importance_score) < 0.3
            
            if (combined_length <= max_size and 
                (topic_overlap or similar_importance)):
                # Merge chunks
                current = Chunk(
                    text=current.text + " " + chunk.text,
                    start_idx=current.start_idx,
                    end_idx=chunk.end_idx,
                    references={**(current.references or {}), **(chunk.references or {})},
                    metadata=self._merge_metadata(current.metadata, chunk.metadata),
                    importance_score=max(current.importance_score, chunk.importance_score),
                    topics=list(set((current.topics or []) + (chunk.topics or [])))
                )
            else:
                merged.append(current)
                current = chunk
                
        merged.append(current)
        return merged
        
    def _merge_metadata(self, meta1: Dict, meta2: Dict) -> Dict:
        """Merge metadata from two chunks"""
        if not meta1 or not meta2:
            return meta1 or meta2 or {}
            
        return {
            "has_citations": meta1.get("has_citations", False) or meta2.get("has_citations", False),
            "has_figures": meta1.get("has_figures", False) or meta2.get("has_figures", False),
            "has_equations": meta1.get("has_equations", False) or meta2.get("has_equations", False),
            "section_indicators": list(set(
                meta1.get("section_indicators", []) + meta2.get("section_indicators", [])
            ))
        }
