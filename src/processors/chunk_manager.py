"""Manage text chunking for optimal processing using LlamaIndex"""

from typing import List, Dict, Optional, Any
import numpy as np
from dataclasses import dataclass
import logging
import re
from nltk.tokenize import sent_tokenize
import nltk
import json
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo

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
    relationships: List[Dict] = None  # Store relationships with other chunks

    def __post_init__(self):
        """Initialize after creation"""
        if self.topics is not None and not isinstance(self.topics, set):
            self.topics = set(self.topics)
        if self.relationships is None:
            self.relationships = []
        if self.metadata is None:
            self.metadata = {}
        if self.references is None:
            self.references = {}

    def to_json(self) -> str:
        """Convert chunk to JSON string"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'Chunk':
        """Create chunk from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary"""
        return {
            "text": self.text,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "references": self.references,
            "metadata": self.metadata,
            "importance_score": float(self.importance_score) if isinstance(self.importance_score, np.float32) else self.importance_score,
            "topics": list(self.topics) if self.topics is not None else None,
            "relationships": self.relationships
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        """Create chunk from dictionary"""
        if data.get("topics") is not None:
            data["topics"] = set(data["topics"])
        return cls(**data)

    def to_node(self) -> TextNode:
        """Convert to LlamaIndex TextNode"""
        return TextNode(
            text=self.text,
            metadata={
                "start_idx": self.start_idx,
                "end_idx": self.end_idx,
                "importance_score": self.importance_score,
                "topics": list(self.topics) if self.topics else [],
                **self.metadata
            }
        )

    @classmethod
    def from_node(cls, node: TextNode) -> 'Chunk':
        """Create from LlamaIndex TextNode"""
        metadata = node.metadata or {}
        return cls(
            text=node.text,
            start_idx=metadata.get("start_idx", 0),
            end_idx=metadata.get("end_idx", len(node.text)),
            metadata={k: v for k, v in metadata.items() 
                     if k not in ["start_idx", "end_idx", "importance_score", "topics"]},
            importance_score=metadata.get("importance_score", 0.0),
            topics=set(metadata.get("topics", []))
        )

class ChunkManager:
    """Manage text chunking and processing using LlamaIndex"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.chunk_size = settings.chunk_size
        self.overlap = settings.overlap or int(self.chunk_size * 0.2)
        
        # Initialize NLTK
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
        except:
            logger.warning("NLTK download failed - falling back to basic processing")
            
    def create_chunks(self, text: str, preserve_sentences: bool = True) -> List[Chunk]:
        """Create chunks using LlamaIndex with enhanced processing"""
        try:
            # Create LlamaIndex document
            doc = Document(text=text)
            
            # Configure sentence splitter
            parser = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                paragraph_separator="\n\n",
                secondary_chunking_regex="(?<=\\. )",
                tokenizer=sent_tokenize
            )
            
            # Get nodes with relationships
            nodes = parser.get_nodes_from_documents([doc])
            
            # Convert to chunks while preserving relationships
            chunks = []
            node_map = {}  # Map node IDs to chunks
            
            for node in nodes:
                chunk = Chunk.from_node(node)
                chunks.append(chunk)
                node_map[node.node_id] = chunk
                
            # Process relationships
            for i, node in enumerate(nodes):
                chunk = chunks[i]
                
                # Add relationships from node
                if hasattr(node, 'relationships'):
                    for rel in node.relationships:
                        if isinstance(rel, NodeRelationship):
                            related_chunk = node_map.get(rel.node_id)
                            if related_chunk:
                                chunk.relationships.append({
                                    "chunk_id": chunks.index(related_chunk),
                                    "type": rel.type_,
                                    "metadata": rel.metadata
                                })
                
                # Analyze chunk content
                self._enhance_chunk(chunk)
                
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}")
            return self._fallback_chunking(text)
            
    def _enhance_chunk(self, chunk: Chunk):
        """Enhance chunk with additional analysis"""
        # Extract semantic metadata
        chunk.metadata.update(self._extract_semantic_metadata(chunk.text))
        
        # Calculate importance score
        chunk.importance_score = self._calculate_importance(chunk.text)
        
        # Extract topics
        chunk.topics = set(self._extract_topics(chunk.text))
        
    def _extract_semantic_metadata(self, text: str) -> Dict:
        """Extract semantic metadata from chunk text"""
        metadata = {
            "has_citations": bool(re.search(r'\[\d+\]|\(\d+\)', text)),
            "has_figures": bool(re.search(r'fig\.|figure|tab\.|table', text, re.I)),
            "has_equations": bool(re.search(r'\$.*?\$', text)),
            "section_indicators": self._identify_section_indicators(text),
            "named_entities": self._extract_named_entities(text)
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
        
    def _extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using NLTK"""
        try:
            tokens = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            chunks = nltk.ne_chunk(pos_tags)
            
            entities = {
                "PERSON": [],
                "ORGANIZATION": [],
                "GPE": [],  # Geo-Political Entities
                "OTHER": []
            }
            
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    entity_text = ' '.join(c[0] for c in chunk.leaves())
                    if chunk.label() in entities:
                        entities[chunk.label()].append(entity_text)
                    else:
                        entities["OTHER"].append(entity_text)
                        
            return entities
            
        except Exception as e:
            logger.warning(f"Named entity extraction failed: {str(e)}")
            return {}
        
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
            
        # Named entities
        entities = self._extract_named_entities(text)
        if sum(len(v) for v in entities.values()) > 0:
            score += 0.15
            
        return min(score, 1.0)
        
    def _extract_topics(self, text: str) -> List[str]:
        """Extract potential topics from chunk text"""
        topics = set()
        
        # Extract noun phrases
        try:
            tokens = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            
            # Find noun phrases
            i = 0
            while i < len(pos_tags):
                if pos_tags[i][1].startswith('JJ'):  # Adjective
                    j = i + 1
                    while j < len(pos_tags) and pos_tags[j][1].startswith('NN'):  # Noun
                        j += 1
                    if j > i + 1:
                        topics.add(' '.join(token for token, _ in pos_tags[i:j]))
                elif pos_tags[i][1].startswith('NN'):  # Noun
                    j = i + 1
                    while j < len(pos_tags) and pos_tags[j][1].startswith('NN'):
                        j += 1
                    if j > i + 1:
                        topics.add(' '.join(token for token, _ in pos_tags[i:j]))
                i += 1
                
        except Exception as e:
            logger.warning(f"Topic extraction failed: {str(e)}")
            
        return list(topics)
        
    def _fallback_chunking(self, text: str) -> List[Chunk]:
        """Fallback method for chunking when LlamaIndex fails"""
        chunks = []
        sentences = sent_tokenize(text) if nltk.data.find('tokenizers/punkt') else text.split('. ')
        
        current_chunk = []
        current_length = 0
        start_idx = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
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
                overlap_text = chunk_text[-self.overlap:] if self.overlap > 0 else ""
                current_chunk = [overlap_text] if overlap_text else []
                current_length = len(overlap_text)
                start_idx = start_idx + len(chunk_text) - len(overlap_text)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
                
        # Add final chunk
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
        
    def optimize_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Optimize chunks for better processing"""
        # Sort by importance
        chunks.sort(key=lambda x: x.importance_score, reverse=True)
        
        # Group related chunks
        chunk_groups = self._group_related_chunks(chunks)
        
        # Flatten groups while preserving relationships
        optimized_chunks = []
        for group in chunk_groups:
            if len(group) == 1:
                optimized_chunks.extend(group)
            else:
                merged = self._merge_chunk_group(group)
                optimized_chunks.append(merged)
                
        return optimized_chunks
        
    def _group_related_chunks(self, chunks: List[Chunk]) -> List[List[Chunk]]:
        """Group related chunks based on topics and relationships"""
        groups = []
        used_chunks = set()
        
        for chunk in chunks:
            if chunk in used_chunks:
                continue
                
            group = [chunk]
            used_chunks.add(chunk)
            
            # Find related chunks
            for other in chunks:
                if other in used_chunks:
                    continue
                    
                # Check topic overlap
                topic_overlap = bool(chunk.topics & other.topics if chunk.topics and other.topics else False)
                
                # Check explicit relationships
                has_relationship = any(
                    rel["chunk_id"] == chunks.index(other)
                    for rel in chunk.relationships
                )
                
                if topic_overlap or has_relationship:
                    group.append(other)
                    used_chunks.add(other)
                    
            groups.append(group)
            
        return groups
        
    def _merge_chunk_group(self, group: List[Chunk]) -> Chunk:
        """Merge a group of related chunks"""
        if not group:
            return None
            
        # Combine text and metadata
        combined_text = " ".join(chunk.text for chunk in group)
        combined_metadata = {}
        combined_topics = set()
        combined_relationships = []
        max_importance = 0.0
        
        for chunk in group:
            # Merge metadata
            for key, value in chunk.metadata.items():
                if key not in combined_metadata:
                    combined_metadata[key] = value
                elif isinstance(value, list):
                    combined_metadata[key] = list(set(combined_metadata[key] + value))
                    
            # Merge topics
            if chunk.topics:
                combined_topics.update(chunk.topics)
                
            # Update importance score
            max_importance = max(max_importance, chunk.importance_score)
            
            # Preserve relationships to chunks outside the group
            for rel in chunk.relationships:
                if not any(chunks.index(c) == rel["chunk_id"] for c in group):
                    combined_relationships.append(rel)
                    
        return Chunk(
            text=combined_text,
            start_idx=group[0].start_idx,
            end_idx=group[-1].end_idx,
            metadata=combined_metadata,
            importance_score=max_importance,
            topics=combined_topics,
            relationships=combined_relationships
        )
