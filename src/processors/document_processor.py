from pathlib import Path
from typing import Dict, Any, Optional, List
from PyPDF2 import PdfReader
import re
import numpy as np
from collections import defaultdict
from ..config.settings import Settings
from .text_cleaner import TextCleaner
from .chunk_manager import ChunkManager, Chunk

class DocumentProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.text_cleaner = TextCleaner()
        self.chunk_manager = ChunkManager(settings)
        
    def process(
        self,
        file_path: Path
    ) -> Dict[str, Any]:
        """Process document and prepare for script generation"""
        try:
            # Extract text and metadata
            text, metadata = self._extract_from_pdf(file_path)
            
            # Clean text
            cleaned_text = self.text_cleaner.clean_text(text)
            
            # Create chunks with specified size
            chunks = self.chunk_manager.create_chunks(cleaned_text["text"])
            
            # Extract document structure
            structure = self._analyze_document_structure(chunks)
            
            # Optimize chunks based on content
            optimized_chunks = self.chunk_manager.optimize_chunks(chunks)
            
            # Group related chunks
            chunk_groups = self._group_related_chunks(optimized_chunks)
            
            return {
                "title": metadata.get("title", file_path.stem),
                "text": cleaned_text["text"],
                "chunks": [chunk.text for chunk in optimized_chunks],
                "structure": structure,
                "chunk_groups": chunk_groups,
                "metadata": {
                    **metadata,
                    "references": cleaned_text.get("references", []),
                    "total_chunks": len(optimized_chunks),
                    "key_topics": self._extract_key_topics(optimized_chunks),
                    "document_type": self._identify_document_type(text, metadata)
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"Document processing error: {str(e)}")
            
    def _extract_from_pdf(self, file_path: Path) -> tuple[str, Dict[str, Any]]:
        """Extract text and metadata from PDF"""
        try:
            reader = PdfReader(str(file_path))
            
            # Extract metadata
            metadata = {
                "title": reader.metadata.get("/Title", file_path.stem) if reader.metadata else file_path.stem,
                "author": reader.metadata.get("/Author", "Unknown") if reader.metadata else "Unknown",
                "subject": reader.metadata.get("/Subject", "") if reader.metadata else "",
                "keywords": reader.metadata.get("/Keywords", "").split(",") if reader.metadata else [],
                "page_count": len(reader.pages)
            }
            
            # Extract text with structure preservation
            text = ""
            section_markers = []
            current_page = 1
            
            for page in reader.pages:
                page_text = page.extract_text() or ""
                
                # Identify section breaks
                if section_match := re.search(r'^(?:\d+\.)?\s*[A-Z][^.!?]*$', 
                                           page_text, re.M):
                    section_markers.append({
                        "title": section_match.group().strip(),
                        "page": current_page
                    })
                    
                text += page_text + "\n\n"
                current_page += 1
                
            metadata["section_markers"] = section_markers
            return text, metadata
            
        except Exception as e:
            raise RuntimeError(f"PDF extraction error: {str(e)}")
            
    def _analyze_document_structure(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Analyze document structure from chunks"""
        structure = {
            "sections": [],
            "hierarchy": defaultdict(list),
            "key_points": []
        }
        
        current_section = None
        section_content = []
        
        for chunk in chunks:
            # Check for section indicators
            if chunk.metadata and chunk.metadata.get("section_indicators"):
                if current_section and section_content:
                    structure["sections"].append({
                        "title": current_section,
                        "content": section_content,
                        "importance": np.mean([c.importance_score for c in section_content])
                    })
                current_section = chunk.metadata["section_indicators"][0]
                section_content = [chunk]
            elif current_section:
                section_content.append(chunk)
                
            # Track key points
            if chunk.importance_score > 0.7:
                structure["key_points"].append({
                    "text": chunk.text,
                    "score": chunk.importance_score,
                    "topics": chunk.topics
                })
                
        # Add final section
        if current_section and section_content:
            structure["sections"].append({
                "title": current_section,
                "content": section_content,
                "importance": np.mean([c.importance_score for c in section_content])
            })
            
        return structure
        
    def _group_related_chunks(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """Group related chunks based on topics and content"""
        groups = []
        current_group = {
            "chunks": [],
            "topics": set(),
            "importance": 0
        }
        
        for chunk in chunks:
            # Check if chunk fits current group
            topic_overlap = bool(current_group["topics"] & set(chunk.topics or []))
            
            if topic_overlap and len(current_group["chunks"]) < 5:
                current_group["chunks"].append(chunk)
                current_group["topics"].update(chunk.topics or [])
                current_group["importance"] = max(
                    current_group["importance"],
                    chunk.importance_score
                )
            else:
                if current_group["chunks"]:
                    groups.append(current_group)
                current_group = {
                    "chunks": [chunk],
                    "topics": set(chunk.topics or []),
                    "importance": chunk.importance_score
                }
                
        if current_group["chunks"]:
            groups.append(current_group)
            
        return sorted(groups, key=lambda x: x["importance"], reverse=True)
        
    def _extract_key_topics(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """Extract and rank key topics from chunks"""
        topic_freq = defaultdict(int)
        topic_chunks = defaultdict(list)
        
        for chunk in chunks:
            if chunk.topics:
                for topic in chunk.topics:
                    topic_freq[topic] += 1
                    topic_chunks[topic].append(chunk)
                    
        # Rank topics by frequency and chunk importance
        ranked_topics = []
        for topic, freq in topic_freq.items():
            avg_importance = np.mean([
                c.importance_score for c in topic_chunks[topic]
            ])
            ranked_topics.append({
                "topic": topic,
                "frequency": freq,
                "importance": avg_importance,
                "score": freq * avg_importance
            })
            
        return sorted(ranked_topics, key=lambda x: x["score"], reverse=True)
        
    def _identify_document_type(self, text: str, metadata: Dict) -> str:
        """Identify the type of academic document"""
        # Check for common paper sections
        has_abstract = bool(re.search(r'\b(abstract|summary)\b', text[:1000], re.I))
        has_references = bool(re.search(r'\b(references|bibliography)\b', text[-5000:], re.I))
        has_methodology = bool(re.search(r'\b(methodology|methods|experimental setup)\b', text, re.I))
        
        # Check for specific markers
        if has_abstract and has_methodology and has_references:
            return "research_paper"
        elif bool(re.search(r'\b(thesis|dissertation)\b', text, re.I)):
            return "thesis"
        elif bool(re.search(r'\b(review|survey)\b', text[:2000], re.I)):
            return "review_paper"
        else:
            return "general_academic"
            
    def validate_pdf(self, file_path: Path) -> bool:
        """Validate PDF file"""
        try:
            reader = PdfReader(str(file_path))
            return len(reader.pages) > 0
        except:
            return False
