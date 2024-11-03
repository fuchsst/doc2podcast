"""Document processing module with enhanced analysis capabilities"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from PyPDF2 import PdfReader
import json
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from crewai import LLM

from ..config.settings import Settings
from .text_cleaner import TextCleaner
from .chunk_manager import ChunkManager, Chunk
from ..pipeline.analysis_agents import AnalysisAgents
from ..pipeline.base import DocumentAnalyzer, AnalysisContext
from ..utils.callback_handler import PipelineCallback
from ..utils.cache_manager import cache_manager
from ..pipeline.config import ProcessingConfig, AnalysisConfig, AgentConfig

class DocumentProcessor(DocumentAnalyzer):
    """Processes documents with enhanced analysis capabilities"""
    
    def __init__(
        self,
        settings: Settings,
        config: Optional[ProcessingConfig] = None
    ):
        self.settings = settings
        self.config = config or ProcessingConfig.from_settings(settings)
        
        # Initialize components
        self.text_cleaner = TextCleaner()
        self.chunk_manager = ChunkManager(settings)
        
        # Initialize LLM
        self.llm = LLM(
            model=settings.text_generation_config.default,
            temperature=settings.text_generation_config.temperature,
            max_tokens=settings.text_generation_config.max_new_tokens,
            api_key=settings.ANTHROPIC_API_KEY
        )
        
        # Initialize analysis components with configs
        analysis_config = self.config.analysis_config or AnalysisConfig.from_settings(settings)
        agent_config = self.config.agent_config or AgentConfig.from_settings(settings)
        self.analysis_agents = AnalysisAgents(self.llm, agent_config)
    
    def analyze(self, context: AnalysisContext) -> Dict[str, Any]:
        """Implement abstract analyze method from DocumentAnalyzer"""
        # Create a temporary file path for the context text
        temp_file = Path("temp_document.txt")
        try:
            # Write context text to temporary file
            temp_file.write_text(context.text)
            
            # Process the file using existing process method
            result = self.process(temp_file, None)
            
            # Update result with context metadata
            if isinstance(result, dict):
                result["metadata"] = {
                    **(result.get("metadata", {})),
                    **context.metadata
                }
            
            return result
            
        finally:
            # Clean up temporary file
            if temp_file.exists():
                temp_file.unlink()
        
    def process(
        self,
        file_path: Path,
        callback: Optional[PipelineCallback] = None
    ) -> Dict[str, Any]:
        """Process document with enhanced analysis"""
        try:
            # Generate file hash
            file_hash = cache_manager.get_file_hash(Path(file_path))
            
            # Create analysis context
            context = AnalysisContext(
                text="",  # Will be populated from PDF
                metadata={},  # Will be populated from PDF
                settings=self.settings,
                cache_key=file_hash
            )
            
            # Extract text and metadata
            if callback:
                callback.on_document_processing(10, "Extracting text from document...")
            
            if self.config.cache_enabled:
                extraction_cache = cache_manager.load_json_cache(file_hash, "extraction")
                if extraction_cache:
                    context.text = extraction_cache["text"]
                    context.metadata = extraction_cache["metadata"]
                else:
                    context.text, context.metadata = self._extract_from_pdf(file_path)
                    cache_manager.cache_json(file_hash, "extraction", {
                        "text": context.text,
                        "metadata": context.metadata
                    })
            else:
                context.text, context.metadata = self._extract_from_pdf(file_path)
            
            # Clean text
            if callback:
                callback.on_document_processing(20, "Cleaning text...")
            
            if self.config.cache_enabled:
                cleaning_cache = cache_manager.load_json_cache(file_hash, "cleaning")
                if cleaning_cache:
                    cleaned_text = cleaning_cache
                else:
                    cleaned_text = self.text_cleaner.clean_text(context.text)
                    cache_manager.cache_json(file_hash, "cleaning", cleaned_text)
            else:
                cleaned_text = self.text_cleaner.clean_text(context.text)
            
            context.text = cleaned_text["text"]
            context.metadata["references"] = cleaned_text.get("references", [])
            
            # Create chunks using LlamaIndex
            if callback:
                callback.on_document_processing(30, "Creating document chunks...")
            
            if self.config.cache_enabled:
                chunks_cache = cache_manager.load_json_cache(file_hash, "chunks")
                if chunks_cache:
                    chunks = [Chunk.from_dict(chunk_data) for chunk_data in chunks_cache]
                else:
                    chunks = self._create_chunks(context.text)
                    cache_manager.cache_json(file_hash, "chunks", [
                        chunk.to_dict() for chunk in chunks
                    ])
            else:
                chunks = self._create_chunks(context.text)
            
            # Process chunks
            if callback:
                callback.on_document_processing(40, "Analyzing chunks...")
            
            chunk_results = []
            total_chunks = len(chunks)
            for i, chunk in enumerate(chunks):
                if callback:
                    progress = 40 + (30 * (i / total_chunks))
                    callback.on_document_processing(
                        progress,
                        f"Processing chunk {i+1}/{total_chunks}"
                    )
                
                # Create context for chunk
                chunk_context = AnalysisContext(
                    text=chunk.text,
                    metadata=chunk.metadata,
                    settings=self.settings,
                    cache_key=f"{file_hash}_chunk_{i}"
                )
                
                # Analyze chunk
                if self.config.cache_enabled:
                    chunk_cache = cache_manager.load_json_cache(chunk_context.cache_key, "analysis")
                    if chunk_cache:
                        result = chunk_cache
                    else:
                        result = self.analysis_agents.analyze(chunk_context)
                        cache_manager.cache_json(chunk_context.cache_key, "analysis", result)
                else:
                    result = self.analysis_agents.analyze(chunk_context)
                
                chunk_results.append(result)
            
            # Consolidate results
            if callback:
                callback.on_document_processing(80, "Consolidating analysis results...")
            
            if self.config.cache_enabled:
                consolidation_cache = cache_manager.load_json_cache(file_hash, "consolidation")
                if consolidation_cache:
                    consolidated_results = consolidation_cache
                else:
                    consolidated_results = self.analysis_agents.consolidate(chunk_results)
                    cache_manager.cache_json(file_hash, "consolidation", consolidated_results)
            else:
                consolidated_results = self.analysis_agents.consolidate(chunk_results)
            
            # Add metadata
            consolidated_results["metadata"] = {
                **context.metadata,
                "total_chunks": len(chunks)
            }
            
            if callback:
                callback.on_document_processing(100, "Document processing complete")
            
            return consolidated_results
            
        except Exception as e:
            if callback:
                callback.on_error("DOCUMENT_PROCESSING", str(e))
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
                "page_count": len(reader.pages),
                "source_path": str(file_path)
            }
            
            # Extract text
            text = ""
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n\n"
            
            return text, metadata
            
        except Exception as e:
            raise RuntimeError(f"PDF extraction error: {str(e)}")
            
    def _create_chunks(self, text: str) -> List[Chunk]:
        """Create document chunks using LlamaIndex"""
        try:
            # Create LlamaIndex document
            doc = Document(text=text)
            parser = SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.overlap
            )
            
            # Get nodes
            nodes = parser.get_nodes_from_documents([doc])
            
            # Convert to chunks
            chunks = []
            for i, node in enumerate(nodes):
                chunk = Chunk(
                    text=node.text,
                    start_idx=node.start_char_idx if hasattr(node, 'start_char_idx') else i * self.config.chunk_size,
                    end_idx=node.end_char_idx if hasattr(node, 'end_char_idx') else (i + 1) * self.config.chunk_size,
                    metadata={"node_info": node.metadata} if hasattr(node, 'metadata') else {}
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            raise RuntimeError(f"Chunk creation error: {str(e)}")
            
    def validate_pdf(self, file_path: Path) -> bool:
        """Validate PDF file"""
        try:
            reader = PdfReader(str(file_path))
            return len(reader.pages) > 0
        except:
            return False
