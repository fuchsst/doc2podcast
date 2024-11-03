"""Analysis tools implementing SOLID principles"""

import re
from typing import Dict, Any, List, Optional
import numpy as np
import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from pydantic import ConfigDict

from .base import AnalysisTool, AnalysisContext
from .config import AnalysisConfig
from .schemas import (
    TopicAnalysisSchema,
    InsightAnalysisSchema,
    QuestionAnalysisSchema
)
from .prompts import PromptTemplates

# Initialize NLTK components
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
except:
    pass

class TextAnalyzer:
    """Base text analysis functionality"""
    
    @staticmethod
    def extract_named_entities(text: str) -> Dict[str, List[Dict]]:
        """Extract and categorize named entities"""
        try:
            tokens = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            chunks = nltk.ne_chunk(pos_tags)
            
            entities = {
                "PERSON": [],
                "ORGANIZATION": [],
                "GPE": [],
                "OTHER": []
            }
            
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    entity_text = ' '.join(c[0] for c in chunk.leaves())
                    context = TextAnalyzer.get_entity_context(entity_text, text)
                    entity_info = {
                        "text": entity_text,
                        "context": context,
                        "frequency": text.lower().count(entity_text.lower())
                    }
                    if chunk.label() in entities:
                        entities[chunk.label()].append(entity_info)
                    else:
                        entities["OTHER"].append(entity_info)
                        
            return entities
            
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def get_entity_context(entity: str, text: str, window: int = 2) -> List[str]:
        """Get context for named entities"""
        sentences = nltk.sent_tokenize(text)
        context = []
        
        for sentence in sentences:
            if entity.lower() in sentence.lower():
                context.append(sentence)
                
        return context[:window]
    
    @staticmethod
    def extract_noun_phrases(text: str) -> List[str]:
        """Extract noun phrases from text"""
        try:
            tokens = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            
            phrases = []
            current_phrase = []
            
            for word, tag in pos_tags:
                if tag.startswith('NN'):
                    current_phrase.append(word)
                elif current_phrase:
                    phrases.append(' '.join(current_phrase))
                    current_phrase = []
                    
            if current_phrase:
                phrases.append(' '.join(current_phrase))
                
            return phrases
            
        except Exception:
            return []

class TopicAnalysisTool(AnalysisTool):
    """Tool for comprehensive topic analysis"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    config: Optional[AnalysisConfig] = None
    text_analyzer: TextAnalyzer = None
    
    def __init__(self, llm, config: Optional[AnalysisConfig] = None):
        super().__init__(
            name="Topic Analysis Tool",
            description="""Analyzes document topics and themes.
            
            Args:
                text (str): The document content to analyze. Must be a non-empty string containing at least one paragraph.
            
            Returns:
                str: A JSON string containing:
                {
                    "main_topics": [{"name": str, "description": str, "subtopics": [str], "significance": str}],
                    "topic_hierarchy": {"levels": [str], "relationships": [str]},
                    "named_entities": {"PERSON": [str], "ORGANIZATION": [str], "GPE": [str]},
                    "key_terms": [{"term": str, "context": str, "importance": float}]
                }
            
            Use this tool when you need to:
            - Understand the main themes and topics in a document
            - Find relationships between different topics
            - Identify key terms and their context
            - Extract named entities and their roles
            
            Do not use this tool for:
            - Analyzing very short texts (less than a paragraph)
            - Code or structured data analysis
            - Real-time text processing"""
        )
        self._llm = llm
        self.config = config or AnalysisConfig(chunk_size=512)
        self.text_analyzer = TextAnalyzer()
        
    @property
    def llm(self):
        return self._llm
        
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze topics in text"""
        # Create document chunks
        doc = Document(text=text)
        parser = SentenceSplitter(chunk_size=self.config.chunk_size)
        nodes = parser.get_nodes_from_documents([doc])
        texts = [node.text for node in nodes]
        
        # Perform TF-IDF analysis
        tfidf_results = self._analyze_tfidf(texts)
        
        # Perform topic modeling
        topic_results = self._analyze_topics(tfidf_results["tfidf_matrix"], tfidf_results["feature_names"])
        
        # Extract named entities
        entities = self.text_analyzer.extract_named_entities(text)
        
        return {
            "tfidf_analysis": tfidf_results["terms"],
            "topic_modeling": topic_results,
            "named_entities": entities
        }
        
    def enhance_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance results with LLM analysis"""
        prompt = PromptTemplates.get_topic_analysis_prompt({"results": results})
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.llm.call(messages)
            enhanced = json.loads(response)
            return TopicAnalysisSchema(**enhanced)
        except Exception as e:
            return {
                "error": f"Failed to enhance results: {str(e)}",
                "original_results": results
            }
            
    def _analyze_tfidf(self, texts: List[str]) -> Dict[str, Any]:
        """Perform TF-IDF analysis"""
        vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        return {
            "tfidf_matrix": tfidf_matrix,
            "feature_names": feature_names,
            "terms": [
                {
                    "term": term,
                    "score": float(score),
                    "context": self._get_term_context(term, texts)
                }
                for term, score in sorted(
                    zip(feature_names, tfidf_matrix.toarray()[0]),
                    key=lambda x: x[1],
                    reverse=True
                )[:20]
            ]
        }
        
    def _analyze_topics(self, tfidf_matrix: Any, feature_names: List[str]) -> Dict[str, Any]:
        """Perform topic modeling"""
        lda = LatentDirichletAllocation(
            n_components=self.config.num_topics,
            random_state=42
        )
        lda_output = lda.fit_transform(tfidf_matrix)
        
        return {
            "topics": [
                {
                    "id": topic_idx,
                    "terms": [
                        {
                            "term": term,
                            "weight": float(score)
                        }
                        for term, score in self._get_topic_terms(topic, feature_names)
                    ],
                    "weight": float(lda_output[0][topic_idx])
                }
                for topic_idx, topic in enumerate(lda.components_)
            ],
            "document_topics": [
                {
                    "node_idx": idx,
                    "topic_weights": [float(weight) for weight in weights]
                }
                for idx, weights in enumerate(lda_output)
            ]
        }
        
    def _get_term_context(self, term: str, texts: List[str]) -> List[Dict]:
        """Get context for a term"""
        context = []
        for text in texts:
            if term.lower() in text.lower():
                sentences = nltk.sent_tokenize(text)
                for sentence in sentences:
                    if term.lower() in sentence.lower():
                        context.append({
                            "text": sentence,
                            "position": text.index(sentence)
                        })
        return context[:3]
        
    def _get_topic_terms(self, topic: np.ndarray, feature_names: List[str]) -> List[tuple]:
        """Get terms for a topic"""
        return sorted(
            zip(feature_names, topic),
            key=lambda x: x[1],
            reverse=True
        )[:10]

class InsightAnalysisTool(AnalysisTool):
    """Tool for extracting and analyzing insights"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    config: Optional[AnalysisConfig] = None
    text_analyzer: TextAnalyzer = None
    
    def __init__(self, llm, config: Optional[AnalysisConfig] = None):
        super().__init__(
            name="Insight Analysis Tool",
            description="""Extracts and analyzes key insights from documents.
            
            Args:
                text (str): The document content to analyze. Can be any type of document.
            
            Returns:
                str: A JSON string containing:
                {
                    "approaches": {
                        "main_approaches": [{"name": str, "description": str, "context": str}],
                        "implementation": [{"technique": str, "description": str, "key_points": [str]}]
                    },
                    "key_points": [{"description": str, "support": str, "importance": str}],
                    "notable_elements": [{"description": str, "uniqueness": str, "applications": [str]}]
                }
            
            Use this tool when you need to:
            - Extract key insights and approaches
            - Understand implementation details
            - Find notable elements and features
            - Analyze practical applications
            
            Do not use this tool for:
            - Basic text summarization
            - Opinion or sentiment analysis
            - Simple text queries"""
        )
        self._llm = llm
        self.config = config or AnalysisConfig(chunk_size=512)
        self.text_analyzer = TextAnalyzer()
        
    @property
    def llm(self):
        return self._llm
        
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze insights in text"""
        # Create document chunks
        doc = Document(text=text)
        parser = SentenceSplitter(chunk_size=self.config.chunk_size)
        nodes = parser.get_nodes_from_documents([doc])
        
        # Extract sentences
        sentences = []
        for node in nodes:
            sentences.extend(nltk.sent_tokenize(node.text))
            
        # Score sentences
        sentence_scores = self._score_sentences(sentences)
        
        # Extract insights
        return {
            "approaches": {
                "main_approaches": self._extract_approaches(sentences),
                "implementation": self._extract_implementation(sentences)
            },
            "key_points": self._extract_key_points(sentences, sentence_scores),
            "notable_elements": self._extract_notable_elements(sentences, sentence_scores)
        }
        
    def enhance_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance results with LLM analysis"""
        prompt = PromptTemplates.get_insight_analysis_prompt({"results": results})
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.llm.call(messages)
            enhanced = json.loads(response)
            return InsightAnalysisSchema(**enhanced)
        except Exception as e:
            return {
                "error": f"Failed to enhance results: {str(e)}",
                "original_results": results
            }
            
    def _score_sentences(self, sentences: List[str]) -> np.ndarray:
        """Score sentences using TF-IDF"""
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        return tfidf_matrix.toarray().sum(axis=1)
        
    def _extract_approaches(self, sentences: List[str]) -> List[Dict]:
        """Extract approach information"""
        approaches = []
        patterns = [
            r'(?:use|using|utilize|utilizing|implement|implementing)',
            r'(?:approach|technique|method|strategy|solution|tool)',
            r'(?:create|build|develop|design|structure)'
        ]
        
        for sentence in sentences:
            for pattern in patterns:
                if re.search(pattern, sentence, re.I):
                    approaches.append({
                        "text": sentence,
                        "type": "approach",
                        "context": self._get_sentence_context(sentence, sentences)
                    })
                    break
                    
        return approaches
        
    def _extract_implementation(self, sentences: List[str]) -> List[Dict]:
        """Extract implementation details"""
        implementations = []
        patterns = [
            r'(?:implement|configure|setup|install|initialize)',
            r'(?:step|process|procedure|workflow|sequence)',
            r'(?:require|need|must|should|can)'
        ]
        
        for sentence in sentences:
            for pattern in patterns:
                if re.search(pattern, sentence, re.I):
                    implementations.append({
                        "text": sentence,
                        "type": "implementation",
                        "context": self._get_sentence_context(sentence, sentences)
                    })
                    break
                    
        return implementations
        
    def _extract_key_points(
        self,
        sentences: List[str],
        importance_scores: np.ndarray
    ) -> List[Dict]:
        """Extract key points"""
        points = []
        patterns = [
            r'(?:important|key|crucial|essential|significant)',
            r'(?:note|remember|consider|ensure|make sure)',
            r'(?:feature|functionality|capability|option)'
        ]
        
        for sentence, score in zip(sentences, importance_scores):
            if score < self.config.min_importance:
                continue
                
            for pattern in patterns:
                if re.search(pattern, sentence, re.I):
                    points.append({
                        "text": sentence,
                        "importance_score": float(score),
                        "context": self._get_sentence_context(sentence, sentences)
                    })
                    break
                    
        return sorted(points, key=lambda x: x["importance_score"], reverse=True)
        
    def _extract_notable_elements(
        self,
        sentences: List[str],
        importance_scores: np.ndarray
    ) -> List[Dict]:
        """Extract notable elements"""
        elements = []
        patterns = [
            r'(?:unique|special|specific|custom|advanced)',
            r'(?:feature|element|component|aspect|part)',
            r'(?:advantage|benefit|improvement|enhancement)'
        ]
        
        for sentence, score in zip(sentences, importance_scores):
            if score < self.config.min_importance:
                continue
                
            for pattern in patterns:
                if re.search(pattern, sentence, re.I):
                    elements.append({
                        "text": sentence,
                        "importance_score": float(score),
                        "context": self._get_sentence_context(sentence, sentences)
                    })
                    break
                    
        return sorted(elements, key=lambda x: x["importance_score"], reverse=True)
        
    def _get_sentence_context(
        self,
        sentence: str,
        sentences: List[str]
    ) -> List[str]:
        """Get context for a sentence"""
        try:
            idx = sentences.index(sentence)
            start = max(0, idx - self.config.context_window)
            end = min(len(sentences), idx + self.config.context_window + 1)
            return sentences[start:end]
        except ValueError:
            return [sentence]

class ObjectiveAnalysisTool(AnalysisTool):
    """Tool for analyzing document objectives and goals"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    config: Optional[AnalysisConfig] = None
    text_analyzer: TextAnalyzer = None
    
    def __init__(self, llm, config: Optional[AnalysisConfig] = None):
        super().__init__(
            name="Objective Analysis Tool",
            description="""Analyzes document objectives and learning goals.
            
            Args:
                text (str): The document content to analyze. Can be any type of document.
            
            Returns:
                str: A JSON string containing:
                {
                    "main_objectives": {"primary": [{"objective": str, "type": str}], "secondary": [{"objective": str}]},
                    "key_considerations": [{"point": str, "factors": {"technical": [str], "practical": [str]}}],
                    "learning_goals": [{"goal": str, "type": str, "success_criteria": [str]}],
                    "next_steps": [{"step": str, "rationale": str}]
                }
            
            Use this tool when you need to:
            - Identify main objectives and goals
            - Extract key considerations
            - Understand learning outcomes
            - Find implementation steps
            
            Do not use this tool for:
            - General question answering
            - FAQ generation
            - Simple text queries"""
        )
        self._llm = llm
        self.config = config or AnalysisConfig(chunk_size=512)
        self.text_analyzer = TextAnalyzer()
        
    @property
    def llm(self):
        return self._llm
        
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze objectives in text"""
        # Create document chunks
        doc = Document(text=text)
        parser = SentenceSplitter(chunk_size=self.config.chunk_size)
        nodes = parser.get_nodes_from_documents([doc])
        
        # Extract sentences
        sentences = []
        for node in nodes:
            sentences.extend(nltk.sent_tokenize(node.text))
            
        return {
            "main_objectives": {
                "primary": self._extract_primary_objectives(sentences),
                "secondary": self._extract_secondary_objectives(sentences)
            },
            "key_considerations": self._extract_considerations(sentences),
            "learning_goals": self._extract_learning_goals(sentences),
            "next_steps": self._extract_next_steps(sentences)
        }
        
    def enhance_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance results with LLM analysis"""
        prompt = PromptTemplates.get_question_analysis_prompt({"results": results})
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.llm.call(messages)
            enhanced = json.loads(response)
            return QuestionAnalysisSchema(**enhanced)
        except Exception as e:
            return {
                "error": f"Failed to enhance results: {str(e)}",
                "original_results": results
            }
            
    def _extract_primary_objectives(self, sentences: List[str]) -> List[Dict]:
        """Extract primary objectives"""
        objectives = []
        patterns = [
            r'(?:main|primary|key|core) (?:goal|objective|purpose|aim)',
            r'(?:this|the) (?:guide|tutorial|document) (?:will|shows|demonstrates)',
            r'(?:learn|understand|master|grasp) how to'
        ]
        
        for sentence in sentences:
            for pattern in patterns:
                if re.search(pattern, sentence, re.I):
                    objectives.append({
                        "text": sentence,
                        "type": "primary",
                        "context": self._get_sentence_context(sentence, sentences)
                    })
                    break
                    
        return objectives
        
    def _extract_secondary_objectives(self, sentences: List[str]) -> List[Dict]:
        """Extract secondary objectives"""
        objectives = []
        patterns = [
            r'(?:also|additionally|furthermore) (?:learn|understand|see)',
            r'(?:other|additional|more) (?:features|aspects|topics)',
            r'(?:explore|discover|examine) how'
        ]
        
        for sentence in sentences:
            for pattern in patterns:
                if re.search(pattern, sentence, re.I):
                    objectives.append({
                        "text": sentence,
                        "type": "secondary",
                        "context": self._get_sentence_context(sentence, sentences)
                    })
                    break
                    
        return objectives
        
    def _extract_considerations(self, sentences: List[str]) -> List[Dict]:
        """Extract key considerations"""
        considerations = []
        patterns = [
            r'(?:consider|note|remember|keep in mind)',
            r'(?:important|crucial|essential) to',
            r'(?:requirement|prerequisite|dependency)'
        ]
        
        for sentence in sentences:
            for pattern in patterns:
                if re.search(pattern, sentence, re.I):
                    considerations.append({
                        "text": sentence,
                        "type": "consideration",
                        "context": self._get_sentence_context(sentence, sentences)
                    })
                    break
                    
        return considerations
        
    def _extract_learning_goals(self, sentences: List[str]) -> List[Dict]:
        """Extract learning goals"""
        goals = []
        patterns = [
            r'(?:by|after) (?:the end|completing|finishing)',
            r'(?:will|should) (?:be able to|understand|know)',
            r'(?:learn|master|grasp) (?:how|about|why)'
        ]
        
        for sentence in sentences:
            for pattern in patterns:
                if re.search(pattern, sentence, re.I):
                    goals.append({
                        "text": sentence,
                        "type": self._classify_goal(sentence),
                        "context": self._get_sentence_context(sentence, sentences)
                    })
                    break
                    
        return goals
        
    def _extract_next_steps(self, sentences: List[str]) -> List[Dict]:
        """Extract next steps"""
        steps = []
        patterns = [
            r'(?:next|following|subsequent) (?:step|stage|phase)',
            r'(?:then|after that|once|when) you',
            r'(?:finally|lastly|in conclusion)'
        ]
        
        for sentence in sentences:
            for pattern in patterns:
                if re.search(pattern, sentence, re.I):
                    steps.append({
                        "text": sentence,
                        "type": "next_step",
                        "context": self._get_sentence_context(sentence, sentences)
                    })
                    break
                    
        return steps
        
    def _classify_goal(self, text: str) -> str:
        """Classify goal type"""
        if re.search(r'(?:understand|comprehend|grasp|know)', text, re.I):
            return "conceptual"
        elif re.search(r'(?:implement|create|build|develop|code)', text, re.I):
            return "practical"
        elif re.search(r'(?:configure|setup|install|deploy)', text, re.I):
            return "technical"
        else:
            return "general"
            
    def _get_sentence_context(
        self,
        sentence: str,
        sentences: List[str]
    ) -> List[str]:
        """Get context for a sentence"""
        try:
            idx = sentences.index(sentence)
            start = max(0, idx - self.config.context_window)
            end = min(len(sentences), idx + self.config.context_window + 1)
            return sentences[start:end]
        except ValueError:
            return [sentence]
