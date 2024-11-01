"""Custom tools for document analysis tasks combining algorithms with LLM reasoning"""

from typing import List, Dict, Any
from crewai_tools import BaseTool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import json
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

class EnhancedAnalysisTool(BaseTool):
    """Base class for enhanced analysis tools combining algorithms with LLM"""
    
    def __init__(self, llm):
        super().__init__()
        self._llm = llm
        
    def _enhance_with_llm(self, algorithmic_results: Dict, prompt: str) -> Dict:
        """Enhance algorithmic results with LLM reasoning"""
        try:
            # Prepare input for LLM
            llm_input = {
                "algorithmic_results": algorithmic_results,
                "task": prompt
            }
            
            # Format message for the LLM
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert analyst. Analyze the algorithmic results and provide structured insights."
                },
                {
                    "role": "user",
                    "content": f"""
                    Analyze and enhance these algorithmic results:
                    {json.dumps(llm_input, indent=2)}
                    
                    {prompt}
                    
                    Return your analysis in valid JSON format.
                    """
                }
            ]
            
            # Get LLM response using the call method
            response = self._llm.call(messages)
            
            # Parse LLM response
            return json.loads(response)
            
        except Exception as e:
            return {
                "error": f"LLM enhancement failed: {str(e)}",
                "original_results": algorithmic_results
            }

class TopicAnalysisTool(EnhancedAnalysisTool):
    """Tool combining TF-IDF, LDA, and LLM for sophisticated topic analysis"""
    name: str = "Topic Analysis"
    description: str = """Performs comprehensive topic analysis using TF-IDF, LDA, and LLM reasoning.
    Returns structured topic hierarchy with relationships and context."""
    
    def _run(self, text: str) -> str:
        # Parse input JSON if needed
        if isinstance(text, str) and text.startswith('{'):
            try:
                data = json.loads(text)
                text = data.get('text', text)
            except json.JSONDecodeError:
                pass
                
        # TF-IDF Analysis
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        # LDA Topic Modeling
        lda = LatentDirichletAllocation(
            n_components=3,
            random_state=42
        )
        lda_output = lda.fit_transform(tfidf_matrix)
        
        # Combine algorithmic results
        algorithmic_results = {
            "tfidf_terms": [
                {"term": term, "score": float(score)}
                for term, score in sorted(
                    zip(feature_names, tfidf_scores),
                    key=lambda x: x[1],
                    reverse=True
                )[:20]
            ],
            "topics": [
                {
                    "id": topic_idx,
                    "terms": [
                        {
                            "term": feature_names[i],
                            "weight": float(score)
                        }
                        for i, score in sorted(
                            enumerate(topic),
                            key=lambda x: x[1],
                            reverse=True
                        )[:10]
                    ],
                    "weight": float(lda_output[0][topic_idx])
                }
                for topic_idx, topic in enumerate(lda.components_)
            ]
        }
        
        # Enhance with LLM
        enhanced_results = self._enhance_with_llm(
            algorithmic_results,
            """
            Analyze these algorithmic topic analysis results and:
            1. Create a hierarchical topic structure
            2. Identify relationships between topics
            3. Provide context and significance for each topic
            4. Suggest potential subtopics
            5. Return a comprehensive topic analysis in this JSON structure:
            {
                "main_topics": [
                    {
                        "name": "topic name",
                        "description": "topic description",
                        "significance": "topic significance",
                        "subtopics": ["subtopic1", "subtopic2"],
                        "related_topics": ["related1", "related2"],
                        "key_terms": ["term1", "term2"]
                    }
                ],
                "topic_relationships": [
                    {
                        "from": "topic1",
                        "to": "topic2",
                        "relationship": "relationship description"
                    }
                ]
            }
            """
        )
        
        return json.dumps(enhanced_results)

class InsightAnalysisTool(EnhancedAnalysisTool):
    """Tool combining sentence importance scoring with LLM for insight extraction"""
    name: str = "Insight Analysis"
    description: str = """Extracts and analyzes insights using sentence importance scoring and LLM reasoning.
    Returns structured insights with context and relationships."""
    
    def _run(self, text: str) -> str:
        # Parse input JSON if needed
        if isinstance(text, str) and text.startswith('{'):
            try:
                data = json.loads(text)
                text = data.get('text', text)
            except json.JSONDecodeError:
                pass
                
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Score sentences using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        importance_scores = tfidf_matrix.toarray().sum(axis=1)
        
        # Initial classification
        algorithmic_results = {
            "sentences": [
                {
                    "text": sent,
                    "importance_score": float(score),
                    "potential_type": self._classify_sentence(sent)
                }
                for sent, score in zip(sentences, importance_scores)
            ]
        }
        
        # Enhance with LLM
        enhanced_results = self._enhance_with_llm(
            algorithmic_results,
            """
            Analyze these sentence-level results and:
            1. Identify key research insights
            2. Extract methodological approaches
            3. Highlight significant findings
            4. Note novel contributions
            5. Suggest future directions
            6. Return a comprehensive analysis in this JSON structure:
            {
                "methodology": {
                    "approaches": [
                        {
                            "description": "approach description",
                            "significance": "why this is important",
                            "context": "how it was used"
                        }
                    ]
                },
                "findings": [
                    {
                        "finding": "description of finding",
                        "evidence": "supporting evidence",
                        "implications": "what this means"
                    }
                ],
                "contributions": [
                    {
                        "contribution": "description",
                        "novelty": "what makes it novel",
                        "impact": "potential impact"
                    }
                ],
                "future_directions": [
                    {
                        "direction": "description",
                        "rationale": "why this direction",
                        "potential": "expected outcomes"
                    }
                ]
            }
            """
        )
        
        return json.dumps(enhanced_results)
        
    def _classify_sentence(self, sentence: str) -> str:
        """Basic sentence classification"""
        sent_lower = sentence.lower()
        
        if any(kw in sent_lower for kw in ["method", "approach", "technique"]):
            return "methodology"
        elif any(kw in sent_lower for kw in ["found", "result", "show"]):
            return "finding"
        elif any(kw in sent_lower for kw in ["contribute", "improve", "novel"]):
            return "contribution"
        elif any(kw in sent_lower for kw in ["future", "could", "would"]):
            return "future_work"
        else:
            return "other"

class QuestionAnalysisTool(EnhancedAnalysisTool):
    """Tool combining question extraction with LLM for comprehensive question analysis"""
    name: str = "Question Analysis"
    description: str = """Analyzes research questions using pattern matching and LLM reasoning.
    Returns structured question analysis with context and relationships."""
    
    def _run(self, text: str) -> str:
        # Parse input JSON if needed
        if isinstance(text, str) and text.startswith('{'):
            try:
                data = json.loads(text)
                text = data.get('text', text)
            except json.JSONDecodeError:
                pass
                
        # Extract sentences
        sentences = sent_tokenize(text)
        
        # Initial classification
        algorithmic_results = {
            "extracted_questions": {
                "explicit": [
                    sent for sent in sentences if '?' in sent
                ],
                "implicit": [
                    sent for sent in sentences
                    if any(kw in sent.lower() for kw in [
                        "investigate", "examine", "explore", "study",
                        "analyze", "determine", "assess", "evaluate"
                    ])
                ],
                "hypotheses": [
                    sent for sent in sentences
                    if any(kw in sent.lower() for kw in [
                        "hypothes", "predict", "expect", "assume"
                    ])
                ]
            }
        }
        
        # Enhance with LLM
        enhanced_results = self._enhance_with_llm(
            algorithmic_results,
            """
            Analyze these extracted questions and:
            1. Identify primary and secondary research questions
            2. Analyze hypotheses and their relationships
            3. Generate relevant follow-up questions
            4. Map questions to potential answers
            5. Return a comprehensive analysis in this JSON structure:
            {
                "research_questions": {
                    "primary": [
                        {
                            "question": "the question",
                            "focus": "what it addresses",
                            "significance": "why it matters"
                        }
                    ],
                    "secondary": [
                        {
                            "question": "the question",
                            "relationship": "how it relates to primary",
                            "contribution": "what it adds"
                        }
                    ]
                },
                "hypotheses": [
                    {
                        "hypothesis": "the hypothesis",
                        "related_question": "which question it addresses",
                        "testability": "how it can be tested"
                    }
                ],
                "follow_up_questions": [
                    {
                        "question": "the question",
                        "rationale": "why this is important",
                        "potential_impact": "what it could reveal"
                    }
                ]
            }
            """
        )
        
        return json.dumps(enhanced_results)
