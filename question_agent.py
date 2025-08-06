from openai import OpenAI
import json
from typing import Dict, List, Tuple
import os
import re

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class SmartQuestionAgent:
    """AI agent that understands, improves, and routes questions intelligently"""
    
    def __init__(self):
        self.conversation_memory = []
        
    def process_question(self, question: str, document_context: str = "") -> Dict:
        """Main agent orchestration method - WITH DIAGNOSTIC LOGGING"""
        
        print(f"\n=== AGENT DIAGNOSTIC START ===")
        print(f"Input question: '{question}'")
        print(f"Context provided: {len(document_context)} chars")
        print(f"Context preview: '{document_context[:200]}...'")
        
        # Step 1: Check if straightforward
        is_straightforward = self._is_straightforward_question(question, document_context)
        print(f"Is straightforward: {is_straightforward}")
        
        if is_straightforward:
            result = {
                "type": "direct",
                "processed_question": question,
                "ready_for_rag": True
            }
            print(f"RESULT: Direct processing (straightforward)")
            self._update_memory(question, result)
            print(f"=== AGENT DIAGNOSTIC END ===\n")
            return result
        
        # Step 2: Analyze the question - WITH DETAILED LOGGING
        print("Question not flagged as straightforward - running analysis...")
        question_analysis = self._analyze_question(question, document_context)
        print(f"Analysis result: {question_analysis}")
        
        # Step 3: Determine strategy
        strategy = self._determine_strategy(question_analysis)
        print(f"Strategy determined: {strategy}")
        
        # Step 4: Execute strategy
        result = self._execute_strategy(question, question_analysis, strategy)
        print(f"RESULT: {result.get('type')} - {result}")
        
        # Step 5: Store in memory
        self._update_memory(question, result)
        
        print(f"=== AGENT DIAGNOSTIC END ===\n")
        return result
    
    def _is_straightforward_question(self, question: str, context: str) -> bool:
        """Quick check for straightforward questions - WITH DIAGNOSTIC LOGGING"""
        
        print(f"--- Checking if straightforward ---")
        
        # Check for common question patterns
        straightforward_patterns = [
            r"what.*say|said|mention",  # "what did X say"
            r"who.*deal|agreement|contract",  # "who made the deal"
            r"what.*deal|agreement|contract",  # "what deal"
            r"when.*happen|occur",  # "when did it happen"
            r"where.*location|place",  # "where"
            r"how much.*cost|price|billion|million",  # "how much"
            r"why.*important|matter",  # "why important"
            r"main.*points?|topics?",  # "main points"
            r"key.*takeaways?|insights?",  # "key takeaways"
            r"summary|summarize",  # summary requests
        ]
        
        question_lower = question.lower()
        print(f"Question lowercase: '{question_lower}'")
        
        # Test each pattern
        for pattern in straightforward_patterns:
            match = re.search(pattern, question_lower)
            print(f"Pattern '{pattern}': {'MATCH' if match else 'NO MATCH'}")
            if match:
                print(f"STRAIGHTFORWARD: Matched pattern '{pattern}'")
                return True
        
        # Check length
        word_count = len(question.split())
        print(f"Word count: {word_count}")
        if word_count <= 8:
            print(f"STRAIGHTFORWARD: Short question ({word_count} words)")
            return True
        
        print(f"NOT STRAIGHTFORWARD: No patterns matched, length > 8 words")
        return False
    
    def _analyze_question(self, question: str, context: str) -> Dict:
        """Analyze question - WITH DIAGNOSTIC LOGGING"""
        
        print(f"--- Analyzing question complexity ---")
        
        analysis_prompt = f"""Analyze this question about the document content. BE LENIENT - most questions are answerable.

Question: "{question}"

Document context preview: {context[:500] if context else "No preview available"}
Previous questions: {len(self.conversation_memory)}

Only mark as "ambiguous" if the question is genuinely unclear or has multiple valid interpretations.
Common questions like "what did X say about Y" are NOT ambiguous if X and Y are mentioned in the document.

Provide analysis in JSON format:
{{
    "intent": "factual|analytical|comparative|clarification|summary",
    "complexity": "simple|moderate|complex", 
    "requires_multiple_sources": true/false,
    "ambiguous_terms": ["only", "genuinely", "unclear", "terms"],
    "suggested_improvements": ["only if really needed"],
    "question_type": "direct_answer|multi_step|requires_examples|needs_comparison"
}}"""

        print(f"Sending to GPT-3.5 for analysis...")
        print(f"Prompt preview: {analysis_prompt[:200]}...")

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            print(f"GPT Response: '{content}'")
            
            # Try to parse JSON
            try:
                analysis = json.loads(content)
                print(f"JSON parsed successfully: {analysis}")
            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}")
                # Try to extract JSON from the content
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    print(f"Found JSON in text: {json_match.group()}")
                    analysis = json.loads(json_match.group())
                    print(f"Extracted JSON parsed: {analysis}")
                else:
                    raise ValueError("No valid JSON found in response")
                    
            return analysis
            
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            print("Using fallback analysis...")
            # Fallback to simple analysis
            fallback = {
                "intent": "factual",
                "complexity": "simple", 
                "requires_multiple_sources": False,
                "ambiguous_terms": [],
                "suggested_improvements": [],
                "question_type": "direct_answer"
            }
            print(f"Fallback analysis: {fallback}")
            return fallback
    
    def _determine_strategy(self, analysis: Dict) -> str:
        """Determine strategy - WITH DIAGNOSTIC LOGGING"""
        
        print(f"--- Determining strategy ---")
        print(f"Analysis input: {analysis}")
        
        ambiguous_count = len(analysis.get("ambiguous_terms", []))
        complexity = analysis.get("complexity", "simple")
        
        print(f"Ambiguous terms count: {ambiguous_count}")
        print(f"Complexity: {complexity}")
        print(f"Memory length: {len(self.conversation_memory)}")
        
        # Decision logic with logging
        if ambiguous_count > 2 and complexity == "complex":
            strategy = "clarification_first"
            print(f"STRATEGY: clarification_first (ambiguous: {ambiguous_count}, complex: {complexity})")
        elif complexity == "complex" and analysis.get("requires_multiple_sources", False):
            strategy = "multi_step_reasoning"
            print(f"STRATEGY: multi_step_reasoning (complex + multi-source)")
        elif len(self.conversation_memory) > 0:
            strategy = "contextual_answer"
            print(f"STRATEGY: contextual_answer (has memory)")
        else:
            strategy = "direct_answer"
            print(f"STRATEGY: direct_answer (default)")
        
        return strategy
    
    def _execute_strategy(self, question: str, analysis: Dict, strategy: str) -> Dict:
        """Execute strategy - WITH DIAGNOSTIC LOGGING"""
        
        print(f"--- Executing strategy: {strategy} ---")
        
        if strategy == "clarification_first":
            return self._handle_clarification(question, analysis)
        elif strategy == "multi_step_reasoning":
            return self._handle_complex_question(question, analysis)
        elif strategy == "contextual_answer":
            return self._handle_contextual_question(question, analysis)
        else:
            return self._handle_direct_question(question, analysis)
    
    def _handle_clarification(self, question: str, analysis: Dict) -> Dict:
        """Handle clarification - WITH DIAGNOSTIC LOGGING"""
        
        print(f"--- Handling clarification ---")
        
        ambiguous_terms = analysis.get('ambiguous_terms', [])
        print(f"Ambiguous terms: {ambiguous_terms}")
        
        if len(ambiguous_terms) < 2:
            print("Less than 2 ambiguous terms - switching to direct")
            return {
                "type": "direct",
                "processed_question": question,
                "ready_for_rag": True
            }
        
        print(f"Generating clarification for {len(ambiguous_terms)} ambiguous terms")
        
        clarification_prompt = f"""The user asked: "{question}"

This question contains potentially ambiguous terms: {ambiguous_terms}

Generate 2 specific clarifying questions to help the user be more precise:"""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": clarification_prompt}],
                temperature=0.3
            )
            
            clarifications = response.choices[0].message.content
            print(f"Generated clarifications: {clarifications}")
            
            return {
                "type": "clarification_needed",
                "original_question": question,
                "clarifying_questions": clarifications,
                "suggested_questions": analysis.get("suggested_improvements", [])
            }
            
        except Exception as e:
            print(f"Clarification generation failed: {e} - falling back to direct")
            return {
                "type": "direct",
                "processed_question": question,
                "ready_for_rag": True
            }
    
    def _handle_complex_question(self, question: str, analysis: Dict) -> Dict:
        """Handle complex questions"""
        print(f"--- Handling complex question ---")
        return {
            "type": "direct", 
            "processed_question": question,
            "ready_for_rag": True
        }
    
    def _handle_contextual_question(self, question: str, analysis: Dict) -> Dict:
        """Handle contextual questions"""
        print(f"--- Handling contextual question ---")
        return {
            "type": "contextual",
            "enhanced_question": question,
            "context_used": True
        }
    
    def _handle_direct_question(self, question: str, analysis: Dict) -> Dict:
        """Handle direct questions"""
        print(f"--- Handling direct question ---")
        return {
            "type": "direct",  
            "processed_question": question,
            "ready_for_rag": True
        }
    
    def _update_memory(self, question: str, result: Dict):
        """Update conversation memory"""
        self.conversation_memory.append({
            "question": question,
            "result_type": result.get("type"),
            "answer": result.get("processed_question", question)
        })
        
        if len(self.conversation_memory) > 10:
            self.conversation_memory = self.conversation_memory[-10:]
    
    def get_suggested_followups(self, question: str, answer: str) -> List[str]:
        """Generate smart follow-up questions"""
        try:
            followup_prompt = f"""Based on this Q&A exchange, suggest 3 relevant follow-up questions:

Question: {question}
Answer: {answer[:500]}...

Generate 3 specific, actionable follow-up questions:"""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": followup_prompt}],
                temperature=0.4
            )
            
            followups = response.choices[0].message.content.strip().split('\n')
            cleaned_followups = []
            for q in followups:
                q = q.strip()
                q = re.sub(r'^\d+[\.\)\-\s]+', '', q)
                if q and len(q) > 10:
                    cleaned_followups.append(q)
            
            return cleaned_followups[:3]
            
        except Exception as e:
            return ["Can you elaborate on this?", "What are the implications?", "How does this relate to the main topic?"]
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.conversation_memory = []