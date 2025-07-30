from openai import OpenAI
import json
from typing import Dict, List, Tuple
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class SmartQuestionAgent:
    """AI agent that understands, improves, and routes questions intelligently"""
    
    def __init__(self):
        self.conversation_memory = []
        
    def process_question(self, question: str, document_context: str = "") -> Dict:
        """Main agent orchestration method"""
        
        # Step 1: Analyze the question
        question_analysis = self._analyze_question(question, document_context)
        
        # Step 2: Decide on processing strategy
        strategy = self._determine_strategy(question_analysis)
        
        # Step 3: Execute the strategy
        result = self._execute_strategy(question, question_analysis, strategy)
        
        # Step 4: Store in memory for context
        self._update_memory(question, result)
        
        return result
    
    def _analyze_question(self, question: str, context: str) -> Dict:
        """Analyze question complexity, intent, and requirements"""
        
        analysis_prompt = f"""Analyze this question about the document content:

Question: "{question}"

Document context available: {"Yes" if context else "No"}
Previous questions in conversation: {len(self.conversation_memory)}

Provide analysis in JSON format:
{{
    "intent": "factual|analytical|comparative|clarification|summary",
    "complexity": "simple|moderate|complex",
    "requires_multiple_sources": true/false,
    "ambiguous_terms": ["list", "of", "unclear", "terms"],
    "suggested_improvements": ["better", "question", "alternatives"],
    "question_type": "direct_answer|multi_step|requires_examples|needs_comparison"
}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1
            )
            
            analysis = json.loads(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            # Fallback to simple analysis
            return {
                "intent": "factual",
                "complexity": "simple", 
                "requires_multiple_sources": False,
                "ambiguous_terms": [],
                "suggested_improvements": [],
                "question_type": "direct_answer"
            }
    
    def _determine_strategy(self, analysis: Dict) -> str:
        """Determine the best strategy for answering based on analysis"""
        
        if analysis["complexity"] == "complex":
            return "multi_step_reasoning"
        elif analysis["requires_multiple_sources"]:
            return "multi_source_synthesis" 
        elif analysis["ambiguous_terms"]:
            return "clarification_first"
        elif len(self.conversation_memory) > 0:
            return "contextual_answer"
        else:
            return "direct_answer"
    
    def _execute_strategy(self, question: str, analysis: Dict, strategy: str) -> Dict:
        """Execute the determined strategy"""
        
        if strategy == "clarification_first":
            return self._handle_clarification(question, analysis)
        elif strategy == "multi_step_reasoning":
            return self._handle_complex_question(question, analysis)
        elif strategy == "contextual_answer":
            return self._handle_contextual_question(question, analysis)
        else:
            return self._handle_direct_question(question, analysis)
    
    def _handle_clarification(self, question: str, analysis: Dict) -> Dict:
        """Handle questions that need clarification"""
        
        clarification_prompt = f"""The user asked: "{question}"

This question contains ambiguous terms: {analysis['ambiguous_terms']}

Generate 2-3 clarifying questions to help the user be more specific:"""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": clarification_prompt}],
                temperature=0.3
            )
            
            clarifications = response.choices[0].message.content
            
            return {
                "type": "clarification_needed",
                "original_question": question,
                "clarifying_questions": clarifications,
                "suggested_questions": analysis.get("suggested_improvements", [])
            }
            
        except Exception as e:
            return {"type": "error", "message": f"Clarification error: {str(e)}"}
    
    def _handle_complex_question(self, question: str, analysis: Dict) -> Dict:
        """Break down complex questions into steps"""
        
        breakdown_prompt = f"""Break down this complex question into 3-5 simpler sub-questions:

Original question: "{question}"

Format as a numbered list of specific, answerable questions:"""

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": breakdown_prompt}],
                temperature=0.2
            )
            
            sub_questions = response.choices[0].message.content
            
            return {
                "type": "multi_step_plan",
                "original_question": question,
                "sub_questions": sub_questions,
                "strategy": "Answer each sub-question then synthesize"
            }
            
        except Exception as e:
            return {"type": "error", "message": f"Breakdown error: {str(e)}"}
    
    def _handle_contextual_question(self, question: str, analysis: Dict) -> Dict:
        """Handle questions that build on previous conversation"""
        
        context = "\n".join([f"Q: {mem['question']} A: {mem['answer'][:200]}..." 
                           for mem in self.conversation_memory[-3:]])
        
        return {
            "type": "contextual",
            "enhanced_question": f"Given our previous discussion:\n{context}\n\nNew question: {question}",
            "context_used": True
        }
    
    def _handle_direct_question(self, question: str, analysis: Dict) -> Dict:
        """Handle straightforward questions"""
        
        # Check if we can suggest better alternatives
        if analysis.get("suggested_improvements"):
            return {
                "type": "direct_with_suggestions",
                "original_question": question,
                "suggested_alternatives": analysis["suggested_improvements"],
                "proceed_with_original": True
            }
        
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
        
        # Keep only last 10 interactions
        if len(self.conversation_memory) > 10:
            self.conversation_memory = self.conversation_memory[-10:]
    
    def get_suggested_followups(self, question: str, answer: str) -> List[str]:
        """Generate smart follow-up questions based on the Q&A"""
        
        followup_prompt = f"""Based on this Q&A exchange, suggest 3 relevant follow-up questions:

Question: {question}
Answer: {answer[:500]}...

Generate 3 specific, actionable follow-up questions:"""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": followup_prompt}],
                temperature=0.4
            )
            
            followups = response.choices[0].message.content.strip().split('\n')
            return [q.strip('123456789. ') for q in followups if q.strip()]
            
        except Exception as e:
            return ["Can you elaborate on this?", "What are the implications?", "How does this relate to the main topic?"]
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.conversation_memory = []