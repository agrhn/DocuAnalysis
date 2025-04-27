"""
Module: agents/qa_agent.py

Purpose:
Defines the `QAAgent` class dedicated to answering user queries about documents, retrieving relevant
information, synthesizing answers from data sources, and supporting conversational interactions with
user context awareness.

Key Responsibilities:
1. **Question Answering**:
   - Accurately respond to user queries, citing document sections and metadata.
   - Identify question types (e.g., factual, analytical) for optimized retrieval.
   - Answer based on context provided by document segments and analysis results.

2. **Information Retrieval**:
   - Retrieve document content, extracted facts, metadata, and analysis results.
   - Use semantic search to locate relevant document segments across workflows.
   - Pull insights from pre-defined document collections and analysis.

3. **Context Management**:
   - Track conversation history for context-aware follow-up questions.
   - Maintain and clear conversation history to manage user session resets.
   - Anticipate user follow-up questions based on previous interactions.

4. **Workflow Integration**:
   - Handle user queries directly or integrate workflow state (e.g., analyzed segments in `workflow_id`).
   - Operate seamlessly with `MemoryManager` to retrieve document and analysis data.

5. **Command Processing**:
   - Interface with user commands like "summarize the document" or "show key facts."
   - Suggest related questions to deepen understanding of document content.

Constructor Parameters:
- `agent_id`, `name`: Identifies the QA Agent and its assigned role.
- `memory_manager`: Instance of `MemoryManager` for managing context, document, and analysis data.
- `model`, `temperature`, `max_tokens`: Configuration for the underlying OpenAI LLM, optimized for factual responses.

Core Methods:
1. Question and Answering Methods:
   - `answer_question`: Generate answers to user queries based on context-aware retrieval and analysis.
   - `suggest_related_questions`: Suggest follow-up questions to expand user exploration.
2. Command Handling:
   - `handle_user_query`: Process questions or commands to retrieve desired information.
   - `_process_user_command`: Interprets user commands for summarizing, retrieving facts, or analysis.
   - `_handle_*` Methods: Handle specific commands (e.g., summarization, facts retrieval, analysis requests).
3. Retrieval Methods:
   - `_retrieve_relevant_segments`: Locate relevant document segments via `MemoryManager`.
   - `_retrieve_relevant_analyses`: Fetch analysis results matching the user's query and type.

Integration Notes:
- Extends `BaseAgent`, inheriting core functionalities like communication and reasoning.
- Designed to query document segments, extracted facts, metadata, and analysis results across workflows.
- Highly modular, enabling extension for additional question types or specialized commands.
"""
from typing import Dict, Any, List, Callable
import uuid
import re
from datetime import datetime
from agents.base_agent import BaseAgent
from memory.memory_manager import MemoryManager


class QAAgent(BaseAgent):
    """
    Q&A Agent responsible for answering user queries about documents,
    retrieving relevant information, and providing accurate, contextual responses.
    """
    def __init__(
        self,
        agent_id: str,
        name: str,
        memory_manager: MemoryManager,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,  # Lower temperature for more factual responses
        max_tokens: int = 1500
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            role="Document Question Answerer",
            memory_manager=memory_manager,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Set Q&A-specific instructions
        qa_instructions = """
        As the Q&A Agent, your primary responsibilities are:

        1. Answer user queries about document content:
           - Provide accurate, factual responses based on document content
           - Cite specific sections or sources from the document
           - Clarify ambiguities in questions when necessary
           - Indicate when information is not available in the document

        2. Retrieve relevant information:
           - Search document segments for information relevant to queries
           - Pull facts and insights from analysis results
           - Synthesize information from multiple document sections
           - Reference metadata and extracted entities when appropriate

        3. Provide contextual responses:
           - Consider document context when answering questions
           - Include relevant background information when helpful
           - Explain relationships between concepts mentioned in queries
           - Connect user questions to document themes and main points

        4. Support follow-up questions:
           - Maintain context across multiple questions
           - Track what information has been provided
           - Anticipate potential follow-up questions
           - Suggest related questions the user might be interested in

        Your answers should be concise, accurate, and directly address the user's query.
        Always ground your responses in the document content rather than general knowledge.
        """

        self.set_additional_instructions(qa_instructions)

        # Question types and answer templates
        self.question_types = {
            "factual": "Questions seeking specific facts or information from the document",
            "conceptual": "Questions about concepts, ideas, or theories discussed in the document",
            "analytical": "Questions requiring analysis or interpretation of the document content",
            "comparative": "Questions comparing different aspects or sections of the document",
            "clarification": "Questions seeking clarification about document content or structure",
            "summary": "Questions asking for summaries of document sections or the entire document",
            "metadata": "Questions about document properties like author, date, sources, etc."
        }

        # Track conversation history for context
        self.conversation_history = []

    def answer_question(self, question: str, document_id: str = None,
                        segment_ids: List[str] = None) -> Dict[str, Any]:
        """
        Answer a user question about document content

        Args:
            question: The user's question
            document_id: ID of the document to query (optional)
            segment_ids: List of specific segment IDs to consider (optional)

        Returns:
            Dict containing the answer and metadata
        """
        # Add question to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": question,
            "timestamp": datetime.now().isoformat()
        })

        # Identify question type for better retrieval
        question_type = self._identify_question_type(question)

        # Retrieve relevant document segments
        relevant_segments = self._retrieve_relevant_segments(
            question, document_id, segment_ids
        )

        # Retrieve relevant analysis results
        relevant_analyses = self._retrieve_relevant_analyses(
            question, document_id, segment_ids, question_type
        )

        # Prepare answer context
        context = self._prepare_answer_context(
            relevant_segments, relevant_analyses, question_type
        )

        # Generate answer based on context
        answer_prompt = f"""
        Answer the following question based on the document content provided.
        Be specific, accurate, and cite your sources from the document when possible.

        Question: {question}

        Question type: {question_type}

        Document context:
        {context}

        Previous conversation:
        {self._format_conversation_history()}

        Your answer should:
        1. Directly address the question
        2. Be based only on information in the context provided
        3. Cite specific parts of the document when appropriate
        4. Acknowledge if the document doesn't contain the answer
        5. Be concise but complete
        """

        # Think through the answer
        thinking_input = {
            "question": question,
            "question_type": question_type,
            "context": context,
            "conversation_history": self.conversation_history,
            "document_id": document_id,
            "segment_ids": segment_ids
        }

        thinking_result = self.think(
            thinking_input,
            context_query=question,
            reasoning_prompt=answer_prompt,
            action_options=["communicate", "retrieve_info"]
        )

        # Prepare answer and metadata
        answer = {
            "content": thinking_result["reasoning"]["output"],
            "question": question,
            "question_type": question_type,
            "document_id": document_id,
            "segment_ids": [s.get("segment_id") for s in relevant_segments],
            "referenced_analyses": [a.get("id") for a in relevant_analyses],
            "answer_time": datetime.now().isoformat(),
            "confidence": self._assess_answer_confidence(
                thinking_result["reasoning"]["output"],
                context,
                question
            )
        }

        # Add answer to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": answer["content"],
            "timestamp": datetime.now().isoformat()
        })

        # Store answer in memory
        answer_id = f"answer_{self.agent_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.memory.store_working_data(answer_id, {
            "answer": answer,
            "question": question,
            "context_used": context[:500] + "..." if len(context) > 500 else context,
            "creation_time": datetime.now().isoformat()
        })

        return answer

    def _identify_question_type(self, question: str) -> str:
        """
        Identify the type of question being asked to better retrieve relevant information.
        """
        question_lower = question.lower()

        # Rule-based classification using regex
        if re.search(r'\bwho\b|\bwhat\b|\bwhen\b|\bwhere\b|\bhow many\b|\blist\b', question_lower):
            return "factual"
        elif re.search(r'\bwhy\b|\bhow\b|\bexplain\b|\bdescribe\b', question_lower):
            return "conceptual"
        elif re.search(r'\banalyze\b|\binterpret\b|\bevaluate\b|\bexamine\b', question_lower):
            return "analytical"
        elif re.search(r'\bcompare\b|\bcontrast\b|\bdifference\b|\bsimilar\b', question_lower):
            return "comparative"
        elif re.search(r'\bclarify\b|\bmeaning\b|\bdefine\b|\bexplain\b', question_lower):
            return "clarification"
        elif re.search(r'\bsummarize\b|\bsummary\b|\boverview\b|\bbriefly\b', question_lower):
            return "summary"
        elif re.search(r'\bauthor\b|\bdate\b|\btitle\b|\bpublish\b|\bsource\b', question_lower):
            return "metadata"
        else:
            return "unknown"  # Explicitly mark unknown types for fallback handling

    def _retrieve_relevant_segments(self, question: str, document_id: str = None,
                                    segment_ids: List[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve document segments relevant to the question.
        """
        relevant_segments = []

        # Specific segment IDs provided: Retrieve them directly
        if segment_ids:
            for segment_id in segment_ids:
                segment_data = self.memory.retrieve_document_segment(segment_id)
                if segment_data:
                    relevant_segments.append(segment_data)
            return relevant_segments

        # Search within document if document_id is provided
        if document_id:
            try:
                search_results = self.memory.search_documents(
                    query=question,
                    collection=f"document_{document_id}_segments",
                    n_results=3
                )
            except Exception as e:
                return {
                    "error": f"Failed to search within document segments: {str(e)}"
                }

            for result in search_results:
                segment_id = result.get("id")
                segment_data = self.memory.retrieve_document_segment(segment_id)
                if segment_data:
                    relevant_segments.append(segment_data)
        else:
            # Search across all document segments in workflow
            try:
                search_results = self.memory.search_documents(
                    query=question,
                    collection="document_segments",
                    n_results=3
                )
            except Exception as e:
                return [{"error": f"Failed to search across document segments: {str(e)}"}]

            for result in search_results:
                segment_id = result.get("id")
                segment_data = self.memory.retrieve_document_segment(segment_id)
                if segment_data:
                    relevant_segments.append(segment_data)

        return relevant_segments

    def _retrieve_relevant_analyses(self, question: str, document_id: str = None,
                                    segment_ids: List[str] = None,
                                    question_type: str = "factual") -> List[Dict[str, Any]]:
        """
        Retrieve analysis results relevant to the question
        """
        relevant_analyses = []

        # Map question types to analysis types
        analysis_type_map = {
            "factual": ["extraction", "summarization"],
            "conceptual": ["summarization", "content_quality"],
            "analytical": ["content_quality", "bias", "argumentation"],
            "comparative": ["content_quality", "alternative_perspectives"],
            "clarification": ["extraction", "summarization"],
            "summary": ["summarization"],
            "metadata": ["extraction"]
        }

        # Get relevant analysis types for this question
        analysis_types = analysis_type_map.get(question_type, ["extraction", "summarization"])

        # If segment IDs are provided, get analyses for those segments
        if segment_ids:
            for segment_id in segment_ids:
                for analysis_type in analysis_types:
                    query = f"segment_id:{segment_id} analysis_type:{analysis_type}"
                    analysis_results = self.memory.get_working_data(query)
                    relevant_analyses.extend(analysis_results)

        # If document ID is provided, get analyses for that document
        elif document_id:
            for analysis_type in analysis_types:
                query = f"document_id:{document_id} analysis_type:{analysis_type}"
                analysis_results = self.memory.get_working_data(query)
                relevant_analyses.extend(analysis_results)

        # If neither is provided, use semantic search to find relevant analyses
        else:
            try:
                search_results = self.memory.search_documents(
                    question,
                    collection="analysis_results",
                    n_results=3
                )
            except Exception as e:
                return [{"error": f"Failed to search across document segments: {str(e)}"}]

            for result in search_results:
                analysis_id = result.get("id")
                analysis_data = self.memory.get_working_data(analysis_id)
                if analysis_data:
                    relevant_analyses.append(analysis_data)

        return relevant_analyses

    def _prepare_answer_context(self,
                                relevant_segments: List[Dict[str, Any]],
                                relevant_analyses: List[Dict[str, Any]],
                                question_type: str) -> str:
        """
        Prepare context for answering the question based on retrieved information
        """
        context_parts = []

        # Add segment content to context
        for i, segment in enumerate(relevant_segments):
            content = segment.get("content", "")
            segment_id = segment.get("segment_id", f"unknown_segment_{i}")

            if content:
                context_parts.append(f"--- SEGMENT {segment_id} ---\n{content}\n")

        # Add analysis results to context based on question type
        for analysis in relevant_analyses:
            analysis_type = analysis.get("analysis_type", "unknown")
            analysis_data = analysis.get("analysis", {})

            # Format analysis data based on type
            if analysis_type == "extraction":
                facts = analysis_data.get("facts", [])
                if facts:
                    facts_text = "\n- ".join([""] + [f"{fact}" for fact in facts])
                    context_parts.append(f"--- EXTRACTED FACTS ---{facts_text}\n")

            elif analysis_type == "summarization":
                summary = analysis_data.get("summary", "")
                if summary:
                    context_parts.append(f"--- DOCUMENT SUMMARY ---\n{summary}\n")

            elif analysis_type == "content_quality" and question_type in ["analytical", "comparative"]:
                quality_assessment = analysis_data.get("overall_assessment", "")
                if quality_assessment:
                    context_parts.append(f"--- CONTENT QUALITY ANALYSIS ---\n{quality_assessment}\n")

            elif analysis_type == "bias" and question_type in ["analytical"]:
                bias_assessment = analysis_data.get("overall_assessment", "")
                if bias_assessment:
                    context_parts.append(f"--- BIAS ANALYSIS ---\n{bias_assessment}\n")

            elif analysis_type == "argumentation" and question_type in ["analytical", "conceptual"]:
                arg_assessment = analysis_data.get("overall_assessment", "")
                if arg_assessment:
                    context_parts.append(f"--- ARGUMENTATION ANALYSIS ---\n{arg_assessment}\n")

            elif analysis_type == "alternative_perspectives" and question_type in ["comparative", "analytical"]:
                perspectives = analysis_data.get("overall_assessment", "")
                if perspectives:
                    context_parts.append(f"--- ALTERNATIVE PERSPECTIVES ---\n{perspectives}\n")

        return "\n".join(context_parts)

    def _format_conversation_history(self, max_turns: int = 3) -> str:
        """
        Format recent conversation history for context
        """
        if not self.conversation_history or len(self.conversation_history) < 2:
            return "No previous conversation."

        # Get last few turns (excluding the current question)
        recent_history = self.conversation_history[-(max_turns*2+1):-1]
        formatted_history = []

        for entry in recent_history:
            role = entry.get("role", "unknown")
            content = entry.get("content", "")
            formatted_history.append(f"{role.capitalize()}: {content}")

        return "\n".join(formatted_history)

    def _assess_answer_confidence(self, answer: str, context: str, question: str) -> float:
        """
        Assess confidence in the answer based on context coverage
        """
        # Simple heuristic: check if answer contains specific phrases indicating uncertainty
        uncertainty_phrases = [
            "the document doesn't mention",
            "not mentioned in the document",
            "not provided in the context",
            "I don't have information",
            "cannot be determined",
            "unclear from the document"
        ]

        # Check for uncertainty indicators
        for phrase in uncertainty_phrases:
            if phrase in answer.lower():
                return 0.5  # Medium-low confidence

        # Check if answer seems relevant to question
        question_keywords = set(re.findall(r'\b\w{3,}\b', question.lower()))
        answer_keywords = set(re.findall(r'\b\w{3,}\b', answer.lower()))

        if question_keywords:
            keyword_overlap = len(question_keywords.intersection(answer_keywords)) / len(question_keywords)
        else:
            keyword_overlap = 0

        # Basic confidence score (could be improved with ML-based approaches)
        if keyword_overlap > 0.7:
            return 0.9  # High confidence
        elif keyword_overlap > 0.4:
            return 0.7  # Medium confidence
        else:
            return 0.5  # Lower confidence

    def suggest_related_questions(self, original_question: str, answer: str,
                                  document_id: str = None) -> List[str]:
        """
        Suggest related questions based on the current question and answer
        """
        suggestion_prompt = f"""
        Based on the following question and answer about a document, suggest 3 related
        follow-up questions that the user might be interested in asking next.

        Original question: {original_question}

        Answer provided: {answer}

        Suggested follow-up questions should:
        1. Be relevant to the original question and answer
        2. Explore related aspects of the document
        3. Help deepen understanding of the document content
        4. Be clear and specific

        Format your response as a simple list of 3 questions, one per line.
        """

        # Generate suggestions
        thinking_input = {
            "original_question": original_question,
            "answer": answer,
            "document_id": document_id
        }

        thinking_result = self.think(
            thinking_input,
            context_query="suggest related questions",
            reasoning_prompt=suggestion_prompt,
            action_options=["communicate"]
        )

        # Parse out the questions
        suggested_questions = []
        reasoning_output = thinking_result["reasoning"]["output"]

        # Find all questions (sentences ending with '?')
        question_matches = re.findall(r'[^.!?]*\?', reasoning_output)

        # Clean up the questions
        for match in question_matches:
            question = match.strip()
            if question and question not in suggested_questions:
                suggested_questions.append(question)

        # If regex didn't find questions, split by newlines as fallback
        if not suggested_questions:
            lines = reasoning_output.split('\n')
            for line in lines:
                line = line.strip()
                if line and '?' in line and line not in suggested_questions:
                    suggested_questions.append(line)

        return suggested_questions[:3]  # Return at most 3 suggestions

    def clear_conversation_history(self) -> None:
        """
        Clear the conversation history to start a new conversation
        """
        self.conversation_history = []

    # Custom actions for this agent

    def _action_answer_question(self, question: str, document_id: str = None,
                                segment_ids: List[str] = None) -> Dict[str, Any]:
        """
        Action to answer a user question
        """
        answer = self.answer_question(question, document_id, segment_ids)

        # Generate suggested follow-up questions
        suggested_questions = self.suggest_related_questions(
            question, answer["content"], document_id
        )

        return {
            "success": True,
            "answer": answer,
            "suggested_questions": suggested_questions
        }

    def _action_retrieve_info(self, query: str, context_type: str = "all") -> Dict[str, Any]:
        """
        Action to retrieve specific information from document memory
        """
        results = {}

        if context_type in ["all", "segments"]:
            # Retrieve document segments
            segment_results = self.memory.search_documents(
                query, collection="document_segments", n_results=3
            )
            results["segments"] = segment_results

        if context_type in ["all", "analyses"]:
            # Retrieve analysis results
            analysis_results = self.memory.search_documents(
                query, collection="analysis_results", n_results=3
            )
            results["analyses"] = analysis_results

        if context_type in ["all", "facts"]:
            # Retrieve extracted facts
            fact_results = self.memory.search_documents(
                query, collection="extracted_facts", n_results=5
            )
            results["facts"] = fact_results

        return {
            "success": True,
            "results": results
        }

    def _action_suggest_questions(self, current_question: str, answer: str,
                                  document_id: str = None) -> Dict[str, Any]:
        """
        Action to suggest follow-up questions
        """
        suggested_questions = self.suggest_related_questions(
            current_question, answer, document_id
        )

        return {
            "success": True,
            "suggested_questions": suggested_questions
        }

    def get_available_actions(self) -> Dict[str, Callable]:
        """
        Get custom actions available for this agent
        """
        base_actions = {
            "communicate": self._action_communicate,
            "store_fact": self._action_store_fact,
            "retrieve_info": self._action_retrieve_info,
            "no_action": self._action_no_action,
            "error": self._action_error
        }

        qa_actions = {
            "answer_question": self._action_answer_question,
            "suggest_questions": self._action_suggest_questions
        }

        return {**base_actions, **qa_actions}

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task assigned by the coordinator
        """
        task_type = task.get("task_type")
        parameters = task.get("parameters", {})

        # Process based on task type
        if task_type == "answer_question":
            question = parameters.get("question", "")
            document_id = parameters.get("document_id")
            segment_ids = parameters.get("segment_ids")

            result = self._action_answer_question(question, document_id, segment_ids)

        elif task_type == "retrieve_information":
            query = parameters.get("query", "")
            context_type = parameters.get("context_type", "all")

            result = self._action_retrieve_info(query, context_type)

        elif task_type == "suggest_questions":
            current_question = parameters.get("current_question", "")
            answer = parameters.get("answer", "")
            document_id = parameters.get("document_id")

            result = self._action_suggest_questions(current_question, answer, document_id)

        else:
            result = {
                "success": False,
                "error": f"Unknown task type: {task_type}"
            }

        # Add task metadata to result
        result["task_id"] = task.get("task_id")
        result["agent_id"] = self.agent_id
        result["completion_time"] = datetime.now().isoformat()

        # Store the task result
        task_result_id = f"task_result_{task.get('task_id', uuid.uuid4())}"
        self.memory.store_working_data(task_result_id, {
            "result": result,
            "task": task,
            "completion_time": datetime.now().isoformat()
        })

        return result

    def handle_user_query(self, query: str, document_id: str = None,
                          workflow_id: str = None) -> Dict[str, Any]:
        """
        High-level method to handle a direct user query
        """
        # Determine if this is a question or a command
        is_question = '?' in query or re.search(r'\bwh[aoy]\b|\bhow\b|\bcan\b|\bcould\b|\bshould\b', query.lower())

        if is_question:
            # Get contextual information from workflow if available
            segment_ids = None
            if workflow_id:
                # Retrieve workflow state to get relevant segments
                workflow_data = self.memory.retrieve_working_data(f"workflow_{workflow_id}")
                if workflow_data:
                    # Get most recently analyzed segments for context
                    segment_ids = workflow_data.get("analyzed_segments", [])[-3:]

            # Process as a question
            answer_result = self._action_answer_question(query, document_id, segment_ids)

            return {
                "success": True,
                "response_type": "answer",
                "answer": answer_result["answer"]["content"],
                "confidence": answer_result["answer"]["confidence"],
                "suggested_questions": answer_result["suggested_questions"]
            }
        else:
            # Process as a command or instruction
            # Here we could add functionality to handle commands like "summarize the document",
            # "show me key facts", etc.

            command_result = self._process_user_command(query, document_id, workflow_id)

            return {
                "success": True,
                "response_type": "command_response",
                "response": command_result["response"],
                "command_type": command_result["command_type"]
            }

    def _process_user_command(self, command: str, document_id: str = None,
                              workflow_id: str = None) -> Dict[str, Any]:
        """
        Process user commands that aren't direct questions
        """
        command_lower = command.lower()

        # Check for different command types
        if re.search(r'\bsummarize\b|\bsummary\b|\boverview\b', command_lower):
            # Handle summarization request
            return self._handle_summarize_command(command, document_id, workflow_id)

        elif re.search(r'\bfacts\b|\bkey points\b|\bhighlights\b', command_lower):
            # Handle request for key facts
            return self._handle_facts_command(command, document_id, workflow_id)

        elif re.search(r'\banalysis\b|\banalyze\b|\bevaluate\b', command_lower):
            # Handle analysis request
            return self._handle_analysis_command(command, document_id, workflow_id)

        elif re.search(r'\breset\b|\bstart over\b|\bnew conversation\b', command_lower):
            # Handle conversation reset
            self.clear_conversation_history()
            return {
                "command_type": "reset",
                "response": "Conversation history has been cleared. What would you like to know about the document?"
            }

        else:
            # Default handling for unrecognized commands
            return {
                "command_type": "unknown",
                "response": "I'm not sure how to process that request. Could you phrase it as a question about " +
                            "the document, or try commands like 'summarize the document' or 'show key facts'?"
            }

    def _handle_summarize_command(self, command: str, document_id: str = None,
                                  workflow_id: str = None) -> Dict[str, Any]:
        """
        Handle a request to summarize the document
        """
        # Try to retrieve existing summaries
        summaries = []

        if document_id:
            query = f"document_id:{document_id} analysis_type:summarization"
            summary_results = self.memory.get_working_data(query)
            summaries.extend(summary_results)

        if not summaries and workflow_id:
            # If no direct summaries, try looking through workflow results
            workflow_data = self.memory.get_working_data(f"workflow_{workflow_id}")
            if workflow_data and "analysis_results" in workflow_data:
                summary_results = workflow_data["analysis_results"].get("summarize", [])
                summaries.extend(summary_results)

        # Process and return the summary
        if summaries:
            # Use the most comprehensive summary available
            selected_summary = max(summaries, key=lambda x: len(str(x.get("result", ""))))
            summary_text = selected_summary.get("result", {}).get("summary", "")

            if not summary_text:
                summary_text = "I found a summary but couldn't extract the content. " +\
                               "You may need to request a new summary."
        else:
            # No existing summary found
            summary_text = "I couldn't find an existing summary for this document. " +\
                           "Would you like me to generate one for you?"

        return {
            "command_type": "summarize",
            "response": summary_text
        }

    def _handle_facts_command(self, command: str, document_id: str = None,
                              workflow_id: str = None) -> Dict[str, Any]:
        """
        Handle a request for key facts from the document
        """
        # Try to retrieve extracted facts
        facts = []

        if document_id:
            query = f"document_id:{document_id} analysis_type:extraction"
            extraction_results = self.memory.get_working_data(query)

            for result in extraction_results:
                result_facts = result.get("result", {}).get("facts", [])
                if result_facts:
                    facts.extend(result_facts)

        if not facts and workflow_id:
            # If no direct facts, try looking through workflow results
            workflow_data = self.memory.get_working_data_by_partial_key(f"workflow_{workflow_id}")
            if workflow_data and "analysis_results" in workflow_data:
                extraction_results = workflow_data["analysis_results"].get("extract_facts", [])
                for result in extraction_results:
                    result_facts = result.get("result", {}).get("facts", [])
                    if result_facts:
                        facts.extend(result_facts)

        # Format and return the facts
        if facts:
            facts_text = "Here are the key facts from the document:\n\n"
            for i, fact in enumerate(facts[:10], 1):  # Limit to 10 facts
                facts_text += f"{i}. {fact}\n"

            if len(facts) > 10:
                facts_text += f"\n... and {len(facts) - 10} more facts."
        else:
            # No facts found
            facts_text = "I couldn't find extracted facts for this document. " +\
                         "Would you like me to extract key information for you?"

        return {
            "command_type": "facts",
            "response": facts_text
        }

    def _handle_analysis_command(self, command: str, document_id: str = None,
                                 workflow_id: str = None) -> Dict[str, Any]:
        """
        Handle a request for document analysis
        """
        # Try to retrieve analysis results
        analyses = []

        if document_id:
            # Look for content quality and bias analyses
            quality_query = f"document_id:{document_id} analysis_type:content_quality"
            bias_query = f"document_id:{document_id} analysis_type:bias"

            quality_results = self.memory.get_working_data(quality_query)
            bias_results = self.memory.get_working_data(bias_query)

            if quality_results:
                analyses.append("**Content Quality Analysis:**\n" + quality_results[0]['content'])
            else:
                analyses.append("No content quality analysis found.")

            if bias_results:
                analyses.append("**Bias Analysis:**\n" + bias_results[0]['content'])
            else:
                analyses.append("No bias analysis found.")
        else:
            analyses.append("Document ID was not provided, so no analysis could be retrieved.")

        analysis_text = "\n\n".join(analyses)

        return {
            "command_type": "analysis",
            "response": analysis_text
        }

    def _handle_unknown_command(self, command: str) -> Dict[str, Any]:
        """
        Fallback for unrecognized commands
        """
        return {
            "command_type": "unknown",
            "response": f"I'm sorry, I didn't understand the command: '{command}'. Please try again or rephrase it."
        }

    def handle_command(self, command: str, document_id: str = None, workflow_id: str = None) -> Dict[str, Any]:
        """
        Entry point for handling user commands
        """
        command_lower = command.strip().lower()

        if "summary" in command_lower:
            return self._handle_summary_command(command, document_id, workflow_id)
        elif "fact" in command_lower or "key point" in command_lower:
            return self._handle_facts_command(command, document_id, workflow_id)
        elif "analysis" in command_lower or "quality" in command_lower or "bias" in command_lower:
            return self._handle_analysis_command(command, document_id, workflow_id)
        else:
            return self._handle_unknown_command(command)
