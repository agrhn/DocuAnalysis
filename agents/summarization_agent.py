"""
Module: agents/summarization_agent.py

Purpose:
Defines the `SummarizationAgent` class, which focuses on generating concise summaries for
document segments at different levels of detail, focus, and complexity. The agent also
merges multiple summaries into cohesive document-level outputs.

Key Responsibilities:
1. Summary Creation:
   - Generate targeted summaries based on pre-defined styles (e.g., executive, technical, comprehensive).
   - Create progressive summaries at multiple levels of detail in one process.
   - Maintain the original meaning and intent of the document while reducing content length.

2. Summary Integration:
   - Merge multiple segment summaries into a cohesive, unified document-level summary.
   - Eliminate redundancy and ensure consistent flow during integration.

3. Workflow Integration:
   - Process tasks assigned by the Coordinator Agent.
   - Report task results back to the Coordinator after completion.
   - Store summaries in memory (short-term and working memory).

Constructor Parameters:
- `agent_id`, `name`: Identifies the Summarization Agent and its assigned role.
- `memory_manager`: Instance of `MemoryManager` for managing summary-related data.
- `model`, `temperature`, `max_tokens`: Configuration for the OpenAI LLM, optimized for summarization.

Core Methods:
1. Summary Creation:
   - `create_summary`: Generate a summary for a document segment.
   - `create_progressive_summary`: Create summaries at various levels of detail.
   - `merge_summaries`: Integrate multiple segment summaries into a cohesive whole.

Custom Actions:
- `_action_create_summary`: Action to generate a summary based on given parameters.
- `_action_create_progressive_summary`: Action to generate multiple summaries at different levels.
- `_action_merge_summaries`: Action to merge several summaries into a unified format.
- `_action_store_summary`: Store a generated summary with metadata.

Integration Notes:
- Extends `BaseAgent`, inheriting core functionalities such as communication, reasoning, and memory integration.
- Designed to support dynamic workflows managed by the Coordinator Agent.
- Pre-defined summary styles ensure flexibility for various use cases (e.g., executive, actionable reports).
"""

from typing import Dict, Any, List, Callable
import json
import uuid
from datetime import datetime
from agents.base_agent import BaseAgent
from memory.memory_manager import MemoryManager


class SummarizationAgent(BaseAgent):
    """
    Summarization Agent responsible for creating concise summaries of document
    segments at different levels of detail and focus.
    """
    def __init__(
        self,
        agent_id: str,
        name: str,
        memory_manager: MemoryManager,
        model: str = "gpt-4o-mini",
        temperature: float = 0.4,
        max_tokens: int = 1500  # Higher token limit for summaries
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            role="Document Summarization Specialist",
            memory_manager=memory_manager,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Set summarization-specific instructions
        summarization_instructions = """
        As the Summarization Agent, your primary responsibilities are:

        1. Create concise summaries of document segments at different levels of detail:
           - Brief summaries (1-2 sentences)
           - Executive summaries (1-2 paragraphs)
           - Comprehensive summaries (multiple paragraphs)

        2. Generate targeted summaries with specific focus areas:
           - General summary (main points)
           - Topic-focused summary (specific subjects)
           - Action-oriented summary (tasks, recommendations)
           - Technical summary (technical details, specifications)

        3. Maintain important information while reducing content length

        4. Preserve the original meaning and intent of the document

        5. Highlight key insights and central themes

        Your summaries should be clear, accurate, and maintain the tone appropriate
        for the document type. Avoid adding interpretations not supported by the text.
        """

        self.set_additional_instructions(summarization_instructions)

        # Summary style presets
        self.summary_styles = {
            "executive": {
                "description": "Brief, high-level overview for executives",
                "focus": "Key insights, business impact, strategic implications",
                "length": "1-2 paragraphs"
            },
            "technical": {
                "description": "Technical details for domain experts",
                "focus": "Methods, data, specifications, technical findings",
                "length": "3-4 paragraphs"
            },
            "action": {
                "description": "Action-oriented summary focusing on tasks and next steps",
                "focus": "Actions, recommendations, decisions needed",
                "length": "1-3 paragraphs with bullet points"
            },
            "comprehensive": {
                "description": "Detailed overview covering all major aspects",
                "focus": "Comprehensive coverage of main points, supporting details, and context",
                "length": "4-6 paragraphs"
            },
            "brief": {
                "description": "Ultra-concise summary of core message",
                "focus": "Essential point only",
                "length": "1-2 sentences"
            }
        }

    def create_summary(self, text: str, style: str = "comprehensive",
                       max_length: int = None, focus_topics: List[str] = None) -> Dict[str, Any]:
        """
        Create a summary of the text based on specified style and parameters
        """
        # Select summary style parameters
        style_params = self.summary_styles.get(style, self.summary_styles["comprehensive"])

        # Build the summarization prompt
        summary_prompt = f"""
        Create a {style} summary of the following text.
        
        Style description: {style_params['description']}
        Focus on: {style_params['focus']}
        Target length: {style_params['length'] if not max_length else f'Maximum {max_length} words'}

        {f'Focus especially on these topics: {", ".join(focus_topics)}' if focus_topics else ''}

        Maintain the original meaning and highlight key insights.

        Text to summarize:
        ```
        {text}
        ```

        Write only the summary text, without headings or labels.
        """

        # Think through the summarization process
        thinking_input = {
            "text": text,
            "style": style,
            "style_params": style_params,
            "max_length": max_length,
            "focus_topics": focus_topics
        }

        thinking_result = self.think(
            thinking_input,
            context_query="document summarization",
            reasoning_prompt=summary_prompt,
            action_options=["store_summary", "communicate"]
        )

        # Create summary metadata
        summary_metadata = {
            "style": style,
            "max_length": max_length,
            "focus_topics": focus_topics,
            "creation_time": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            "tokens_used": thinking_result["reasoning"].get("tokens", {})
        }

        # Return the summary and metadata
        return {
            "summary": thinking_result["reasoning"]["output"],
            "metadata": summary_metadata
        }

    def create_progressive_summary(self, text: str, levels: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create multiple summaries at different levels of detail/complexity
        """
        if not levels:
            levels = ["brief", "executive", "comprehensive"]

        summaries = []

        for level in levels:
            summary = self.create_summary(text, style=level)
            summaries.append({
                "level": level,
                "summary": summary["summary"],
                "metadata": summary["metadata"]
            })

        return {
            "text_sample": text[:100],  # Include a sample of the original text
            "summaries": summaries,
            "creation_time": datetime.now().isoformat(),
            "agent_id": self.agent_id
        }

    def merge_summaries(self, summaries: List[Dict[str, Any]], style: str = "comprehensive") -> Dict[str, Any]:
        """
        Merge multiple segment summaries into a cohesive document-level summary
        """
        # Extract text from summaries
        summary_texts = []
        for summary in summaries:
            if isinstance(summary, dict) and "summary" in summary:
                summary_texts.append(summary["summary"])
            elif isinstance(summary, str):
                summary_texts.append(summary)

        # Join the summary texts
        combined_text = "\n\n".join(summary_texts)

        if self.memory.count_tokens(combined_text) > self.max_tokens:
            combined_text = self.memory.truncate_text(combined_text, self.max_tokens)

        # Create a prompt for merging
        merge_prompt = f"""
        Create a cohesive {style} summary that integrates these segment summaries
        from different parts of a document.

        Maintain consistent tone and narrative flow. Eliminate redundancies and
        organize information logically. The final summary should read as a unified
        piece rather than disconnected segments.

        Segment summaries:
        ```
        {combined_text}
        ```

        Write a cohesive integrated summary that synthesizes these segments.
        """

        # Think through the merging process
        thinking_input = {
            "summary_texts": summary_texts,
            "style": style
        }

        thinking_result = self.think(
            thinking_input,
            context_query="summary integration",
            reasoning_prompt=merge_prompt,
            action_options=["store_summary", "communicate"]
        )

        # Create merged summary metadata
        merged_metadata = {
            "style": style,
            "source_count": len(summaries),
            "creation_time": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            "merged": True,
            "tokens_used": thinking_result["reasoning"].get("tokens", {})
        }

        # Return the merged summary and metadata
        return {
            "summary": thinking_result["reasoning"]["output"],
            "metadata": merged_metadata
        }

    # Custom actions for this agent

    def _action_create_summary(self,
                               text: str,
                               style: str = "comprehensive",
                               max_length: int = None,
                               focus_topics: List[str] = None) -> Dict[str, Any]:
        """
        Action to create a summary
        """
        summary_result = self.create_summary(text, style, max_length, focus_topics)

        # Store summary in memory
        summary_key = f"summary_{style}_{self.agent_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.memory.store_working_data(summary_key, {
            "summary": summary_result["summary"],
            "text_sample": text[:100],
            "style": style,
            "creation_time": datetime.now().isoformat()
        })

        return {
            "success": True,
            "summary": summary_result["summary"],
            "metadata": summary_result["metadata"]
        }

    def _action_create_progressive_summary(self,
                                           text: str,
                                           levels: List[str] = None) -> Dict[str, Any]:
        """
        Action to create progressive summaries
        """
        progressive_result = self.create_progressive_summary(text, levels)

        # Store summaries in memory
        progressive_key = f"progressive_summary_{self.agent_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.memory.store_working_data(progressive_key, progressive_result)

        return {
            "success": True,
            "summaries": progressive_result["summaries"],
            "levels": [s["level"] for s in progressive_result["summaries"]]
        }

    def _action_merge_summaries(self,
                                summaries: List[Dict[str, Any]],
                                style: str = "comprehensive") -> Dict[str, Any]:
        """
        Action to merge multiple summaries
        """
        merged_result = self.merge_summaries(summaries, style)

        # Store merged summary in memory
        merged_key = f"merged_summary_{self.agent_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.memory.store_working_data(merged_key, {
            "summary": merged_result["summary"],
            "source_count": len(summaries),
            "style": style,
            "creation_time": datetime.now().isoformat()
        })

        return {
            "success": True,
            "summary": merged_result["summary"],
            "metadata": merged_result["metadata"]
        }

    def _action_store_summary(self, summary: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Action to store a summary in memory
        """
        summary_id = f"summary_{uuid.uuid4()}"

        summary_data = {
            "summary": summary,
            "metadata": metadata or {},
            "creation_time": datetime.now().isoformat(),
            "agent_id": self.agent_id
        }

        # Store in working memory
        self.memory.store_working_data(summary_id, summary_data)

        return {
            "success": True,
            "summary_id": summary_id
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

        summarization_actions = {
            "create_summary": self._action_create_summary,
            "create_progressive_summary": self._action_create_progressive_summary,
            "merge_summaries": self._action_merge_summaries,
            "store_summary": self._action_store_summary
        }

        return {**base_actions, **summarization_actions}

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task assigned by the coordinator
        """
        task_type = task.get("task_type")
        segment_id = task.get("segment_id")
        parameters = task.get("parameters", {})

        # Retrieve the document segment content
        segment_data = self.memory.retrieve_document_segment(segment_id) if segment_id else None
        text_content = segment_data.get("content") if segment_data else parameters.get("text", "")

        # Process based on task type
        supported_task_types = {"create_summary", "create_progressive_summary", "merge_summaries"}
        if task_type not in supported_task_types:
            result = {
                "success": False,
                "error": f"Unsupported task type: {task_type}. Supported types are {', '.join(supported_task_types)}"
            }
        elif task_type == "create_summary":
            result = self._action_create_summary(
                text_content,
                style=parameters.get("style", "comprehensive"),
                max_length=parameters.get("max_length"),
                focus_topics=parameters.get("focus_topics")
            )
        elif task_type == "create_progressive_summary":
            result = self._action_create_progressive_summary(
                text_content,
                levels=parameters.get("levels")
            )
        elif task_type == "merge_summaries":
            result = self._action_merge_summaries(
                parameters.get("summaries", []),
                style=parameters.get("style", "comprehensive")
            )

        # Report results back to coordinator
        self.send_message(parameters.get("coordinator_id", "coordinator"), json.dumps({
            "task_id": task.get("task_id"),
            "result": result,
            "status": "completed" if result.get("success", False) else "failed"
        }), {
            "message_type": "task_result",
            "task_id": task.get("task_id")
        })

        return result
