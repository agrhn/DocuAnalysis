"""
Module: agents/critical_analysis_agent.py

Purpose:
Defines the `CriticalAnalysisAgent` class, a specialized agent for evaluating document content quality,
identifying biases, analyzing argument structures, providing alternative perspectives, and generating
critical reports to improve the relevance, fairness, and overall quality of document content.

Key Responsibilities:
1. Content Quality Analysis:
   - Assess logical coherence, evidence quality, and argument structure.
   - Detect inconsistencies, unsupported claims, and areas for improvement.

2. Bias Detection:
   - Identify potential biases, such as perspective, selection, or language biases.
   - Provide severity ratings and recommendations for improving balance.

3. Argumentation Analysis:
   - Map argument structures (claims, evidence, counterarguments).
   - Identify logical fallacies and evaluate argument strength.

4. Alternative Perspectives:
   - Suggest missing viewpoints, overlooked factors, or complementary perspectives.
   - Provide actionable recommendations to improve diversity of thought.

5. Report Generation:
   - Generate synthesized critical analysis reports across various document segments.
   - Integrate findings from multiple analyses into cohesive recommendations.

Constructor Parameters:
- `agent_id`, `name`: Identifies the Critical Analysis Agent and its assigned role.
- `memory_manager`: Instance of `MemoryManager` for managing analysis-related data.
- `model`, `temperature`, `max_tokens`: Configuration for the underlying OpenAI LLM,
  optimized for nuanced critical analysis.

Core Methods:
1. Analysis Methods:
   - `analyze_content_quality`: Assess reasoning, evidence, and structure.
   - `identify_bias`: Detect and rate the presence of different types of biases.
   - `analyze_argumentation`: Evaluate argumentation structure and logical validity.
   - `provide_alternative_perspectives`: Suggest alternative viewpoints on the content.

2. Report Methods:
   - `generate_critical_report`: Combine multiple segment analyses into a synthesized critical report.

3. Task-Oriented Methods:
   - `process_task`: Process tasks delegated by the Coordinator, including quality analysis, bias detection,
     and comprehensive analysis.
   - `_action_*` Methods: Support task-specific actions for quality, bias, argumentation, and perspectives analysis.

Integration Notes:
- Extends `BaseAgent`, inheriting core functionalities such as communication, reasoning, and memory integration.
- Designed to analyze complex content across multiple dimensions (bias, argumentation, evidence).
- Supports workflows managed by the Coordinator Agent, facilitating collaborative document insights.
"""
from typing import Dict, Any, List, Callable
import json
import uuid
import re
from datetime import datetime
from agents.base_agent import BaseAgent
from memory.memory_manager import MemoryManager


class CriticalAnalysisAgent(BaseAgent):
    """
    Critical Analysis Agent responsible for evaluating document content quality,
    identifying bias, assessing arguments, and providing critical perspectives.
    """
    def __init__(
        self,
        agent_id: str,
        name: str,
        memory_manager: MemoryManager,
        model: str = "gpt-4o-mini",
        temperature: float = 0.6,  # Slightly higher temperature for more nuanced analysis
        max_tokens: int = 1500
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            role="Critical Content Analyst",
            memory_manager=memory_manager,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Set critical analysis-specific instructions
        critical_analysis_instructions = """
        As the Critical Analysis Agent, your primary responsibilities are:

        1. Evaluate the quality of content in document segments:
           - Assess logical coherence and reasoning
           - Identify unsupported claims or assumptions
           - Evaluate evidence quality and relevance
           - Check for inconsistencies or contradictions

        2. Identify potential bias in content:
           - Detect perspective bias and framing
           - Note selection bias in evidence or examples
           - Identify loaded language and emotional appeals
           - Recognize cultural or demographic assumptions

        3. Analyze argumentation structure:
           - Map claim-evidence relationships
           - Identify fallacies or logical errors
           - Assess strength of arguments
           - Evaluate counterarguments and limitations

        4. Provide constructive critical perspectives:
           - Suggest alternative viewpoints
           - Identify overlooked factors or stakeholders
           - Suggest additional context needed
           - Highlight strengths alongside weaknesses

        Your analysis should be balanced, fair, and focus on substance rather than style.
        Provide specific examples from the text to support your analysis.
        """

        self.set_additional_instructions(critical_analysis_instructions)

        # Analysis frameworks
        self.analysis_frameworks = {
            "bias": [
                "perspective_bias", "selection_bias", "framing_bias",
                "language_bias", "cultural_bias", "confirmation_bias"
            ],
            "logical_fallacies": [
                "ad_hominem", "appeal_to_authority", "appeal_to_emotion",
                "false_dichotomy", "hasty_generalization", "slippery_slope",
                "circular_reasoning", "correlation_causation", "straw_man"
            ],
            "evidence_quality": [
                "relevance", "sufficiency", "representativeness",
                "accuracy", "recency", "reliability_of_sources"
            ],
            "argument_structure": [
                "clarity_of_claims", "supporting_evidence", "counterarguments",
                "qualifications", "conclusion_validity", "overall_coherence"
            ]
        }

    def analyze_content_quality(self, text: str, focus_areas: List[str] = None) -> Dict[str, Any]:
        """
        Analyze the quality of content in terms of reasoning, evidence, and structure
        """
        # Default focus areas
        if not focus_areas:
            focus_areas = ["evidence", "reasoning", "structure"]

        # Build analysis prompt
        quality_prompt = f"""
        Analyze the quality of the following content, focusing on:
        {', '.join(focus_areas)}

        For each focus area, provide:
        1. An overall rating (1-5, where 5 is highest quality)
        2. Specific strengths with examples from the text
        3. Specific weaknesses with examples from the text
        4. Suggestions for improvement

        Format your analysis as a JSON object with these sections.

        Text to analyze:
        ```
        {text}
        ```
        """

        # Think through the analysis process
        thinking_input = {
            "text": text,
            "focus_areas": focus_areas
        }

        thinking_result = self.think(
            thinking_input,
            context_query="content quality analysis",
            reasoning_prompt=quality_prompt,
            action_options=["store_analysis", "communicate"]
        )

        # Try to parse the analysis from the reasoning output
        try:
            # Look for JSON object or fallback to structured interpretation
            analysis_match = re.search(r'{\s*"[^"]+"\s*:.*}', thinking_result["reasoning"]["output"], re.DOTALL)
            if analysis_match:
                analysis_json = analysis_match.group(0)
                analysis = json.loads(analysis_json)
            else:
                # Construct structured fallback with output text
                analysis = {
                    "overall_assessment": thinking_result["reasoning"]["output"],
                    "focus_areas": focus_areas,
                    "structured": False
                }

            # Add metadata to the analysis
            analysis["analysis_time"] = datetime.now().isoformat()
            analysis["analysis_agent"] = self.agent_id

            return analysis

        except json.JSONDecodeError as e:
            # Log full error and reasoning output for better debugging
            return {
                "success": False,
                "error": f"JSON parsing failed: {str(e)}",
                "raw_output": thinking_result["reasoning"]["output"],
                "focus_areas": focus_areas,
                "structured": False,
                "analysis_time": datetime.now().isoformat(),
                "analysis_agent": self.agent_id
            }

    def identify_bias(self, text: str, bias_types: List[str] = None) -> Dict[str, Any]:
        """
        Identify potential biases in the content
        """
        # Default to all bias types if none specified
        if not bias_types:
            bias_types = self.analysis_frameworks["bias"]

        # Build bias analysis prompt
        bias_prompt = f"""
        Analyze the following text for potential biases, focusing on:
        {', '.join(bias_types)}

        For each type of bias:
        1. Indicate if present (Yes/No/Maybe)
        2. Rate severity (1-5, where 5 is strongest bias)
        3. Provide specific examples from the text
        4. Suggest how the content could be more balanced

        Format your analysis as a JSON object with these bias types as keys.

        Text to analyze:
        ```
        {text}
        ```
        """

        # Think through the bias analysis
        thinking_input = {
            "text": text,
            "bias_types": bias_types
        }

        thinking_result = self.think(
            thinking_input,
            context_query="bias analysis",
            reasoning_prompt=bias_prompt,
            action_options=["store_analysis", "communicate"]
        )

        # Try to parse the bias analysis from the reasoning output
        try:
            # Look for JSON object in the text
            bias_match = re.search(r'{\s*"[^"]+"\s*:.*}', thinking_result["reasoning"]["output"], re.DOTALL)
            if bias_match:
                bias_json = bias_match.group(0)
                bias_analysis = json.loads(bias_json)
            else:
                # Create structured analysis from the output
                bias_analysis = {
                    "overall_assessment": thinking_result["reasoning"]["output"],
                    "bias_types": bias_types,
                    "structured": False
                }

            # Add metadata to analysis
            bias_analysis["analysis_time"] = datetime.now().isoformat()
            bias_analysis["analysis_agent"] = self.agent_id

            return bias_analysis

        except json.JSONDecodeError:
            return {
                "overall_assessment": thinking_result["reasoning"]["output"],
                "bias_types": bias_types,
                "structured": False,
                "analysis_time": datetime.now().isoformat(),
                "analysis_agent": self.agent_id
            }

    def analyze_argumentation(self, text: str) -> Dict[str, Any]:
        """
        Analyze the argumentation structure and logical validity of content
        """
        # Build argumentation analysis prompt
        arg_prompt = """
        Analyze the argumentation structure of the following text:

        1. Identify the main claims/thesis
        2. Map supporting arguments and evidence
        3. Identify any counterarguments addressed
        4. Analyze logical structure and reasoning patterns
        5. Identify any logical fallacies present (provide specific examples)
        6. Rate overall argument strength (1-5, with 5 being strongest)
        7. Suggest how the argumentation could be improved

        Format your analysis as a structured JSON object with these sections.

        Text to analyze:
        ```
        {text}
        ```
        """

        # Think through the argumentation analysis
        thinking_input = {"text": text}

        thinking_result = self.think(
            thinking_input,
            context_query="argumentation analysis",
            reasoning_prompt=arg_prompt.format(text=text),
            action_options=["store_analysis", "communicate"]
        )

        # Try to parse the analysis from the reasoning output
        try:
            # Look for JSON object in the text
            arg_match = re.search(r'{\s*"[^"]+"\s*:.*}', thinking_result["reasoning"]["output"], re.DOTALL)
            if arg_match:
                arg_json = arg_match.group(0)
                arg_analysis = json.loads(arg_json)
            else:
                # Create structured analysis from the output
                arg_analysis = {
                    "overall_assessment": thinking_result["reasoning"]["output"],
                    "structured": False
                }

            # Add metadata to analysis
            arg_analysis["analysis_time"] = datetime.now().isoformat()
            arg_analysis["analysis_agent"] = self.agent_id

            return arg_analysis

        except json.JSONDecodeError:
            return {
                "overall_assessment": thinking_result["reasoning"]["output"],
                "structured": False,
                "analysis_time": datetime.now().isoformat(),
                "analysis_agent": self.agent_id
            }

    def provide_alternative_perspectives(self, text: str, num_perspectives: int = 3) -> Dict[str, Any]:
        """
        Provide alternative viewpoints or perspectives on the content
        """
        # Build alternative perspectives prompt
        perspectives_prompt = f"""
        Analyze the following text and provide {num_perspectives} alternative perspectives
        or viewpoints that might challenge or complement the content.

        For each alternative perspective:
        1. Briefly describe the perspective
        2. Explain how it differs from the original content
        3. Provide key points this perspective would emphasize
        4. Note what evidence or considerations support this perspective

        Format your response as a JSON object with numbered perspectives.

        Text to analyze:
        ```
        {text}
        ```
        """

        # Think through alternative perspectives
        thinking_input = {
            "text": text,
            "num_perspectives": num_perspectives
        }

        thinking_result = self.think(
            thinking_input,
            context_query="alternative perspectives",
            reasoning_prompt=perspectives_prompt,
            action_options=["store_analysis", "communicate"]
        )

        # Try to parse the perspectives from the reasoning output
        try:
            # Look for JSON object in the text
            perspectives_match = re.search(r'{\s*"[^"]+"\s*:.*}', thinking_result["reasoning"]["output"], re.DOTALL)
            if perspectives_match:
                perspectives_json = perspectives_match.group(0)
                perspectives = json.loads(perspectives_json)
            else:
                # Create structured perspectives from the output
                perspectives = {
                    "overall_assessment": thinking_result["reasoning"]["output"],
                    "structured": False
                }

            # Add metadata to perspectives
            perspectives["analysis_time"] = datetime.now().isoformat()
            perspectives["analysis_agent"] = self.agent_id

            return perspectives

        except json.JSONDecodeError:
            return {
                "overall_assessment": thinking_result["reasoning"]["output"],
                "structured": False,
                "analysis_time": datetime.now().isoformat(),
                "analysis_agent": self.agent_id
            }

    # Custom actions for this agent

    def _action_analyze_content_quality(self, text: str, focus_areas: List[str] = None) -> Dict[str, Any]:
        """
        Action to analyze content quality
        """
        analysis = self.analyze_content_quality(text, focus_areas)

        # Store analysis in memory
        analysis_key = f"quality_analysis_{self.agent_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.memory.store_working_data(analysis_key, {
            "analysis": analysis,
            "text_sample": text[:100],
            "analysis_type": "content_quality",
            "analysis_time": datetime.now().isoformat()
        })

        return {
            "success": True,
            "analysis": analysis
        }

    def _action_identify_bias(self, text: str, bias_types: List[str] = None) -> Dict[str, Any]:
        """
        Action to identify bias in content
        """
        bias_analysis = self.identify_bias(text, bias_types)

        # Store analysis in memory
        bias_key = f"bias_analysis_{self.agent_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.memory.store_working_data(bias_key, {
            "analysis": bias_analysis,
            "text_sample": text[:100],
            "analysis_type": "bias",
            "analysis_time": datetime.now().isoformat()
        })

        return {
            "success": True,
            "analysis": bias_analysis
        }

    def _action_analyze_argumentation(self, text: str) -> Dict[str, Any]:
        """
        Action to analyze argumentation structure
        """
        arg_analysis = self.analyze_argumentation(text)

        # Store analysis in memory
        arg_key = f"argumentation_analysis_{self.agent_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.memory.store_working_data(arg_key, {
            "analysis": arg_analysis,
            "text_sample": text[:100],
            "analysis_type": "argumentation",
            "analysis_time": datetime.now().isoformat()
        })

        return {
            "success": True,
            "analysis": arg_analysis
        }

    def _action_provide_alternative_perspectives(self,
                                                 text: str,
                                                 num_perspectives: int = 3) -> Dict[str, Any]:
        """
        Action to provide alternative perspectives
        """
        perspectives = self.provide_alternative_perspectives(text, num_perspectives)

        # Store analysis in memory
        perspectives_key = f"perspectives_{self.agent_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.memory.store_working_data(perspectives_key, {
            "analysis": perspectives,
            "text_sample": text[:100],
            "analysis_type": "alternative_perspectives",
            "analysis_time": datetime.now().isoformat()
        })

        return {
            "success": True,
            "analysis": perspectives
        }

    def _action_store_analysis(self, analysis: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        """
        Action to store an analysis in memory
        """
        analysis_id = f"analysis_{analysis_type}_{uuid.uuid4()}"

        analysis_data = {
            "analysis": analysis,
            "analysis_type": analysis_type,
            "creation_time": datetime.now().isoformat(),
            "agent_id": self.agent_id
        }

        # Store in working memory
        self.memory.store_working_data(analysis_id, analysis_data)

        return {
            "success": True,
            "analysis_id": analysis_id
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

        analysis_actions = {
            "analyze_content_quality": self._action_analyze_content_quality,
            "identify_bias": self._action_identify_bias,
            "analyze_argumentation": self._action_analyze_argumentation,
            "provide_alternative_perspectives": self._action_provide_alternative_perspectives,
            "store_analysis": self._action_store_analysis
        }

        return {**base_actions, **analysis_actions}

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
        if task_type == "analyze_quality":
            focus_areas = parameters.get("focus_areas", ["evidence", "reasoning", "structure"])
            result = self._action_analyze_content_quality(text_content, focus_areas)

        elif task_type == "identify_bias":
            bias_types = parameters.get("bias_types", None)  # Default to all in method
            result = self._action_identify_bias(text_content, bias_types)

        elif task_type == "analyze_argumentation":
            result = self._action_analyze_argumentation(text_content)

        elif task_type == "provide_perspectives":
            num_perspectives = parameters.get("num_perspectives", 3)
            result = self._action_provide_alternative_perspectives(text_content, num_perspectives)

        elif task_type == "comprehensive_analysis":
            # Perform a full analysis with all methods
            quality_result = self._action_analyze_content_quality(text_content)
            bias_result = self._action_identify_bias(text_content)
            argumentation_result = self._action_analyze_argumentation(text_content)
            perspectives_result = self._action_provide_alternative_perspectives(text_content)

            # Combine all results
            result = {
                "success": True,
                "quality_analysis": quality_result["analysis"],
                "bias_analysis": bias_result["analysis"],
                "argumentation_analysis": argumentation_result["analysis"],
                "alternative_perspectives": perspectives_result["analysis"],
                "segment_id": segment_id
            }

            # Store the comprehensive analysis
            analysis_id = f"comprehensive_analysis_{self.agent_id}_{uuid.uuid4()}"
            self.memory.store_working_data(analysis_id, {
                "analysis": result,
                "text_sample": text_content[:100] if text_content else "",
                "analysis_type": "comprehensive",
                "segment_id": segment_id,
                "creation_time": datetime.now().isoformat()
            })

        else:
            result = {
                "success": False,
                "error": f"Unknown task type: {task_type}",
                "segment_id": segment_id
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

    def generate_critical_report(self, segment_ids: List[str] = None,
                                 analysis_types: List[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive critical report based on multiple analyses

        Args:
            segment_ids: List of document segment IDs to include in report
            analysis_types: Types of analyses to include (quality, bias, argumentation, perspectives)

        Returns:
            Dict containing the report and metadata
        """
        # Default to all analysis types if none specified
        if not analysis_types:
            analysis_types = list(self.analysis_frameworks.keys())

        # Retrieve relevant analyses from memory
        analyses = []

        # If segment IDs are provided, retrieve analyses for those segments
        if segment_ids:
            for segment_id in segment_ids:
                segment_data = self.memory.retrieve_document_segment(segment_id)
                if not segment_data:
                    continue

                # Query memory for analyses related to this segment
                for analysis_type in analysis_types:
                    query = f"segment_id:{segment_id} analysis_type:{analysis_type}"
                    analysis_results = self.memory.get_working_data(query)
                    analyses.extend(analysis_results)

        # If no segment IDs or no analyses found, retrieve most recent analyses by type
        if not segment_ids or not analyses:
            for analysis_type in analysis_types:
                query = f"analysis_type:{analysis_type}"
                recent_analyses = self.memory.get_working_data(
                    query, sort_by="creation_time", limit=5
                )
                analyses.extend(recent_analyses)

        # If still no analyses found, return error
        if not analyses:
            return {
                "success": False,
                "error": "No analyses found to generate report",
                "report_time": datetime.now().isoformat()
            }

        # Build report prompt
        report_prompt = f"""
        Generate a comprehensive critical analysis report based on the following analyses:

        {json.dumps(analyses, indent=2)}

        Your report should:
        1. Synthesize insights across different analyses
        2. Identify patterns in quality issues, biases, or argumentative strengths/weaknesses
        3. Provide high-level recommendations for improving the content
        4. Note any contradictions or differing perspectives across analyses

        Format your report with clear sections, maintaining a balanced and fair perspective.
        """

        # Generate the report
        thinking_input = {
            "analyses": analyses,
            "analysis_types": analysis_types
        }

        thinking_result = self.think(
            thinking_input,
            context_query="critical analysis report",
            reasoning_prompt=report_prompt,
            action_options=["communicate"]
        )

        # Prepare report result
        report = {
            "content": thinking_result["reasoning"]["output"],
            "analysis_count": len(analyses),
            "analysis_types": analysis_types,
            "segment_ids": segment_ids if segment_ids else [],
            "report_time": datetime.now().isoformat(),
            "agent_id": self.agent_id
        }

        # Store the report
        report_id = f"critical_report_{self.agent_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.memory.store_working_data(report_id, {
            "report": report,
            "source_analyses": [a.get("id", "unknown") for a in analyses],
            "report_type": "critical_analysis",
            "creation_time": datetime.now().isoformat()
        })

        return {
            "success": True,
            "report": report,
            "report_id": report_id
        }
