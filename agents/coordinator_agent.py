"""
Module: agents/coordinator_agent.py

Purpose:
Defines the `CoordinatorAgent` class, a specialized agent responsible for orchestrating
the document analysis workflow, delegating tasks to other agents, and maintaining a unified
global state for all ongoing tasks and analyses.

Key Responsibilities:
1. Workflow Orchestration:
   - Initialize workflows with document-specific metadata.
   - Segment documents into manageable parts for analysis.
   - Transition workflows through different phases (e.g., loading → analysis → results compilation).
   - Compile and integrate analysis results from specialized agents.

2. Task Delegation:
   - Assign specific tasks (e.g., extraction, summarization) to specialized agents.
   - Track task progress, update task statuses, and handle task completions.

3. Workflow State Management:
   - Maintain a centralized record of the workflow state, including:
     - Current phase
     - Task statuses and assignments
     - Document segment progress
     - Analysis results
   - Continuously update and persist the state using working memory.

4. Communication:
   - Communicate tasks and instructions to specialized agents clearly.
   - Broadcast phase transitions or status updates to all agents.
   - Log all major actions, including task delegation, workflow transitions, and result integrations.

Constructor Parameters:
- `agent_id`, `name`: Identify and name the Coordinator Agent.
- `memory_manager`: Instance of `MemoryManager` for managing global state and task data.
- `model`, `temperature`, `max_tokens`: Configuration for the OpenAI LLM (used minimally for coordination tasks).

Core Methods:
1. Workflow Management:
   - `initialize_workflow`, `segment_document`, `transition_phase`, `compile_analysis_results`
   - `check_workflow_progress`

2. Task Management:
   - `assign_task`, `update_task_status`

3. Custom Actions:
   - `_action_delegate_task`, `_action_check_status`, `_action_integrate_results`

4. Utilities:
   - `get_available_actions`: Returns available action handlers tailored to the Coordinator's role.

Integration Notes:
- Acts as the hub of the document analysis system, coordinating collaboration between specialized agents.
- Maintains consistent state by leveraging Working Memory to store workflow-specific data.
- Extends `BaseAgent`, inheriting core thinking and communication capabilities.
"""

from typing import Dict, Any, List, Callable
import json
import uuid
from datetime import datetime
from agents.base_agent import BaseAgent
from memory.memory_manager import MemoryManager


class CoordinatorAgent(BaseAgent):
    """
    Coordinator Agent responsible for orchestrating the document analysis workflow,
    delegating tasks to specialized agents, and maintaining global state.
    """
    def __init__(
        self,
        agent_id: str,
        name: str,
        memory_manager: MemoryManager,
        model: str = "gpt-4o-mini",
        temperature: float = 0.5,
        max_tokens: int = 1000
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            role="Document Analysis Coordinator",
            memory_manager=memory_manager,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Set coordinator-specific instructions
        coordinator_instructions = """
        As the Coordinator Agent, your primary responsibilities are:

        1. Orchestrate the document analysis workflow
        2. Assign tasks to specialized agents based on their expertise
        3. Track progress of document analysis
        4. Integrate results from different agents
        5. Maintain global view of the document analysis state
        6. Provide status updates to users
        7. Handle transitions between analysis phases

        You should maintain awareness of:
        - Which document sections have been analyzed
        - Which agents are working on what tasks
        - What insights have already been discovered
        - What areas need further analysis

        When communicating with other agents, be clear and specific about:
        - The exact task they should perform
        - The context they need to complete their task
        - Any deadlines or priorities
        - How their work fits into the overall analysis
        """

        self.set_additional_instructions(coordinator_instructions)

        # Initialize workflow state
        self.workflow_state = {
            "current_phase": "initialization",
            "document_id": None,
            "document_segments": [],
            "analyzed_segments": [],
            "active_tasks": {},
            "completed_tasks": [],
            "agent_assignments": {},
            "analysis_results": {}
        }

    def initialize_workflow(self, document_id: str, document_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize a new document analysis workflow
        """
        workflow_id = f"workflow_{uuid.uuid4()}"

        # Update workflow state
        self.workflow_state = {
            "workflow_id": workflow_id,
            "document_id": document_id,
            "document_metadata": document_metadata,
            "current_phase": "document_loading",
            "start_time": datetime.now().isoformat(),
            "document_segments": [],
            "analyzed_segments": [],
            "active_tasks": {},
            "completed_tasks": [],
            "agent_assignments": {},
            "analysis_results": {}
        }

        # Store workflow state in working memory
        self.memory.store_working_data(f"workflow_{workflow_id}", self.workflow_state)

        self._log_action("workflow", "initialize", document_id, f"Started workflow: {workflow_id}")
        return self.workflow_state

    def segment_document(self, document_content: str, segment_size: int = 2000) -> List[Dict[str, Any]]:
        """
        Split document into manageable segments for processing
        """
        # Simple character-based segmentation (could be improved with semantic segmentation)
        segments = []
        content_length = len(document_content)

        for i in range(0, content_length, segment_size):
            end_idx = min(i + segment_size, content_length)
            segment_content = document_content[i:end_idx]

            segment_id = f"segment_{len(segments) + 1}"
            segments.append({
                "segment_id": segment_id,
                "start_idx": i,
                "end_idx": end_idx,
                "content": segment_content,
                "status": "unprocessed"
            })

        # Update workflow state
        self.workflow_state["document_segments"] = segments
        self.memory.store_working_data(f"workflow_{self.workflow_state['workflow_id']}", self.workflow_state)

        self._log_action("workflow", "segment_document", self.workflow_state["document_id"],
                         f"Created {len(segments)} segments")
        return segments

    def assign_task(self, agent_id: str, task_type: str, segment_id: str = None,
                    parameters: Dict[str, Any] = None) -> str:
        """
        Assign a task to a specific agent
        """
        task_id = f"task_{uuid.uuid4()}"

        task = {
            "task_id": task_id,
            "agent_id": agent_id,
            "task_type": task_type,
            "segment_id": segment_id,
            "parameters": parameters or {},
            "status": "assigned",
            "assigned_time": datetime.now().isoformat(),
            "completion_time": None,
            "result": None
        }

        # Update workflow state
        self.workflow_state["active_tasks"][task_id] = task

        if agent_id not in self.workflow_state["agent_assignments"]:
            self.workflow_state["agent_assignments"][agent_id] = []

        self.workflow_state["agent_assignments"][agent_id].append(task_id)

        # If task is for a specific segment, update segment status
        if segment_id:
            for i, segment in enumerate(self.workflow_state["document_segments"]):
                if segment["segment_id"] == segment_id:
                    self.workflow_state["document_segments"][i]["status"] = "in_progress"
                    break

        # Store updated workflow state
        self.memory.store_working_data(f"workflow_{self.workflow_state['workflow_id']}", self.workflow_state)

        # Send task message to agent
        task_message = {
            "task_id": task_id,
            "task_type": task_type,
            "segment_id": segment_id,
            "parameters": parameters or {}
        }

        self.send_message(agent_id, json.dumps(task_message), {
            "message_type": "task_assignment",
            "task_id": task_id
        })

        self._log_action("workflow", "assign_task", agent_id,
                         f"Task {task_id} ({task_type}) assigned to {agent_id}")
        return task_id

    def update_task_status(self, task_id: str, status: str, result: Any = None) -> Dict[str, Any]:
        """
        Update the status of a task in the workflow
        """
        if task_id not in self.workflow_state["active_tasks"]:
            return {"success": False, "error": f"Task {task_id} not found"}

        task = self.workflow_state["active_tasks"][task_id]
        task["status"] = status

        if status == "completed":
            task["completion_time"] = datetime.now().isoformat()
            task["result"] = result

            # Move from active to completed
            self.workflow_state["completed_tasks"].append(task)
            del self.workflow_state["active_tasks"][task_id]

            # If task was for a specific segment, update segment status
            if task["segment_id"]:
                for i, segment in enumerate(self.workflow_state["document_segments"]):
                    if segment["segment_id"] == task["segment_id"]:
                        self.workflow_state["document_segments"][i]["status"] = "processed"
                        if segment["segment_id"] not in self.workflow_state["analyzed_segments"]:
                            self.workflow_state["analyzed_segments"].append(segment["segment_id"])
                        break

        # Store updated workflow state
        self.memory.store_working_data(f"workflow_{self.workflow_state['workflow_id']}", self.workflow_state)

        self._log_action("workflow", "update_task", task_id,
                         f"Task status updated to {status}")
        return task

    def check_workflow_progress(self) -> Dict[str, Any]:
        """
        Check the progress of the current workflow
        """
        total_segments = len(self.workflow_state["document_segments"])
        analyzed_segments = len(self.workflow_state["analyzed_segments"])
        active_tasks = len(self.workflow_state["active_tasks"])
        completed_tasks = len(self.workflow_state["completed_tasks"])

        progress = {
            "workflow_id": self.workflow_state["workflow_id"],
            "document_id": self.workflow_state["document_id"],
            "current_phase": self.workflow_state["current_phase"],
            "segment_progress": {
                "total": total_segments,
                "analyzed": analyzed_segments,
                "percentage": (analyzed_segments / total_segments * 100) if total_segments > 0 else 0
            },
            "task_status": {
                "active": active_tasks,
                "completed": completed_tasks,
                "total": active_tasks + completed_tasks
            }
        }

        self._log_action("workflow", "check_progress", self.workflow_state["workflow_id"],
                         f"Progress: {progress['segment_progress']['percentage']:.1f}%")
        return progress

    def transition_phase(self, new_phase: str) -> Dict[str, Any]:
        """
        Transition the workflow to a new phase
        """
        old_phase = self.workflow_state["current_phase"]
        self.workflow_state["current_phase"] = new_phase
        self.workflow_state[f"{new_phase}_start_time"] = datetime.now().isoformat()

        # Store updated workflow state
        self.memory.store_working_data(f"workflow_{self.workflow_state['workflow_id']}", self.workflow_state)

        self._log_action("workflow", "transition_phase", self.workflow_state["workflow_id"],
                         f"Phase transition: {old_phase} -> {new_phase}")

        phase_transition = {
            "workflow_id": self.workflow_state["workflow_id"],
            "previous_phase": old_phase,
            "new_phase": new_phase,
            "transition_time": datetime.now().isoformat()
        }

        # Broadcast phase transition to all agents
        self.broadcast_message(json.dumps({
            "event_type": "phase_transition",
            "data": phase_transition
        }))

        return phase_transition

    def compile_analysis_results(self) -> Dict[str, Any]:
        """
        Compile all analysis results from the workflow
        """
        # Extract results from completed tasks
        results = {}

        for task in self.workflow_state["completed_tasks"]:
            if task.get("compiled", False):  # Skip tasks already compiled
                continue

            task_type = task["task_type"]

            if task_type not in results:
                results[task_type] = []

            results[task_type].append({
                "task_id": task["task_id"],
                "segment_id": task["segment_id"],
                "agent_id": task["agent_id"],
                "result": task["result"]
            })

            # Mark task as compiled
            task["compiled"] = True

        # Update workflow state
        self.workflow_state["analysis_results"].update(results)  # Merge new results
        self.memory.store_working_data(f"workflow_{self.workflow_state['workflow_id']}", self.workflow_state)

        self._log_action("workflow", "compile_results", self.workflow_state["workflow_id"],
                         f"Compiled results for {len(results)} task types")
        return self.workflow_state["analysis_results"]  # Return all results

    # Additional coordinator-specific actions

    def _action_delegate_task(self,
                              agent_id: str,
                              task_type: str,
                              segment_id: str = None,
                              parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Action to delegate a task to another agent
        """
        task_id = self.assign_task(agent_id, task_type, segment_id, parameters)
        return {
            "success": True,
            "task_id": task_id
        }

    def _action_check_status(self) -> Dict[str, Any]:
        """
        Action to check workflow status
        """
        progress = self.check_workflow_progress()
        return {
            "success": True,
            "progress": progress
        }

    def _action_integrate_results(self, task_ids: List[str] = None) -> Dict[str, Any]:
        """
        Action to integrate results from multiple tasks
        """
        if not task_ids:
            # If no task IDs provided, collect all completed tasks
            results = self.compile_analysis_results()
        else:
            # Otherwise, collect specific tasks
            results = {}
            for task_id in task_ids:
                for task in self.workflow_state["completed_tasks"]:
                    if task["task_id"] == task_id:
                        task_type = task["task_type"]

                        if task_type not in results:
                            results[task_type] = []

                        results[task_type].append({
                            "task_id": task["task_id"],
                            "segment_id": task["segment_id"],
                            "agent_id": task["agent_id"],
                            "result": task["result"]
                        })
        return {
            "success": True,
            "results": results
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

        coordinator_actions = {
            "delegate_task": self._action_delegate_task,
            "check_status": self._action_check_status,
            "integrate_results": self._action_integrate_results
        }

        return {**base_actions, **coordinator_actions}
