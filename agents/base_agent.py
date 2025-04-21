from typing import Dict, Any, List, Union, Callable
import json
import uuid
from datetime import datetime
import os
import time
from openai import OpenAI
from memory.memory_manager import MemoryManager


class BaseAgent:
    """
    Base agent class providing core functionality for all specialized agents.
    Implements memory access, communication, and thinking processes.
    """
    def __init__(
        self,
        agent_id: str,
        name: str,
        role: str,
        memory_manager: MemoryManager,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.memory = memory_manager
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize OpenAI client
        self.client = OpenAI()

        # Define system prompt template
        self.system_prompt_template = """
        You are {name}, an AI assistant with the role of {role}.

        Your agent ID is: {agent_id}

        In this role, you should:
        - Focus on your specific responsibilities
        - Be concise and clear in your communications
        - Collaborate effectively with other agents
        - Always provide reasoning for your conclusions

        {additional_instructions}
        """

        # Additional instructions specific to this agent
        self.additional_instructions = ""

        # Create logs directory
        self.logs_path = "data/logs/agents/"
        os.makedirs(self.logs_path, exist_ok=True)

    def set_additional_instructions(self, instructions: str) -> None:
        """
        Set additional role-specific instructions for the agent
        """
        self.additional_instructions = instructions

    def get_system_prompt(self) -> str:
        """
        Generate the system prompt for this agent
        """
        return self.system_prompt_template.format(
            name=self.name,
            role=self.role,
            agent_id=self.agent_id,
            additional_instructions=self.additional_instructions
        )

    # ======== COMMUNICATION METHODS ========

    def send_message(self, recipient_id: str, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        Send a message to another agent via the memory system
        """
        message_id = str(uuid.uuid4())

        message = {
            "message_id": message_id,
            "sender_id": self.agent_id,
            "recipient_id": recipient_id,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        # Store in memory
        self.memory.store_message(self.agent_id, content, {
            "message_id": message_id,
            "recipient_id": recipient_id,
            "is_communication": True,
            **(metadata or {})
        })

        self._log_action("communication", "send_message", recipient_id, content[:50])

        return message_id

    def get_messages(self, n: int = 5, from_agent: str = None) -> List[Dict[str, Any]]:
        """
        Get recent messages, optionally filtered by sender
        """
        messages = self.memory.get_recent_messages(n)

        # Filter messages if a specific sender is specified
        if from_agent:
            messages = [msg for msg in messages if msg.get("agent_id") == from_agent]

        return messages

    def broadcast_message(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        Broadcast a message to all agents
        """
        message_id = str(uuid.uuid4())

        message = {
            "message_id": message_id,
            "sender_id": self.agent_id,
            "recipient_id": "all",
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        # Store in memory
        self.memory.store_message(self.agent_id, content, {
            "message_id": message_id,
            "recipient_id": "all",
            "is_broadcast": True,
            **(metadata or {})
        })

        self._log_action("communication", "broadcast", "all", content[:50])

        return message_id

    # ======== THINKING PROCESS METHODS ========

    def perceive(self,
                 input_data: Union[str, Dict[str, Any], List[Dict[str, Any]]],
                 context_query: str = None) -> Dict[str, Any]:
        """
        First step of the thinking process: Perception
        Process input data and relevant context from memory
        """
        # Process input data
        if isinstance(input_data, str):
            processed_input = {"text": input_data}
        else:
            processed_input = input_data

        # Build context from memory
        context = self.memory.build_agent_context(
            self.agent_id,
            query=context_query or (processed_input.get("text", "") if isinstance(processed_input, dict) else "")
        )

        perception_result = {
            "input": processed_input,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }

        self._log_action("thinking", "perceive", "",
                         f"Input type: {type(input_data).__name__}, Context length: {len(context)}")

        return perception_result

    def reason(self,
               perception: Dict[str, Any],
               reasoning_prompt: str = None) -> Dict[str, Any]:
        """
        Second step of the thinking process: Reasoning
        Analyze the perception and generate thoughts about it
        """
        # Prepare the messages for the LLM
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
        ]

        # Add context if available
        if perception.get("context"):
            messages.append({
                "role": "system",
                "content": f"Context:\n{perception['context']}"
            })

        # Add input as user message
        if isinstance(perception.get("input"), dict) and perception["input"].get("text"):
            input_text = perception["input"]["text"]
        elif isinstance(perception.get("input"), str):
            input_text = perception["input"]
        else:
            input_text = json.dumps(perception.get("input", {}))

        # Add reasoning prompt if provided
        if reasoning_prompt:
            prompt_text = f"{reasoning_prompt}\n\nInput: {input_text}"
        else:
            prompt_text = f"Please analyze this input: {input_text}"

        messages.append({
            "role": "user",
            "content": prompt_text
        })

        # Call the LLM
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            reasoning_output = response.choices[0].message.content
            tokens_used = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        except Exception as e:
            reasoning_output = f"Error during reasoning: {str(e)}"
            tokens_used = {"error": str(e)}

        elapsed_time = time.time() - start_time

        reasoning_result = {
            "output": reasoning_output,
            "tokens": tokens_used,
            "elapsed_time": elapsed_time,
            "timestamp": datetime.now().isoformat()
        }

        self._log_action("thinking", "reason", "",
                         f"Output length: {len(reasoning_output)}, Time: {elapsed_time:.2f}s")

        return reasoning_result

    def decide_action(self,
                      reasoning: Dict[str, Any],
                      action_options: List[str] = None) -> Dict[str, Any]:
        """
        Third step of the thinking process: Action Decision
        Determine what action to take based on reasoning
        """
        # Prepare the messages for the LLM
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "system", "content": """
             Based on your reasoning, decide what action to take.
             Respond in the following JSON format:
             {
                "action": "action_name",
                "parameters": {
                    "param1": "value1",
                    "param2": "value2"
                },
                "reasoning": "Brief explanation for this action choice"
             }
             """}
        ]

        # Add reasoning as context
        messages.append({
            "role": "system",
            "content": f"Your reasoning:\n{reasoning.get('output', '')}"
        })

        # Add action options if provided
        if action_options:
            actions_text = "Available actions:\n" + "\n".join(f"- {action}" for action in action_options)
            messages.append({
                "role": "system",
                "content": actions_text
            })

        messages.append({
            "role": "user",
            "content": "What action should you take now?"
        })

        # Call the LLM
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            action_text = response.choices[0].message.content

            # Try to parse as JSON
            try:
                action_data = json.loads(action_text)
            except json.JSONDecodeError:
                # If not valid JSON, create a simple action
                action_data = {
                    "action": "communicate",
                    "parameters": {
                        "message": action_text
                    },
                    "reasoning": "Direct response (not in JSON format)"
                }

            tokens_used = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        except Exception as e:
            action_data = {
                "action": "error",
                "parameters": {
                    "error": str(e)
                },
                "reasoning": f"Error occurred during action decision: {str(e)}"
            }
            tokens_used = {"error": str(e)}

        elapsed_time = time.time() - start_time

        action_result = {
            "action": action_data,
            "tokens": tokens_used,
            "elapsed_time": elapsed_time,
            "timestamp": datetime.now().isoformat()
        }

        self._log_action("thinking", "decide_action", "",
                         f"Action: {action_data.get('action')}")

        return action_result

    def execute_action(self,
                       action_result: Dict[str, Any],
                       available_actions: Dict[str, Callable] = None) -> Dict[str, Any]:
        """
        Fourth step of the thinking process: Action Execution
        Execute the decided action
        """
        action_data = action_result.get("action", {})
        action_name = action_data.get("action", "no_action")
        parameters = action_data.get("parameters", {})

        # Use provided actions or default to basic actions
        actions = available_actions or {
            "communicate": self._action_communicate,
            "store_fact": self._action_store_fact,
            "retrieve_info": self._action_retrieve_info,
            "no_action": self._action_no_action,
            "error": self._action_error
        }

        # Execute the action
        start_time = time.time()
        try:
            if action_name in actions:
                result = actions[action_name](**parameters)
            else:
                result = {
                    "success": False,
                    "result": f"Unknown action: {action_name}"
                }
        except Exception as e:
            result = {
                "success": False,
                "result": f"Error executing action {action_name}: {str(e)}"
            }

        elapsed_time = time.time() - start_time

        execution_result = {
            "action_name": action_name,
            "parameters": parameters,
            "result": result,
            "elapsed_time": elapsed_time,
            "timestamp": datetime.now().isoformat()
        }

        self._log_action("thinking", "execute_action", "",
                         f"Action: {action_name}, Success: {result.get('success', False)}")

        return execution_result

    def think(self,
              input_data: Union[str, Dict[str, Any], List[Dict[str, Any]]],
              context_query: str = None,
              reasoning_prompt: str = None,
              action_options: List[str] = None,
              available_actions: Dict[str, Callable] = None) -> Dict[str, Any]:
        """
        Execute the complete thinking process
        """
        thinking_start = time.time()

        # Step 1: Perception
        perception = self.perceive(input_data, context_query)

        # Step 2: Reasoning
        reasoning = self.reason(perception, reasoning_prompt)

        # Step 3: Action Decision
        action_decision = self.decide_action(reasoning, action_options)

        # Step 4: Action Execution
        execution = self.execute_action(action_decision, available_actions)

        thinking_time = time.time() - thinking_start

        # Compile results
        thinking_result = {
            "perception": perception,
            "reasoning": reasoning,
            "action_decision": action_decision,
            "execution": execution,
            "total_time": thinking_time,
            "timestamp": datetime.now().isoformat()
        }

        # Store thinking process in working memory
        thinking_key = f"thinking_{self.agent_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.memory.store_working_data(thinking_key, {
            "agent_id": self.agent_id,
            "reasoning_output": reasoning.get("output", ""),
            "action": execution.get("action_name", ""),
            "timestamp": datetime.now().isoformat()
        })

        self._log_action("thinking", "complete", "",
                         f"Total time: {thinking_time:.2f}s")

        return thinking_result

    # ======== DEFAULT ACTIONS ========

    def _action_communicate(self,
                            message: str,
                            recipient_id: str = None,
                            broadcast: bool = False) -> Dict[str, Any]:
        """
        Default action to communicate with other agents
        """
        if broadcast:
            message_id = self.broadcast_message(message)
            return {
                "success": True,
                "message_id": message_id,
                "type": "broadcast"
            }
        elif recipient_id:
            message_id = self.send_message(recipient_id, message)
            return {
                "success": True,
                "message_id": message_id,
                "type": "direct",
                "recipient": recipient_id
            }
        else:
            # If no recipient specified, log internal thought
            self._log_action("internal", "thought", "", message[:100])
            return {
                "success": True,
                "type": "internal_thought"
            }

    def _action_store_fact(self,
                           fact_id: str = None,
                           content: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Default action to store a fact in long-term memory
        """
        if not fact_id:
            fact_id = f"fact_{self.agent_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        if not content:
            return {
                "success": False,
                "error": "No content provided"
            }

        self.memory.store_fact(fact_id, content)

        return {
            "success": True,
            "fact_id": fact_id
        }

    def _action_retrieve_info(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Default action to retrieve information from long-term memory
        """
        if not query:
            return {
                "success": False,
                "error": "No query provided"
            }

        results = self.memory.search_documents(query, n_results)

        return {
            "success": True,
            "results": results,
            "count": len(results)
        }

    def _action_no_action(self) -> Dict[str, Any]:
        """
        Default action when no action is needed
        """
        return {
            "success": True,
            "result": "No action taken"
        }

    def _action_error(self, error: str = "Unknown error") -> Dict[str, Any]:
        """
        Default action for error handling
        """
        return {
            "success": False,
            "error": error
        }

    # ======== UTILITY METHODS ========

    def _log_action(self,
                    category: str,
                    action: str,
                    target: str,
                    details: str) -> None:
        """
        Log agent actions for debugging and monitoring
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            "category": category,
            "action": action,
            "target": target,
            "details": details
        }

        log_file = f"{self.logs_path}{self.agent_id}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
