"""
Module: memory/short_term_memory.py

Purpose:
Implements the ShortTermMemory class to handle recent agent interactions and maintain contextual awareness
during a session. Acts as a rolling log of recent messages for coordination and conversation tracking.

Key Responsibilities:
- Temporarily stores recent messages exchanged between agents or with the user
- Maintains a rolling window of message history (default: 50 messages)
- Supports save/load to disk for session persistence

Class: ShortTermMemory

Constructor:
- ShortTermMemory(max_messages: int = 50)
    - Initializes memory with a maximum number of retained messages
    - Sets up disk storage path

Key Methods:
- add_message(message: Dict[str, Any]) -> None
    - Adds a message to memory
    - Automatically timestamps messages if not provided
    - Maintains size limit by removing oldest messages

- get_recent_messages(n: int = None) -> List[Dict[str, Any]]
    - Returns the last `n` messages or all messages if `n` is None

- clear() -> None
    - Empties the short-term memory

- save(session_id: str) -> None
    - Saves current message history to a JSON file associated with a session

- load(session_id: str) -> bool
    - Loads a previously saved message history for a given session
    - Returns `True` if successful, `False` otherwise

Integration Notes:
- Used by the Coordinator Agent to maintain context of recent interactions
- Can be queried by any agent to review conversation flow or shared information
- Complements WorkingMemory by tracking chronological conversational state
"""

from typing import List, Dict, Any
import json
import os
from datetime import datetime


class ShortTermMemory:
    """
    Stores recent messages and interactions between agents.
    Implements a simple rolling window of recent messages.
    """
    def __init__(self, max_messages: int = 50):
        self.max_messages = max_messages
        self.messages = []
        self.storage_path = "data/memory/short_term/"

        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)

    def add_message(self, message: Dict[str, Any]) -> None:
        """Add a message to the short-term memory"""
        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.now().isoformat()

        # Add the message
        self.messages.append(message)

        # Enforce the message limit (rolling window)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_recent_messages(self, n: int = None) -> List[Dict[str, Any]]:
        """Retrieve the n most recent messages"""
        if n is None or n > len(self.messages):
            return self.messages
        return self.messages[-n:]

    def clear(self) -> None:
        """Clear all messages from memory"""
        self.messages = []

    def save(self, session_id: str) -> None:
        """Save the current state to disk"""
        filename = f"{self.storage_path}{session_id}_short_term.json"
        with open(filename, 'w') as f:
            json.dump(self.messages, f)

    def load(self, session_id: str) -> bool:
        """Load a previous state from disk"""
        filename = f"{self.storage_path}{session_id}_short_term.json"
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    self.messages = json.load(f)
                return True
            return False
        except Exception as e:
            print(f"Error loading short-term memory: {str(e)}")
            return False
