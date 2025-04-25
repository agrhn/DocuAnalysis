"""
Module: working_memory.py

Purpose:
Implements the WorkingMemory class for short-lived, collaborative data storage among agents.
Acts as a shared temporary workspace to coordinate intermediate results, tasks, or focus areas.

Key Responsibilities:
- Provides volatile memory with optional expiration for collaborative agent tasks
- Supports session persistence to disk
- Automatically removes expired items to keep working set relevant

Class: WorkingMemory

Constructor:
- WorkingMemory(expiry_minutes: int = 30)
    - Initializes temporary memory store with expiration timing
    - Sets up storage path for persistence

Key Methods:
- set(key: str, value: Any) -> None
    - Stores a value with a specific key and sets its expiration time

- get(key: str) -> Optional[Any]
    - Retrieves a value if it exists and hasn't expired

- get_all() -> Dict[str, Any]
    - Returns all non-expired key-value pairs currently in memory

- clear() -> None
    - Clears all working memory data and expiry tracking

- save(session_id: str) -> None
    - Saves current memory state and expiry info to disk for a given session

- load(session_id: str) -> bool
    - Loads saved memory and expiry info for a session
    - Returns `True` if successful, `False` otherwise

Integration Notes:
- Acts as a live scratchpad for agents to pass intermediate results
- Useful for coordinated agent workflows (e.g., Extraction → Analysis → Summary)
- Coordinator Agent may use this for task handoff or attention control
"""

from typing import Dict, Any, Optional
import json
import os
from datetime import datetime, timedelta


class WorkingMemory:
    """
    Working memory for temporary data sharing between agents.
    Functions as a workspace for collaborative tasks.
    """
    def __init__(self, expiry_minutes: int = 30):
        self.data = {}
        self.expiry_minutes = expiry_minutes
        self.expiry_times = {}
        self.storage_path = "data/memory/working/"

        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)

    def set(self, key: str, value: Any) -> None:
        """
        Store data in working memory with expiration
        """
        self.data[key] = value
        self.expiry_times[key] = datetime.now() + timedelta(minutes=self.expiry_minutes)

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve data from working memory if not expired
        """
        # Check if key exists
        if key not in self.data:
            return None

        # Check if expired
        if datetime.now() > self.expiry_times[key]:
            del self.data[key]
            del self.expiry_times[key]
            return None

        return self.data[key]

    def get_all(self) -> Dict[str, Any]:
        """
        Get all non-expired data
        """
        # Clear expired data
        self._clean_expired()

        return self.data

    def _clean_expired(self) -> None:
        """
        Remove all expired items
        """
        now = datetime.now()
        expired_keys = [k for k, exp_time in self.expiry_times.items() if now > exp_time]

        for key in expired_keys:
            if key in self.data:
                del self.data[key]
            if key in self.expiry_times:
                del self.expiry_times[key]

    def clear(self) -> None:
        """
        Clear all data in working memory
        """
        self.data = {}
        self.expiry_times = {}

    def save(self, session_id: str) -> None:
        """
        Save the current state to disk
        """
        filename = f"{self.storage_path}{session_id}_working.json"

        # Convert datetime objects to strings
        serializable_expiry = {k: v.isoformat() for k, v in self.expiry_times.items()}

        with open(filename, 'w') as f:
            json.dump({
                "data": self.data,
                "expiry_times": serializable_expiry
            }, f)

    def load(self, session_id: str) -> bool:
        """
        Load a previous state from disk
        """
        filename = f"{self.storage_path}{session_id}_working.json"
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    saved_data = json.load(f)

                self.data = saved_data["data"]
                # Convert string dates back to datetime
                self.expiry_times = {
                    k: datetime.fromisoformat(v)
                    for k, v in saved_data["expiry_times"].items()
                }

                # Clean expired items
                self._clean_expired()
                return True
            return False
        except Exception as e:
            print(f"Error loading working memory: {str(e)}")
            return False
