"""
Module: memory/persistence.py

Purpose:
Provides file-based session management and persistence for memory systems,
allowing memory states to be saved, restored, listed, or deleted.

Class: MemoryPersistence

Constructor:
- MemoryPersistence(base_path="data/sessions/")
    - Sets up the storage directory and initializes the session index

Key Features:

=== Session Management ===
- create_session(session_name=None, metadata=None) -> str
    - Creates a new session with a unique ID, directories, and metadata
    - Returns the session ID

- delete_session(session_id) -> bool
    - Removes all session data and updates the index

- list_sessions() -> List[Dict]
    - Returns all sessions with metadata

- get_session_info(session_id) -> Optional[Dict]
    - Fetches metadata for a single session

=== Persistence Operations ===
- save_session_state(session_id, memory_manager) -> bool
    - Saves:
        - Short-Term Memory messages
        - Working Memory data and expiry times
    - Assumes Long-Term Memory (ChromaDB) is already persistent

- load_session_state(session_id, memory_manager) -> bool
    - Restores memory from disk into the given `memory_manager`
    - Parses ISO date strings into `datetime` for working memory
    - Triggers WM cleanup after load

=== Internal Helpers ===
- _initialize_sessions_index()
    - Creates an empty session index if not present

- _load_sessions_index() / _save_sessions_index()
    - Loads/saves the `sessions_index.json` tracking all sessions

- _update_session_access_time(session_id)
    - Updates the `last_accessed` timestamp for session tracking

Storage Structure:
data/
    └── sessions/
        ├── sessions_index.json
        └── session_YYYYMMDD_HHMMSS/
            ├── short_term/messages.json
            ├── working/data.json
            └── long_term/ (placeholder - managed externally)

Integration Notes:
- Works seamlessly with `MemoryManager.save_all()` and `load_all()`
- Modular session design makes it easy to switch between different sessions
- Robust to missing or corrupted index files (auto-recovers)
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List


class MemoryPersistence:
    """
    Handles persistence operations for all memory types across sessions.
    Provides file-based storage with session management.
    """
    def __init__(self, base_path: str = "data/sessions/"):
        self.base_path = base_path
        self.sessions_index_path = f"{base_path}sessions_index.json"

        # Create necessary directories
        os.makedirs(base_path, exist_ok=True)

        # Initialize sessions index if it doesn't exist
        if not os.path.exists(self.sessions_index_path):
            self._initialize_sessions_index()

    def _initialize_sessions_index(self) -> None:
        """Create an empty sessions index file"""
        with open(self.sessions_index_path, 'w') as f:
            json.dump({
                "sessions": [],
                "last_updated": datetime.now().isoformat()
            }, f)

    def create_session(self,
                       session_name: str = None,
                       metadata: Dict[str, Any] = None) -> str:
        """
        Create a new session and return the session ID
        """
        # Generate session ID based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"session_{timestamp}"

        # Create session directory
        session_dir = f"{self.base_path}{session_id}/"
        os.makedirs(session_dir, exist_ok=True)

        # Create subdirectories for different memory types
        os.makedirs(f"{session_dir}short_term/", exist_ok=True)
        os.makedirs(f"{session_dir}working/", exist_ok=True)
        os.makedirs(f"{session_dir}long_term/", exist_ok=True)

        # Update sessions index
        sessions = self._load_sessions_index()
        sessions["sessions"].append({
            "id": session_id,
            "name": session_name or f"Session {timestamp}",
            "created": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "metadata": metadata or {}
        })
        sessions["last_updated"] = datetime.now().isoformat()

        # Save updated index
        self._save_sessions_index(sessions)

        return session_id

    def save_session_state(self,
                           session_id: str,
                           memory_manager) -> bool:
        """
        Save the current state of all memory systems for a session
        """
        session_dir = f"{self.base_path}{session_id}/"

        # Ensure session exists
        if not os.path.exists(session_dir):
            return False

        try:
            # Save short-term memory
            short_term_path = f"{session_dir}short_term/messages.json"
            with open(short_term_path, 'w') as f:
                json.dump(memory_manager.stm.messages, f)

            # Save working memory
            working_path = f"{session_dir}working/data.json"
            # Convert datetime objects to strings
            serializable_expiry = {
                k: v.isoformat()
                for k, v in memory_manager.wm.expiry_times.items()
            }

            with open(working_path, 'w') as f:
                json.dump({
                    "data": memory_manager.wm.data,
                    "expiry_times": serializable_expiry
                }, f)

            # For long-term memory, we don't need to do anything special
            # as ChromaDB is already persistent

            # Update session access time
            self._update_session_access_time(session_id)

            return True
        except Exception as e:
            print(f"Error saving session state: {str(e)}")
            return False

    def load_session_state(self,
                           session_id: str,
                           memory_manager) -> bool:
        """
        Load a session's memory state into the memory manager
        """
        session_dir = f"{self.base_path}{session_id}/"

        # Ensure session exists
        if not os.path.exists(session_dir):
            return False

        try:
            # Load short-term memory
            short_term_path = f"{session_dir}short_term/messages.json"
            if os.path.exists(short_term_path):
                with open(short_term_path, 'r') as f:
                    memory_manager.stm.messages = json.load(f)

            # Load working memory
            working_path = f"{session_dir}working/data.json"
            if os.path.exists(working_path):
                with open(working_path, 'r') as f:
                    saved_data = json.load(f)

                memory_manager.wm.data = saved_data["data"]
                # Convert string dates back to datetime
                memory_manager.wm.expiry_times = {
                    k: datetime.fromisoformat(v)
                    for k, v in saved_data["expiry_times"].items()
                }

                # Clean expired items
                memory_manager.wm._clean_expired()

            # For long-term memory, nothing special needed
            # as ChromaDB is already persistent

            # Update session access time
            self._update_session_access_time(session_id)

            return True
        except Exception as e:
            print(f"Error loading session state: {str(e)}")
            return False

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its associated data
        """
        session_dir = f"{self.base_path}{session_id}/"

        # Ensure session exists
        if not os.path.exists(session_dir):
            return False

        try:
            # Delete session directory
            shutil.rmtree(session_dir)

            # Update sessions index
            sessions = self._load_sessions_index()
            sessions["sessions"] = [
                s for s in sessions["sessions"] if s["id"] != session_id
            ]
            sessions["last_updated"] = datetime.now().isoformat()

            # Save updated index
            self._save_sessions_index(sessions)

            return True
        except Exception as e:
            print(f"Error deleting session: {str(e)}")
            return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        Get a list of all available sessions
        """
        sessions = self._load_sessions_index()
        return sessions["sessions"]

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific session
        """
        sessions = self._load_sessions_index()
        for session in sessions["sessions"]:
            if session["id"] == session_id:
                return session
        return None

    def _load_sessions_index(self) -> Dict[str, Any]:
        """Load the sessions index file"""
        try:
            with open(self.sessions_index_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading session index: {str(e)}")
            self._initialize_sessions_index()
            with open(self.sessions_index_path, 'r') as f:
                return json.load(f)

    def _save_sessions_index(self, sessions: Dict[str, Any]) -> None:
        """Save the sessions index file"""
        with open(self.sessions_index_path, 'w') as f:
            json.dump(sessions, f)

    def _update_session_access_time(self, session_id: str) -> None:
        """Update the last accessed time for a session"""
        sessions = self._load_sessions_index()
        for session in sessions["sessions"]:
            if session["id"] == session_id:
                session["last_accessed"] = datetime.now().isoformat()
                break

        sessions["last_updated"] = datetime.now().isoformat()
        self._save_sessions_index(sessions)
