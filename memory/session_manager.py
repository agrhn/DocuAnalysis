"""
Module: memory/session_manager.py

Purpose:
Provides a high-level interface to manage user sessions, delegating memory
state persistence and restoration to the underlying `MemoryPersistence` and
`MemoryManager` classes.

Class: SessionManager

Constructor:
- SessionManager(memory_manager=None, base_path="data/sessions/")
    - Initializes with an optional `MemoryManager`
    - Uses `MemoryPersistence` for file-based session I/O
    - Tracks the current session ID and its metadata

Key Features:

=== Session Lifecycle ===
- start_new_session(session_name=None, metadata=None) -> str
    - Ends and saves current session if one exists
    - Clears short-term and working memory (long-term is persistent)
    - Starts a new session and stores a `session_start` system message
    - Returns the new session ID

- load_session(session_id) -> bool
    - Saves the current session (if active)
    - Loads specified session into memory
    - Stores a `session_resume` system message
    - Updates `current_session_id` and metadata

- end_current_session() -> bool
    - Saves current session and clears state
    - Logs a `session_end` system message

=== Persistence ===
- save_current_session() -> bool
    - Triggers memory persistence for current session

=== Session Info ===
- list_available_sessions() -> List[Dict]
    - Returns all sessions with metadata from index

- get_session_info(session_id=None) -> Optional[Dict]
    - Returns metadata for a session (defaults to current)

- get_current_session_id() -> Optional[str]
    - Returns the active session ID

- delete_session(session_id) -> bool
    - Deletes the given session
    - Cannot delete currently active session

Session Flow Summary:
User action ─┐
             ↓ [Start new session] ────────> Clears STM + WM, Creates new session dir, Logs session_start
               [Load session] ─────────────> Saves current (if any), Loads saved STM + WM, Logs session_resume
               [End session] ──────────────> Logs session_end, Saves session, Clears session tracking
               [Delete session] ───────────> Permanently removes session folder (if not active)

Integration Notes:
- Seamlessly coordinates with `MemoryPersistence` and `MemoryManager`
- Ensures consistent transitions between sessions
- Metadata tracking allows for enhanced user personalization or auditing
"""

from typing import Dict, Any, Optional, List
from .persistence import MemoryPersistence
from .memory_manager import MemoryManager


class SessionManager:
    """
    Manages user sessions and provides an interface for memory persistence
    across different sessions.
    """
    def __init__(self,
                 memory_manager: MemoryManager = None,
                 base_path: str = "data/sessions/"):

        self.memory_manager = memory_manager or MemoryManager()
        self.persistence = MemoryPersistence(base_path)
        self.current_session_id = None
        self.session_metadata = {}

    def start_new_session(self,
                          session_name: str = None,
                          metadata: Dict[str, Any] = None) -> str:
        """
        Start a new session, create storage, and return session ID
        """
        # Save current session if one exists
        if self.current_session_id:
            self.save_current_session()

        # Reset memory states
        self.memory_manager.stm.clear()
        self.memory_manager.wm.clear()
        # Note: We don't clear long-term memory as it's persistent

        # Create new session
        session_id = self.persistence.create_session(session_name, metadata)
        self.current_session_id = session_id
        self.session_metadata = metadata or {}

        # Store session start in short-term memory
        self.memory_manager.store_message(
            "system",
            f"Session started: {session_name or session_id}",
            {"event": "session_start", "session_id": session_id}
        )

        return session_id

    def load_session(self, session_id: str) -> bool:
        """
        Load an existing session
        """
        # Save current session if one exists
        if self.current_session_id:
            self.save_current_session()

        # Load requested session
        success = self.persistence.load_session_state(session_id, self.memory_manager)

        if success:
            self.current_session_id = session_id
            session_info = self.persistence.get_session_info(session_id)
            self.session_metadata = session_info.get("metadata", {}) if session_info else {}

            # Store session resume in short-term memory
            self.memory_manager.store_message(
                "system",
                f"Session resumed: {session_id}",
                {"event": "session_resume", "session_id": session_id}
            )

        return success

    def save_current_session(self) -> bool:
        """
        Save the current session state
        """
        if not self.current_session_id:
            return False

        return self.persistence.save_session_state(
            self.current_session_id,
            self.memory_manager
        )

    def end_current_session(self) -> bool:
        """
        End current session, saving state
        """
        if not self.current_session_id:
            return False

        # Store session end in short-term memory
        self.memory_manager.store_message(
            "system",
            f"Session ended: {self.current_session_id}",
            {"event": "session_end", "session_id": self.current_session_id}
        )

        # Save session state
        success = self.save_current_session()

        # Clear current session
        if success:
            self.current_session_id = None
            self.session_metadata = {}

        return success

    def list_available_sessions(self) -> List[Dict[str, Any]]:
        """
        List all available sessions
        """
        return self.persistence.list_sessions()

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its data
        """
        # Can't delete current session
        if session_id == self.current_session_id:
            return False

        return self.persistence.delete_session(session_id)

    def get_current_session_id(self) -> Optional[str]:
        """
        Get the current session ID
        """
        return self.current_session_id

    def get_session_info(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get info about a session (defaults to current session)
        """
        target_id = session_id or self.current_session_id
        if not target_id:
            return None

        return self.persistence.get_session_info(target_id)
