"""
Module: memory/memory_manager.py

Purpose:
Acts as the central orchestrator for memory-related operations across different memory types:
- Short-Term Memory (STM): recent message history
- Long-Term Memory (LTM): structured facts and searchable documents
- Working Memory (WM): volatile, collaborative memory with expiration

Provides unified interfaces for storing, retrieving, building context, and pruning memory.

Class: MemoryManager

Constructor:
- MemoryManager(short_term_memory, long_term_memory, working_memory, max_tokens)
    - Initializes memory subsystems and token counter

Key Features:

=== Store Methods ===
- store_message(agent_id, content, metadata)
    - Adds a message to STM

- store_document(document)
    - Stores one or more `Document` objects into LTM

- store_fact(fact_id, content)
    - Persists structured facts in LTM

- store_working_data(key, value)
    - Caches transient shared data into WM

=== Retrieve Methods ===
- get_recent_messages(n=10, agent_id=None)
    - Retrieves recent messages (optionally filtered by agent)

- search_documents(query, n_results=5)
    - Returns LTM documents similar to query

- get_fact(fact_id)
    - Retrieves a specific fact from LTM

- get_working_data(key), get_all_working_data()
    - Retrieves working memory entries

=== Context Builder ===
- build_agent_context(agent_id, query=None, max_context_tokens=2000)
    - Creates a composite memory snapshot for an agent
    - Combines STM, relevant LTM docs, and WM (based on token budget)

=== Pruning & Utilities ===
- prune_short_term_memory(keep_last_n=20)
    - Trims STM to recent messages only

- truncate_text(text, max_tokens)
    - Crops a text string to a specific token count

- count_tokens(text)
    - Counts GPT-4 tokens using `tiktoken`

- save_all(session_id), load_all(session_id)
    - Serializes STM & WM to disk (LTM assumed persistent)

- _log_operation(type, target, id, size)
    - Internal tracker for operations via JSONL logs

Integration Notes:
- This is the interface agents use to access all memory types
- Designed for modular replacement or expansion of memory systems
- Key in managing memory-token budget for context-aware AI agents
"""

from typing import Dict, Any, List, Optional, Union
import tiktoken
from datetime import datetime
import json
import os
from .short_term_memory import ShortTermMemory
from .long_term_memory import LongTermMemory
from .working_memory import WorkingMemory
from langchain_core.documents import Document


class MemoryManager:
    """
    Central manager for different memory types with functions to store,
    retrieve, and prune memory contents across different memory systems.
    """
    def __init__(self,
                 short_term_memory: ShortTermMemory = None,
                 long_term_memory: LongTermMemory = None,
                 working_memory: WorkingMemory = None,
                 max_tokens: int = 4000):

        # Initialize memory systems
        self.stm = short_term_memory if short_term_memory else ShortTermMemory()
        self.ltm = long_term_memory if long_term_memory else LongTermMemory()
        self.wm = working_memory if working_memory else WorkingMemory()

        # Token management
        self.max_tokens = max_tokens
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding

        # Create logs directory
        self.logs_path = "data/memory/logs/"
        os.makedirs(self.logs_path, exist_ok=True)

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string"""
        return len(self.encoding.encode(text))

    # ======== STORE METHODS ========

    def store_message(self,
                      agent_id: str,
                      content: str,
                      metadata: Dict[str, Any] = None) -> None:
        """
        Store a message in short-term memory
        """
        message = {
            "agent_id": agent_id,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        self.stm.add_message(message)
        self._log_operation("store", "message", agent_id, len(content))

    def store_document(self, document: Union[Document, List[Document]]) -> None:
        """
        Store document(s) in long-term memory
        """
        if isinstance(document, Document):
            documents = [document]
        else:
            documents = document

        self.ltm.add_documents(documents)
        self._log_operation("store", "document",
                            documents[0].metadata.get("filename", "unknown"),
                            len(documents))

    def store_fact(self, fact_id: str, content: Dict[str, Any]) -> None:
        """
        Store a structured fact in long-term memory
        """
        self.ltm.store_fact(fact_id, content)
        self._log_operation("store", "fact", fact_id,
                            self.count_tokens(json.dumps(content)))

    def store_working_data(self, key: str, value: Any) -> None:
        """
        Store data in working memory for agent collaboration
        """
        self.wm.set(key, value)
        try:
            token_count = self.count_tokens(str(value))
        except ValueError:
            token_count = 0

        self._log_operation("store", "working_data", key, token_count)

    # ======== RETRIEVE METHODS ========

    def get_recent_messages(self, n: int = 10,
                            agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve recent messages, optionally filtered by agent
        """
        messages = self.stm.get_recent_messages(n)

        # Filter by agent if specified
        if agent_id:
            messages = [msg for msg in messages if msg.get("agent_id") == agent_id]

        self._log_operation("retrieve", "messages", agent_id or "all", len(messages))
        return messages

    def search_documents(self, query: str, collection: str = None, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents based on semantic similarity within a specified collection.
        """
        results = self.ltm.search(query, n_results=n_results, collection_name=collection)
        self._log_operation("retrieve", "documents", query[:20], len(results))
        return results

    def get_fact(self, fact_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a stored fact by ID
        """
        fact = self.ltm.retrieve_fact(fact_id)
        self._log_operation("retrieve", "fact", fact_id,
                            1 if fact else 0)
        return fact

    def get_working_data(self, key: str) -> Optional[Any]:
        """
        Retrieve data from working memory
        """
        data = self.wm.get(key)
        self._log_operation("retrieve", "working_data", key,
                            1 if data is not None else 0)
        return data

    def get_working_data_by_partial_key(self, partial_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve working memory entries where the key contains the given partial key.
        """
        all_working_data = self.get_all_working_data()

        for key, value in all_working_data.items():
            if partial_key in key:
                return {"key": key, "value": value}

        return None

    def get_all_working_data(self) -> Dict[str, Any]:
        """
        Get all available working memory data
        """
        all_data = self.wm.get_all()
        self._log_operation("retrieve", "all_working_data", "", len(all_data))
        return all_data

    def retrieve_document_segment(self, segment_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document segment from Working Memory using a specific segment ID.
        """
        all_working_data = self.get_all_working_data()

        # Loop through all working memory to find the segment
        for key, value in all_working_data.items():
            if isinstance(value, dict) and "document_segments" in value:
                # Search the "document_segments" list within this working memory entry
                for segment in value["document_segments"]:
                    if segment.get("segment_id") == segment_id:
                        return segment

        return None  # Return None if the segment is not found

    # ======== CONTEXT BUILDING METHODS ========

    def build_agent_context(self,
                            agent_id: str,
                            query: str = None,
                            max_context_tokens: int = 2000) -> str:
        """
        Build a context for an agent combining relevant information
        from different memory sources
        """
        context_parts = []
        token_budget = max_context_tokens

        # 1. Add recent messages (30% of budget)
        message_budget = int(token_budget * 0.3)
        recent_messages = self.get_recent_messages(n=10)
        message_context = ""

        for msg in recent_messages:
            message_text = f"{msg['agent_id']}: {msg['content']}\n"
            msg_tokens = self.count_tokens(message_text)

            if msg_tokens < message_budget:
                message_context += message_text
                message_budget -= msg_tokens
            else:
                break

        if message_context:
            context_parts.append("## Recent Messages\n" + message_context)
            token_budget -= self.count_tokens(message_context)

        # 2. Add relevant documents if query provided (50% of original budget)
        if query:
            doc_budget = int(max_context_tokens * 0.5)
            relevant_docs = self.search_documents(query, n_results=3)
            doc_context = ""

            for doc in relevant_docs:
                # Truncate doc content if too long
                content = doc["content"]
                if self.count_tokens(content) > 500:
                    content = self.truncate_text(content, 500)

                doc_text = (f"Document: {doc['metadata'].get('filename', 'Unknown')}"
                            f" (Page {doc['metadata'].get('page', 'Unknown')})\n"
                            f"{content}\n\n")

                doc_tokens = self.count_tokens(doc_text)

                if doc_tokens < doc_budget:
                    doc_context += doc_text
                    doc_budget -= doc_tokens
                else:
                    break

            if doc_context:
                context_parts.append("## Relevant Documents\n" + doc_context)
                token_budget -= self.count_tokens(doc_context)

        # 3. Add working memory data (remaining budget)
        working_data = self.get_all_working_data()
        if working_data:
            working_context = "## Working Data\n"

            for key, value in working_data.items():
                try:
                    if isinstance(value, dict) or isinstance(value, list):
                        value_str = json.dumps(value, indent=2)[:200]  # Limit size
                    else:
                        value_str = str(value)[:200]  # Limit size

                    entry = f"{key}: {value_str}\n"
                    entry_tokens = self.count_tokens(entry)

                    if entry_tokens < token_budget:
                        working_context += entry
                        token_budget -= entry_tokens
                    else:
                        working_context += f"{key}: [Content too large]\n"
                except (TypeError, ValueError):
                    working_context += f"{key}: [Unserializable content]\n"

            context_parts.append(working_context)

        # Combine all context parts
        full_context = "\n\n".join(context_parts)
        self._log_operation("build", "context", agent_id,
                            self.count_tokens(full_context))

        return full_context

    # ======== PRUNING METHODS ========

    def prune_short_term_memory(self, keep_last_n: int = 20) -> None:
        """
        Prune short-term memory to keep only the most recent n messages
        """
        current_messages = self.stm.get_recent_messages()
        if len(current_messages) <= keep_last_n:
            return

        # Keep only the most recent messages
        pruned_messages = current_messages[-keep_last_n:]

        # Reset the memory with pruned messages
        self.stm.clear()
        for message in pruned_messages:
            self.stm.add_message(message)

        self._log_operation("prune", "short_term", "",
                            len(current_messages) - len(pruned_messages))

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit
        """
        if self.count_tokens(text) <= max_tokens:
            return text

        # Simple truncation strategy - could be improved
        tokens = self.encoding.encode(text)
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens) + "..."

    # ======== UTILITY METHODS ========

    def _log_operation(self,
                       operation_type: str,
                       target_type: str,
                       target_id: str,
                       size: int) -> None:
        """
        Log memory operations for debugging and monitoring
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation_type,
            "target_type": target_type,
            "target_id": target_id,
            "size": size
        }

        log_file = f"{self.logs_path}memory_operations.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")

    def save_all(self, session_id: str) -> None:
        """
        Save all memory systems to disk
        """
        self.stm.save(session_id)
        self.wm.save(session_id)
        # Long-term memory is persistent by default

        self._log_operation("save", "all", session_id, 0)

    def load_all(self, session_id: str) -> bool:
        """
        Load all memory systems from disk
        """
        stm_loaded = self.stm.load(session_id)
        wm_loaded = self.wm.load(session_id)
        # Long-term memory is persistent by default

        self._log_operation("load", "all", session_id,
                            1 if (stm_loaded or wm_loaded) else 0)

        return stm_loaded or wm_loaded
