import os
from memory.short_term_memory import ShortTermMemory
from memory.long_term_memory import LongTermMemory
from memory.working_memory import WorkingMemory
from memory.memory_manager import MemoryManager
from memory.session_manager import SessionManager
from langchain_core.documents import Document


def test_memory_systems():
    print("Testing Memory Systems...")

    # Create test directories if they don't exist
    os.makedirs("data/memory/short_term", exist_ok=True)
    os.makedirs("data/memory/long_term", exist_ok=True)
    os.makedirs("data/memory/working", exist_ok=True)

    # 1. Test Short-Term Memory
    print("\n=== Testing Short-Term Memory ===")
    stm = ShortTermMemory(max_messages=5)

    # Add messages
    stm.add_message({"agent_id": "test_agent", "content": "Test message 1"})
    stm.add_message({"agent_id": "test_agent", "content": "Test message 2"})

    # Retrieve messages
    messages = stm.get_recent_messages()
    print(f"Retrieved {len(messages)} messages")
    print(f"Last message: {messages[-1]['content']}")

    # Test persistence
    stm.save("test_session")
    stm.clear()
    assert len(stm.get_recent_messages()) == 0, "Clear failed"

    stm.load("test_session")
    messages = stm.get_recent_messages()
    print(f"After load: Retrieved {len(messages)} messages")
    print(f"After load: Last message: {messages[-1]['content']}")

    # 2. Test Working Memory
    print("\n=== Testing Working Memory ===")
    wm = WorkingMemory(expiry_minutes=5)

    # Store data
    wm.set("test_key", "test_value")
    wm.set("test_dict", {"name": "test", "value": 123})

    # Retrieve data
    value = wm.get("test_key")
    print(f"Retrieved value: {value}")

    # Test persistence
    wm.save("test_session")
    wm.clear()
    assert wm.get("test_key") is None, "Clear failed"

    wm.load("test_session")
    value = wm.get("test_key")
    print(f"After load: Retrieved value: {value}")

    # 3. Test Long-Term Memory
    print("\n=== Testing Long-Term Memory ===")
    ltm = LongTermMemory(collection_name="test_collection")

    # Add a document
    test_doc = Document(
        page_content="This is a test document with important information about memory systems",
        metadata={"source": "test", "filename": "test.pdf", "page": 1}
    )
    ltm.add_documents([test_doc])

    # Search for documents
    results = ltm.search("important information")
    print(f"Search returned {len(results)} results")
    if results:
        print(f"Top result content: {results[0]['content'][:50]}...")

    # Store and retrieve a fact
    fact_id = "test_fact_1"
    ltm.store_fact(fact_id, {"type": "concept", "name": "Memory Systems", "details": "Test details"})
    fact = ltm.retrieve_fact(fact_id)
    print(f"Retrieved fact: {fact['name'] if fact else 'None'}")

    # 4. Test Memory Manager
    print("\n=== Testing Memory Manager ===")
    mm = MemoryManager()

    # Store a message
    mm.store_message("test_agent", "This is a test message")

    # Store a document
    test_doc = Document(
        page_content="This is a test document content",
        metadata={"source": "test", "filename": "test.pdf", "page": 1}
    )
    mm.store_document(test_doc)

    # Store working data
    mm.store_working_data("test_result", {"found": True, "count": 5})

    # Test context building
    context = mm.build_agent_context("test_agent", "test")
    print(f"Context length: {len(context)} chars")
    print(f"Context preview: {context[:100]}...")

    # 5. Test Session Manager
    print("\n=== Testing Session Manager ===")
    sm = SessionManager(mm)

    # Create new session
    session_id = sm.start_new_session("Test Session", {"purpose": "testing"})
    print(f"Created session: {session_id}")

    # Save session
    sm.save_current_session()
    print("Session saved")

    # List sessions
    sessions = sm.list_available_sessions()
    print(f"Available sessions: {len(sessions)}")

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    test_memory_systems()
