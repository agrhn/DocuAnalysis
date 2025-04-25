import os
from memory.memory_manager import MemoryManager
from agents.base_agent import BaseAgent
from agents.coordinator_agent import CoordinatorAgent
from agents.extraction_agent import ExtractionAgent
from agents.summarization_agent import SummarizationAgent
from agents.critical_analysis_agent import CriticalAnalysisAgent
from agents.qa_agent import QAAgent


def test_agents():
    print("Testing Agent Systems...")

    # Initialize MemoryManager
    mm = MemoryManager()
    os.makedirs("data/agents/logs", exist_ok=True)

    # 1. Test BaseAgent
    print("\n=== Testing BaseAgent ===")
    base_agent = BaseAgent(
        agent_id="base_agent",
        name="BaseAgent Test",
        role="Basic Testing Agent",
        memory_manager=mm
    )

    # Test communication
    message_id = base_agent.send_message(
        recipient_id="test_recipient",
        content="Hello! This is a base agent test message."
    )
    print(f"Sent message ID: {message_id}")

    # Retrieve context
    context = base_agent.memory.build_agent_context(agent_id=base_agent.agent_id, query="test")
    print(f"Context built: {len(context)} chars")

    # Test reasoning process
    perception = base_agent.perceive("This is an input example.", context_query="test")
    reasoning = base_agent.reason(perception, "Explain the input above.")
    print(f"Reasoning Output: {reasoning['output']}")

    # 2. Test CoordinatorAgent
    print("\n=== Testing CoordinatorAgent ===")
    coordinator_agent = CoordinatorAgent(
        agent_id="coordinator_agent",
        name="Coordinator Test",
        memory_manager=mm
    )

    # Initialize workflow
    workflow = coordinator_agent.initialize_workflow(
        document_id="doc_1",
        document_metadata={"name": "Test Document"}
    )
    print(f"Initialized workflow: {workflow['workflow_id']}")

    # Segment document
    segments = coordinator_agent.segment_document(
        "This is a test document. It has multiple sentences spread across segments.", 20
    )
    print(f"Document segmented into {len(segments)} segments.")

    # Assign task
    task_id = coordinator_agent.assign_task(
        agent_id="extraction_agent",
        task_type="extract_entities",
        segment_id=segments[0]["segment_id"],
        parameters={"priority": "high"}
    )
    print(f"Assigned task ID: {task_id}")

    # 3. Test ExtractionAgent
    print("\n=== Testing ExtractionAgent ===")
    extraction_agent = ExtractionAgent(
        agent_id="extraction_agent",
        name="Extraction Test",
        memory_manager=mm
    )

    # Extract entities
    entities = extraction_agent.extract_entities("Email: test@example.com, URL: https://example.com")
    print(f"Extracted Entities: {entities}")

    # Extract key facts
    facts = extraction_agent.extract_key_facts("The population of the city is 1 million.")
    print(f"Extracted Facts: {facts}")

    # Process task
    extraction_result = extraction_agent.process_task({
        "task_type": "extract_entities",
        "segment_id": segments[0]["segment_id"],
        "parameters": {"context": {}}
    })
    print(f"Extraction Task Result: {extraction_result}")

    # 4. Test SummarizationAgent
    print("\n=== Testing SummarizationAgent ===")
    summarization_agent = SummarizationAgent(
        agent_id="summarization_agent",
        name="Summarization Test",
        memory_manager=mm
    )

    # Create summary
    summary = summarization_agent.create_summary("This document explains the details about AI systems.",
                                                 style="brief")
    print(f"Created Summary: {summary['summary']}")

    # Progressive Summary
    progressive_summaries = summarization_agent.create_progressive_summary(
        "This document explains AI systems at multiple levels of detail.",
        levels=["brief", "executive", "comprehensive"]
    )
    print(f"Created Progressive Summary with Levels: {progressive_summaries['summaries']}")

    # Process task
    summarization_result = summarization_agent.process_task({
        "task_type": "create_summary",
        "segment_id": segments[0]["segment_id"],
        "parameters": {"style": "executive"}
    })
    print(f"Summarization Task Result: {summarization_result}")

    # 5. Test CriticalAnalysisAgent
    print("\n=== Testing CriticalAnalysisAgent ===")
    critical_agent = CriticalAnalysisAgent(
        agent_id="critical_agent",
        name="Critical Analysis Test",
        memory_manager=mm
    )

    # Analyze content quality
    content_quality = critical_agent.analyze_content_quality(
        "The arguments in this document are logically inconsistent.",
        focus_areas=["evidence", "reasoning"]
    )
    print(f"Content Quality Analysis: {content_quality}")

    # Identify bias
    bias_analysis = critical_agent.identify_bias("This document uses emotionally charged language.")
    print(f"Bias Analysis: {bias_analysis}")

    # Provide perspectives
    perspectives = critical_agent.provide_alternative_perspectives(
        "The document discusses AI but ignores privacy concerns.", num_perspectives=3
    )
    print(f"Alternative Perspectives: {perspectives}")

    # Process task
    analysis_result = critical_agent.process_task({
        "task_type": "analyze_quality",
        "parameters": {"focus_areas": ["reasoning", "structure"]}
    })
    print(f"Critical Analysis Task Result: {analysis_result}")

    # 6. Test QAAgent
    mm = MemoryManager()
    print("\n=== Testing QAAgent ===")
    qa_agent = QAAgent(
        agent_id="qa_agent",
        name="Q&A Test",
        memory_manager=mm
    )

    # Answer question
    answer = qa_agent.answer_question("What is the importance of memory systems?")
    print(f"Answer: {answer['content']}")

    # Suggest questions
    suggestions = qa_agent.suggest_related_questions("What is the importance of memory systems?",
                                                     "Memory systems store essential information.")
    print(f"Suggested Questions: {suggestions}")

    # Process task
    qa_result = qa_agent.process_task({
        "task_type": "answer_question",
        "parameters": {"question": "What is the role of memory systems in AI?"}
    })
    print(f"QA Task Result: {qa_result}")

    # Clear conversation history
    qa_agent.clear_conversation_history()
    print("Cleared conversation history")

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    test_agents()
