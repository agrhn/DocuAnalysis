import os
from agents.coordinator_agent import CoordinatorAgent
from agents.extraction_agent import ExtractionAgent
from agents.summarization_agent import SummarizationAgent
from agents.critical_analysis_agent import CriticalAnalysisAgent
from agents.qa_agent import QAAgent
from memory.memory_manager import MemoryManager
from utils.pdf_loader import load_pdf


def main_loop():
    """
    Main loop that manages agent interactions and task allocation for document processing.
    """
    print("Starting the Document Analysis System...")

    # Step 1: Initialize MemoryManager and Agents
    memory_manager = MemoryManager()

    coordinator = CoordinatorAgent(
        agent_id="coordinator",
        name="CoordinatorAgent",
        memory_manager=memory_manager
    )
    extraction_agent = ExtractionAgent(
        agent_id="extraction_agent",
        name="ExtractionAgent",
        memory_manager=memory_manager
    )
    summarization_agent = SummarizationAgent(
        agent_id="summarization_agent",
        name="SummarizationAgent",
        memory_manager=memory_manager
    )
    critical_analysis_agent = CriticalAnalysisAgent(
        agent_id="critical_agent",
        name="CriticalAnalysisAgent",
        memory_manager=memory_manager
    )
    qa_agent = QAAgent(
        agent_id="qa_agent",
        name="QAAgent",
        memory_manager=memory_manager
    )

    # Step 2: Load Document
    document_path = "data/sample_document.pdf"  # Path to an example PDF document
    if not os.path.exists(document_path):
        raise FileNotFoundError(f"Cannot find document at: {document_path}")

    print("Loading document...")
    document_text, metadata = load_pdf(document_path)
    document_id = metadata.get("filename", "sample_document")
    print(f"Document loaded: {metadata.get('title', 'Unknown Title')}, ID: {document_id}")

    # Step 3: Initialize Workflow in Coordinator
    coordinator.initialize_workflow(
        document_id=document_id,
        document_metadata=metadata
    )

    # Step 4: Segment Document
    print("Segmenting document...")
    segments = coordinator.segment_document(document_text, segment_size=2000)
    print(f"Document segmented into {len(segments)} segments.")

    # Step 5: Main Coordination Loop
    for segment in segments:
        if segment["status"] != "unprocessed":
            continue

        print(f"Processing segment: {segment['segment_id']}")

        # Assign extraction task
        print(f"Assigning extraction task for segment: {segment['segment_id']}")
        extraction_task_id = coordinator.assign_task(
            agent_id="extraction_agent",
            task_type="extract_entities",
            segment_id=segment["segment_id"],
            parameters={}
        )
        extraction_task_result = extraction_agent.process_task({
            "task_type": "extract_facts",
            "segment_id": segment["segment_id"],
            "parameters": {}
        })
        print(f"Extraction Result for {segment['segment_id']}: {extraction_task_result}")

        # Assign summarization task
        print(f"Assigning summarization task for segment: {segment['segment_id']}")
        summarization_task_id = coordinator.assign_task(
            agent_id="summarization_agent",
            task_type="create_summary",
            segment_id=segment["segment_id"],
            parameters={"style": "brief"}
        )
        summarization_task_result = summarization_agent.process_task({
            "task_type": "create_summary",
            "segment_id": segment["segment_id"],
            "parameters": {"style": "brief"}
        })
        print(f"Summarization Result for {segment['segment_id']}: {summarization_task_result}")

        # Assign critical analysis task
        print(f"Assigning critical analysis task for segment: {segment['segment_id']}")
        critical_task_id = coordinator.assign_task(
            agent_id="critical_agent",
            task_type="analyze_quality",
            segment_id=segment["segment_id"],
            parameters={"focus_areas": ["reasoning", "evidence"]}
        )
        critical_task_result = critical_analysis_agent.process_task({
            "task_type": "analyze_quality",
            "segment_id": segment["segment_id"],
            "parameters": {"focus_areas": ["reasoning", "evidence"]}
        })
        print(f"Critical Analysis Result for {segment['segment_id']}: {critical_task_result}")

        # Mark tasks as completed in the workflow
        coordinator.update_task_status(
            task_id=extraction_task_id,
            status="completed",
            result=extraction_task_result
        )
        coordinator.update_task_status(
            task_id=summarization_task_id,
            status="completed",
            result=summarization_task_result
        )
        coordinator.update_task_status(
            task_id=critical_task_id,
            status="completed",
            result=critical_task_result
        )

        # Update workflow progress
        print("Workflow progress updated.")

    print("All segments processed!")

    # Compile document-level insights
    print("Compiling document analysis results...")
    analysis_results = coordinator.compile_analysis_results()
    print(f"Compiled Analysis Results:\n{analysis_results}")

    # Step 6: Handle User Q&A
    print("Starting Q&A Phase...")
    while True:
        question = input("Ask a question about the document (or type 'exit' to quit): ").strip()
        if question.lower() == "exit":
            break

        answer = qa_agent.answer_question(question)
        print(f"Answer: {answer['content']}")

    print("Exiting system. Goodbye!")


if __name__ == '__main__':
    main_loop()
