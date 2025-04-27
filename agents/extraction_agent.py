"""
Module: agents/extraction_agent.py

Purpose:
Defines the `ExtractionAgent` class, a specialized agent responsible for extracting factual information,
entities, and metadata from document segments. The agent focuses on information extraction at a segment-level
and ensures all extracted outputs are structured, categorized, and associated with confidence scores.

Key Responsibilities:
1. Information Extraction:
   - Identify and extract key facts, entities, and numerical/statistical data.
   - Extract document metadata (title, authors, dates, identifiers, etc.).
   - Perform regex-based pattern matching and LLM-based reasoning for extraction.

2. Fact Structuring:
   - Organize extracted outputs into JSON objects with standardized fields:
     - `fact`, `type`, `entities`, `confidence`, `source_text`, etc.
   - Include additional metadata like extraction time, agent ID, and relationships.

3. Workflow Integration:
   - Process tasks assigned by the Coordinator Agent.
   - Return extracted results to the workflow and store them in memory (short-term and long-term).
   - Report results back to the Coordinator upon task completion.

Constructor Parameters:
- `agent_id`, `name`: Identifies the Extraction Agent and its assigned identity.
- `memory_manager`: Instance of `MemoryManager` for maintaining and retrieving extracted data.
- `model`, `temperature`, `max_tokens`: Configuration for the underlying LLM, optimized for factual extraction.

Core Methods:
1. Entity Extraction:
   - `extract_entities`: Uses regex for pattern-based entity recognition (email, URL, phone, etc.).
2. Fact Extraction:
   - `extract_key_facts`: Uses LLM prompts to extract facts and associate confidence scores, types, and entities.
3. Metadata Extraction:
   - `extract_document_metadata`: Focuses on extracting metadata specific to document types.

Custom Actions:
- `_action_extract_entities`, `_action_extract_facts`, `_action_extract_metadata`:
  Perform extractions and store results in memory.
- `process_task`:
  Delegated by the Coordinator to handle assigned tasks (e.g., entity or fact extraction for a document segment).

Integration Notes:
- Designed to work with the Coordinator Agent, leveraging Working Memory for task-relevant data.
- Extends `BaseAgent` to inherit core functionalities like reasoning, communication, and memory integration.
- Built for extensibility, enabling future enhancements (e.g., improved Named Entity Recognition models).
"""

from typing import Dict, Any, List, Callable
import json
from datetime import datetime
import re
from agents.base_agent import BaseAgent
from memory.memory_manager import MemoryManager


class ExtractionAgent(BaseAgent):
    """
    Extraction Agent responsible for identifying key facts, entities, and metadata
    from document segments.
    """
    def __init__(
        self,
        agent_id: str,
        name: str,
        memory_manager: MemoryManager,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,  # Lower temperature for more factual extraction
        max_tokens: int = 1000
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            role="Information Extraction Specialist",
            memory_manager=memory_manager,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Set extraction-specific instructions
        extraction_instructions = """
        As the Extraction Agent, your primary responsibilities are:

        1. Identify and extract key facts from document segments
        2. Recognize named entities (people, organizations, locations, dates, etc.)
        3. Extract structured data and relationships
        4. Identify numerical data and statistics
        5. Extract document metadata (dates, authors, references, etc.)
        6. Structure extracted information in a consistent format
        7. Assign confidence scores to extracted information

        Be precise and objective in your extractions. Focus on factual information
        rather than interpretations or analyses. Include relevant context with each
        extraction to ensure accurate interpretation.

        When extracting information, provide:
        - The extracted fact/entity
        - The category or type of information
        - The location in the document
        - A confidence score (0.0-1.0)
        - Any relevant relationships to other extracted information
        """

        self.set_additional_instructions(extraction_instructions)

        # Entity types this agent specializes in
        self.entity_types = [
            "person", "organization", "location", "date", "time",
            "money", "percentage", "email", "phone", "url",
            "product", "event", "document_reference", "concept", "technical_term"
        ]

        # Regular expressions for common patterns
        self.patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "url": r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',
            "phone": r'\b(?:\+\d{1,3}\s?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            "date": r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})|(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)' +
                    r'[a-z]* \d{1,2},? \d{4})\b'
        }

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text using pattern matching
        This is a basic implementation that can be enhanced with NER models
        """
        entities = []

        # Simple pattern matching for some entity types
        for entity_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text)

            for match in matches:
                entities.append({
                    "text": match.group(0),
                    "type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.9,  # High confidence for regex matches
                    "extraction_method": "pattern"
                })
        return entities

    def extract_key_facts(self, text: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Extract key facts from text using LLM-based reasoning
        """
        extraction_prompt = f"""
        Extract key facts from the following text. For each fact:
        1. State the fact clearly and concisely
        2. Categorize the fact type
        3. Identify related entities
        4. Assign a confidence score (0.0-1.0)

        Format each fact as a JSON object with the following structure:
        {{
            "fact": "The stated fact",
            "type": "category of the fact",
            "entities": ["related", "entities"],
            "confidence": 0.9,
            "source_text": "direct quote from source containing this fact"
        }}

        Text to analyze:
        ```
        {text}
        ```

        Return the facts as a JSON array.
        """

        # Think through the extraction process
        thinking_input = {
            "text": text,
            "context": context or {}
        }

        thinking_result = self.think(
            thinking_input,
            context_query="fact extraction",
            reasoning_prompt=extraction_prompt,
            action_options=["store_fact", "communicate"],
        )

        # Try to parse the extracted facts from the reasoning output
        try:
            # Look for JSON array in the text
            facts_match = re.search(r'\[\s*{\s*"fact".*}\s*\]', thinking_result["reasoning"]["output"], re.DOTALL)
            if facts_match:
                facts_json = facts_match.group(0)
                facts = json.loads(facts_json)
            else:
                # Try to find individual fact objects and combine them
                fact_matches = re.finditer(r'{\s*"fact"[^}]*}', thinking_result["reasoning"]["output"], re.DOTALL)
                facts = []
                for match in fact_matches:
                    try:
                        fact = json.loads(match.group(0))
                        facts.append(fact)
                    except json.JSONDecodeError:
                        continue

            # Add metadata to each fact
            for fact in facts:
                fact["extraction_time"] = datetime.now().isoformat()
                fact["extraction_agent"] = self.agent_id

            return facts

        except (json.JSONDecodeError, AttributeError) as e:
            # If parsing fails, return a structured error
            return [{
                "error": f"Failed to parse extracted facts: {str(e)}",
                "raw_output": thinking_result["reasoning"]["output"],
                "extraction_time": datetime.now().isoformat(),
                "extraction_agent": self.agent_id
            }]

    def extract_document_metadata(self, text: str, doc_type: str = None) -> Dict[str, Any]:
        """
        Extract document metadata based on document type
        """
        metadata_prompt = f"""
        Extract metadata from this {doc_type} document. Include:

        - Title or heading
        - Author(s) or creator(s)
        - Date of creation/publication
        - Document type or category
        - Organization or source
        - Version or revision information
        - Any identifiers (IDs, DOIs, URLs)
        - Keywords or tags

        Only include metadata that is explicitly present in the text.
        Return the metadata as a JSON object.

        Document text:
        ```
        {text}
        ```
        """

        # Set default doc type if not provided
        doc_type = doc_type or "unspecified"

        # Think through the metadata extraction
        thinking_input = {
            "text": text[:1000],  # Use beginning of document for metadata
            "doc_type": doc_type
        }

        reasoning_prompt = metadata_prompt.format(doc_type=doc_type, text=text[:1000])

        thinking_result = self.think(
            thinking_input,
            context_query="document metadata",
            reasoning_prompt=reasoning_prompt,
            action_options=["store_fact", "communicate"],
        )

        # Try to parse the metadata from the reasoning output
        try:
            # Look for JSON object in the text
            metadata_match = re.search(r'{\s*"[^"]+"\s*:.*}', thinking_result["reasoning"]["output"], re.DOTALL)
            if metadata_match:
                metadata_json = metadata_match.group(0)
                metadata = json.loads(metadata_json)

                # Add extraction metadata
                metadata["extraction_time"] = datetime.now().isoformat()
                metadata["extraction_agent"] = self.agent_id
                metadata["confidence"] = 0.8  # Default confidence

                return metadata
            else:
                return {
                    "error": "Could not locate metadata JSON in output",
                    "extraction_time": datetime.now().isoformat(),
                    "extraction_agent": self.agent_id,
                    "raw_output": thinking_result["reasoning"]["output"]
                }

        except json.JSONDecodeError:
            return {
                "error": "Failed to parse metadata as JSON",
                "extraction_time": datetime.now().isoformat(),
                "extraction_agent": self.agent_id,
                "raw_output": thinking_result["reasoning"]["output"]
            }

    # Custom actions for this agent

    def _action_extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Action to extract entities from text
        """
        entities = self.extract_entities(text)

        # Store extracted entities in memory
        entity_key = f"entities_{self.agent_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.memory.store_working_data(entity_key, {
            "entities": entities,
            "text_sample": text[:100],
            "extraction_time": datetime.now().isoformat()
        })
        return {
            "success": True,
            "entities": entities,
            "count": len(entities)
        }

    def _action_extract_facts(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Action to extract key facts from text
        """
        facts = self.extract_key_facts(text, context)

        # Store extracted facts in memory
        facts_key = f"facts_{self.agent_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.memory.store_working_data(facts_key, {
            "facts": facts,
            "text_sample": text[:100],
            "extraction_time": datetime.now().isoformat()
        })

        # Also store each fact individually in long-term memory for later retrieval
        for i, fact in enumerate(facts):
            if "error" not in fact:  # Only store valid facts
                fact_id = f"fact_{self.agent_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}"
                self.memory.store_fact(fact_id, fact)
        return {
            "success": True,
            "facts": facts,
            "count": len(facts)
        }

    def _action_extract_metadata(self, text: str, doc_type: str = None) -> Dict[str, Any]:
        """
        Action to extract document metadata
        """
        metadata = self.extract_document_metadata(text, doc_type)

        # Store metadata in memory
        metadata_key = f"metadata_{self.agent_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.memory.store_working_data(metadata_key, {
            "metadata": metadata,
            "doc_type": doc_type,
            "extraction_time": datetime.now().isoformat()
        })

        return {
            "success": True,
            "metadata": metadata
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

        extraction_actions = {
            "extract_entities": self._action_extract_entities,
            "extract_facts": self._action_extract_facts,
            "extract_metadata": self._action_extract_metadata
        }

        return {**base_actions, **extraction_actions}

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task assigned by the coordinator
        """
        task_type = task.get("task_type")
        segment_id = task.get("segment_id")
        parameters = task.get("parameters", {})

        # Retrieve the document segment content
        segment_data = self.memory.retrieve_document_segment(segment_id) if segment_id else None
        text_content = segment_data.get("content") if segment_data else parameters.get("text", "")

        # Process based on task type
        if task_type == "extract_entities":
            result = self._action_extract_entities(text_content)
        elif task_type == "extract_facts":
            result = self._action_extract_facts(text_content, parameters.get("context"))
        elif task_type == "extract_metadata":
            result = self._action_extract_metadata(text_content, parameters.get("doc_type"))
        else:
            result = {
                "success": False,
                "error": f"Unknown task type: {task_type}"
            }

        # Report results back to the coordinator
        self.send_message(parameters.get("coordinator_id", "coordinator"), json.dumps({
            "task_id": task.get("task_id"),
            "result": result,
            "status": "completed" if result.get("success", False) else "failed"
        }), {
            "message_type": "task_result",
            "task_id": task.get("task_id")
        })

        return result
