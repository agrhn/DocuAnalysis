"""
Module: memory/long_term_memory.py

Purpose:
Implements the LongTermMemory class, providing persistent document memory using ChromaDB for embedding storage
and retrieval, along with structured fact storage in JSON format. This is a core part of the long-term memory
system in the Document Analysis Assistant project.

Key Responsibilities:
- Stores document embeddings in a persistent vector database (ChromaDB)
- Supports semantic search over stored documents using OpenAI embeddings
- Stores and retrieves structured facts extracted from documents as JSON files

Class: LongTermMemory

Constructor:
- LongTermMemory(collection_name: str = "document_memory", embeddings_model: str = "text-embedding-3-small")
    - Initializes ChromaDB collection and OpenAI embeddings
    - Sets up paths for persistent storage

Key Methods:
- add_documents(documents: List[Document]) -> None
    - Embeds and stores documents in the vector database
    - `Document` must have `page_content` and optional `metadata`

- search(query: str, n_results: int = 5) -> List[Dict[str, Any]]
    - Performs semantic search using embedded query
    - Returns list of dicts with `content`, `metadata`, and `id`

- store_fact(fact_id: str, content: Dict[str, Any]) -> None
    - Saves structured fact to disk using a unique identifier

- retrieve_fact(fact_id: str) -> Optional[Dict[str, Any]]
    - Loads previously stored fact if it exists

- clear() -> None
    - Clears and reinitializes the vector store for fresh indexing

Integration Notes:
- Used by Extraction Agent and Coordinator Agent for storing and retrieving facts and embeddings
- Document embeddings can be searched by any agent needing context or related content
- Fact storage enables structured output reuse and persistence between sessions
"""

import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import hashlib
import json
from dotenv import load_dotenv

load_dotenv()


class LongTermMemory:
    """
    Long-term memory using ChromaDB for storing document embeddings
    and important information for persistent retrieval.
    """
    def __init__(self, collection_name: str = "document_memory", embeddings_model: str = "text-embedding-3-small"):
        self.storage_path = "data/memory/long_term/"
        os.makedirs(self.storage_path, exist_ok=True)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.storage_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Create or get collection
        try:
            self.collection = self.client.get_collection(collection_name)
        except chromadb.errors.NotFoundError:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"),
                                           model=embeddings_model)

        # Additional storage for structured data
        self.facts_path = f"{self.storage_path}facts/"
        os.makedirs(self.facts_path, exist_ok=True)

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store
        """
        for doc in documents:
            # Generate a stable ID based on content
            doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()

            # Get embedding
            embedding = self.embeddings.embed_query(doc.page_content)

            # Add to collection
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[doc.metadata],
                documents=[doc.page_content]
            )

    def set_collection(self, collection_name: str) -> None:
        """
        Dynamically set the target collection for semantic search.
        """
        try:
            self.collection = self.client.get_collection(collection_name)
        except chromadb.errors.NotFoundError:
            raise ValueError(f"Collection '{collection_name}' does not exist.")

    def search(self, query: str, n_results: int = 5, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant documents based on query within a specific collection.
        """
        if collection_name:
            self.set_collection(collection_name)  # Dynamically switch collections

        # Get embedding for query
        query_embedding = self.embeddings.embed_query(query)

        # Search collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        # Format results
        formatted_results = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                formatted_results.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "id": results["ids"][0][i] if results["ids"] else "unknown"
                })

        return formatted_results

    def store_fact(self, fact_id: str, content: Dict[str, Any]) -> None:
        """
        Store structured facts extracted from documents
        """
        with open(f"{self.facts_path}{fact_id}.json", 'w') as f:
            json.dump(content, f)

    def retrieve_fact(self, fact_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a stored fact by ID
        """
        try:
            with open(f"{self.facts_path}{fact_id}.json", 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def clear(self) -> None:
        """Clear the vector store"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
