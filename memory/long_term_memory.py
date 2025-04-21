import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_community.embeddings import OpenAIEmbeddings
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
        except:
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

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents based on query
        """
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
        except:
            return None

    def clear(self) -> None:
        """Clear the vector store"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
