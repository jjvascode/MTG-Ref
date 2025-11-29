import json
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import (
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    PROCESSED_DATA_DIR
)


class MTGVectorStore:
    """
    Manages the ChromaDB vector store for MTG cards and rules.
    Uses LangChain for embeddings and persistent ChromaDB storage.

    """

    def __init__(self):
        """Initialize LangChain OpenAI embeddings and persistent ChromaDB."""

        # Ensure ChromaDB directory exists
        CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

        # Initialize LangChain OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )

        # Initialize Chroma vector store with persistent storage
        self.vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_DB_DIR)  # Persistent storage location
        )

    
    # Helper to load data into chromadb
    def _load_items(self, filepath: str) -> int:
        """
        Function to load items from JSON and add to chromadb
        Takes in the filepath and will return an int that is the length of the items loaded
        """

        # Load processed information from JSON (convert to python object for embeddings and insertion into db)
        with open(filepath, 'r', encoding='utf-8') as f:
            items = json.load(f)

        # Convert each item to a langchain document object
        # ChromaDB doesn't support lists in metadata, so convert lists to strings
        documents = []
        for item in items:
            # Clean metadata - convert lists to comma-separated strings
            # if the value of a metadata key is a list, convery to a str by joining with commas
            clean_metadata = {}
            for key, value in item['metadata'].items():
                if isinstance(value, list):
                    clean_metadata[key] = ', '.join(str(v) for v in value)
                else:
                    clean_metadata[key] = value

            # Append the converted documents
            documents.append(Document(
                page_content=item['text'],
                metadata=clean_metadata
            ))

        # Add the above documents to the vector store
        # LangChain automatically generates embeddings and stores them
        batch_size = 1000
        # Ceiling division trick to ensure all documents are processed
        batch_amount = (len(documents) + batch_size -1) // batch_size

        for idx in range(batch_amount):
            # start the batch at the idx times 1000 (ex. idx = 0, start = 0; idx = 1, start = 1000, etc.)
            # end is 1000 cards after the start
            # should batch from 0-999, 1000-1999, etc.
            start = idx * batch_size
            end = start + batch_size

            # slicing to get the batch
            # add each batch to the vector store
            batch = documents[start:end]
            self.vectorstore.add_documents(batch)

        return len(documents)


    def load_cards(self) -> int:
        """
        Load processed cards from JSON into ChromaDB with embeddings.

        Returns:
            Number of cards loaded
        """
        print("\nLoading cards into vector store")

        cards_file = PROCESSED_DATA_DIR / "processed_cards.json"

        # If there is not processed card data, error
        if not cards_file.exists():
            raise FileNotFoundError(f"Processed cards file not found: {cards_file}")
        
        # call helper function to load items
        count = self._load_items(str(cards_file))
        print(f"Loaded {count} cards into chromadb")
        return count

    def load_rules(self) -> int:
        """
        Load processed rules from JSON into ChromaDB with embeddings.

        Returns:
            Number of rule chunks loaded
        """

        print("\nLoading rules into vector store")

        rules_file = PROCESSED_DATA_DIR / "processed_rules.json"

        # if there is no processed rules, error
        if not rules_file.exists():
            raise FileNotFoundError(f"Processed rules file not found: {rules_file}")

        # call helper function to load items
        count = self._load_items(str(rules_file))
        print(f"Loaded {count} cards into chromadb")
        return count

    def load_all_data(self) -> Dict[str, int]:
        """
        Load both processed cards and rules into the vector store.

        Returns:
            Dictionary with counts of loaded items
        """
        print("Loading Data into ChromaDB Vector Store")

        cards_count = self.load_cards()
        rules_count = self.load_rules()

        total = self.vectorstore._collection.count()

        print("Vector Store Loading Complete!")
        print(f"Total documents in vector store: {total}")

        return {
            'cards': cards_count,
            'rules': rules_count,
            'total': total
        }

    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search the vector store for relevant documents (cards and rules).

        Args:
            query: Search query text
            k: Number of results to return

        Returns:
            List of Document objects with content and metadata
        """
        results = self.vectorstore.similarity_search(query, k=k)
        return results

    def search_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """
        Search the vector store and return results with relevance scores.

        Args:
            query: Search query text
            k: Number of results to return

        Returns:
            List of tuples (Document, score) where lower score = more relevant
        """
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results

    # function to clear the database
    def clear_database(self):
        """Delete all data from the ChromaDB collection."""
        try:
            self.vectorstore._client.delete_collection(CHROMA_COLLECTION_NAME)
            print(f"Deleted collection: {CHROMA_COLLECTION_NAME}")

            # Recreate empty collection
            self.vectorstore = Chroma(
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(CHROMA_DB_DIR)
            )
            print("Created fresh empty collection")
        except Exception as e:
            print(f"Error clearing database: {e}")


if __name__ == "__main__":

    mtg_vector_store = MTGVectorStore()

    # Clear existing data
    mtg_vector_store.clear_database()

    mtg_vector_store.load_all_data()
