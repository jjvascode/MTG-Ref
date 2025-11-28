"""
Test script for MTGRetriever - demonstrates retrieval functionality
"""
from src.vectorstore.vector_store import MTGVectorStore
from src.retrieval.retriever import MTGRetriever


def test_retriever():
    """Test the retriever with sample queries."""

    print("Initializing vector store...")
    vector_store = MTGVectorStore()

    print("Initializing retriever...")
    retriever = MTGRetriever(vector_store)

    # Test queries demonstrating different types
    test_query = "How does commander work? "  # Rules query

    # Use smart retrieval (LLM automatically determines k_cards and k_rules)
    results = retriever.retrieve(test_query)

    # Format context (what gets sent to LLM)
    print("Formatted Contex:")
    context = retriever.format_hybrid_context(results)
    print(context)


if __name__ == "__main__":
    test_retriever()
