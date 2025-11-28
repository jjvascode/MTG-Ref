from typing import List, Dict
from pathlib import Path
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import json

from src.config import OPENAI_API_KEY

# Prompts directory
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class MTGRetriever:
    """
    Retrieves relevant MTG cards and rules from the vector store.
    Uses LLM-based query routing to intelligently determine retrieval strategy.
    Provides formatted context for the LLM generator.
    """

    def __init__(self, vector_store, use_llm_routing: bool = True):
        """
        Initialize the retriever with a vector store instance.

        Arguments:
            vector_store: MTGVectorStore instance
            use_llm_routing: Whether to use LLM for query classification (default: True)
        """
        self.vector_store = vector_store
        self.use_llm_routing = use_llm_routing

        # Initialize a small, fast LLM for query routing
        if use_llm_routing:
            self.router_llm = ChatOpenAI(
                model="gpt-4o-mini",
                openai_api_key=OPENAI_API_KEY,
                temperature=0  # Deterministic routing
            )

    def classify_query_with_llm(self, query: str) -> Dict[str, int]:
        """
        Use LLM to classify query and determine optimal retrieval strategy.
        Determined prompt will guide LLM to analyze query and return amount of cards and rules to retrieve
        """
        # Load the prompt template that will be used by the LLM
        prompt_file = PROMPTS_DIR / "query_classifier.txt"
        with open(prompt_file, 'r', encoding='utf-8') as f: 
            classification_prompt_template = f.read()

        # Format the prompt with the query
        classification_prompt = classification_prompt_template.format(query=query)

        # Invoke LLM
        response = self.router_llm.invoke(classification_prompt)
        result = json.loads(response.content.strip())

        # set the values to the max value between 0 and either 5 or the value given by the LLM (or 3 as a default)
        k_cards = max(0, min(5, result.get('k_cards', 3)))
        k_rules = max(0, min(5, result.get('k_rules', 3)))

        return {'k_cards': k_cards, 'k_rules': k_rules}

    def retrieve(self, query: str) -> Dict[str, List[Document]]:
        """
        
        """
        # Use LLM or rule-based classification to determine retrieval counts
        classification = self.classify_query_with_llm(query)

        # Call hybrid_retrieve with determined k values
        return self.hybrid_retrieve(
            query,
            k_cards=classification['k_cards'],
            k_rules=classification['k_rules']
        )

    def hybrid_retrieve(self, query: str, k_cards: int = 3, k_rules: int = 3) -> Dict[str, List[Document]]:
        """
        Retrieve both cards and rules for a query.
        This is the main retrieval method for the RAG system.

        Arguments:
            query: User's search query
            k_cards: Number of cards to retrieve
            k_rules: Number of rule chunks to retrieve

        Returns:
            Dictionary with cards and rules keys containing Document lists
        """

        # This prevents cards from dominating results (there are 25K+ cards vs few rule chunks)
        # Search 1: Get extra results to filter for cards only
        all_results = self.vector_store.search(query, k=(k_cards + k_rules) * 3)

        cards = []
        rules = []

        # Separate cards and rules from combined search
        for doc in all_results:
            doc_type = doc.metadata.get('type', '')

            if doc_type == 'card' and len(cards) < k_cards:
                cards.append(doc)
            elif doc_type == 'rule' and len(rules) < k_rules:
                rules.append(doc)

            # Stop if we have enough of both
            if len(cards) >= k_cards and len(rules) >= k_rules:
                break

        # Search 2: If we didn't get enough rules, do a rule-focused search
        # This helps with queries like "How does deathtouch work?" which need rules
        if len(rules) < k_rules:
            # Add keywords to bias toward rules
            rule_query = f"{query} rules comprehensive"
            rule_results = self.vector_store.search(rule_query, k=k_rules * 3)

            for doc in rule_results:
                if doc.metadata.get('type') == 'rule' and len(rules) < k_rules:
                    # Avoid duplicates
                    if doc not in rules:
                        rules.append(doc)

                if len(rules) >= k_rules:
                    break

        return {
            'cards': cards,
            'rules': rules
        }

    def format_hybrid_context(self,hybrid_results: Dict[str, List[Document]]) -> str:
        """
        Format hybrid retrieval results (cards + rules) into context for the LLM.

        Arguments:
            hybrid_results: Dictionary with 'cards' and 'rules' keys

        Returns:
            Formatted context string that will be fed to the LLM
        """

        context_parts = []

        # Add cards section
        cards = hybrid_results.get('cards', [])
        if cards:
            context_parts.append("RELEVANT CARDS:\n")
            # Enumerate through to get both the index of the card (for numbering) and the doc information
            for i, doc in enumerate(cards, 1):
                name = doc.metadata.get('name', 'Unknown')
                mana_cost = doc.metadata.get('mana_cost', '')
                card_type = doc.metadata.get('card_type', '')

                # Format card header
                # Number the card and include the name 
                header = f"[Card {i}] {name}"
                # if the card has a cost, add it to the header
                if mana_cost:
                    header += f" {mana_cost}"

                # Append the header
                context_parts.append(header)

                # Add type line
                if card_type:
                    context_parts.append(f"Type: {card_type}")

                # Add card text by accessing the page content of the document object 
                context_parts.append(f"{doc.page_content}\n")

        # Add rules section
        rules = hybrid_results.get('rules', [])
        if rules:
            # if we have added cards, print a new line so text is not cramped
            if cards:  
                context_parts.append("\n")

            # Add the relevant rules
            context_parts.append("RELEVANT RULES:\n")
            # Enumerate through rules
            for i, doc in enumerate(rules, 1):
                chunk_index = doc.metadata.get('chunk_index', 'N/A')
                context_parts.append(f"[Rule Chunk {chunk_index}]")
                context_parts.append(f"{doc.page_content}\n")

        # If no results at all
        if not cards and not rules:
            return "No relevant cards or rules found for this query."

        
        return "\n".join(context_parts)
