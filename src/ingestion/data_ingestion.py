import requests
import json
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SCRYFALL_BULK_DATA_URL,
    SCRYFALL_CARDS_TYPE,
    MTG_RULES_URL,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


class MTGDataIngestion:
    """Handles downloading and processing MTG card data and rules. 
    
    
    Attributes:
        raw cards file: Path used to store the raw scryfall card data
        raw rules file: Path used to store the raw comprehensive rules data
        processed cards file: Path used to store the processed card data
        processed rules file: Path used to store the processed rules data
    
    """
    def __init__(self):

        # Initialize file paths/attributes
        self.raw_cards_file = RAW_DATA_DIR / "scryfall_raw_cards.json"
        self.raw_rules_file = RAW_DATA_DIR / "comprehensive_rules.txt"
        self.processed_cards_file = PROCESSED_DATA_DIR / "processed_cards.json"
        self.processed_rules_file = PROCESSED_DATA_DIR / "processed_rules.json"

        # Ensure directories to store data exist
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Initialize LangChain text splitter to chunk text for RAG embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )

    def fetch_scryfall_cards(self) -> None:
        """
        Download the latest Scryfall bulk card data.
        Uses the oracle_cards endpoint which has one entry per unique card to avoid reprints/special cards/foils vs normals, etc..
        """

        # Get the bulk data info from Scryfall API and convvert to JSON
        # raise error if the request fails
        response = requests.get(SCRYFALL_BULK_DATA_URL)
        response.raise_for_status()
        bulk_data = response.json()

        # Find the oracle cards download URL
        # if the type is equal to the cards type (oracle_cards), set that object to oracle_data and break from loop
        oracle_data = None
        for item in bulk_data['data']:
            if item['type'] == SCRYFALL_CARDS_TYPE:
                oracle_data = item
                break

        # If the designated type is not found, raise an error.
        if not oracle_data:
            raise ValueError(f"Could not find {SCRYFALL_CARDS_TYPE} in bulk data")

        # Set the download Url/uri to access all the oralce bulk cards. This is present within the JSON response
        download_url = oracle_data['download_uri']
        print(f"Downloading cards from: {download_url}")

        # Download the cards and check for errors
        response = requests.get(download_url, stream=True)
        response.raise_for_status()

        # Since file is large (159 MB), write in chunks to avoid issues with memory
        with open(self.raw_cards_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def fetch_comprehensive_rules(self) -> None:

        """Download the MTG Comprehensive Rules text file."""

        
        # Download the rules and check for errors
        response = requests.get(MTG_RULES_URL)
        response.raise_for_status()

        # Save the rules to file
        with open(self.raw_rules_file, 'wb') as f:
            f.write(response.content)


    # return a list of dicts whose keys are strs and values are any type (int, str, list, etc.)
    def process_cards(self) -> List[Dict[str, Any]]:
        """
        Process raw Scryfall cards into a format suitable for RAG.
        
        """
    
        # If the raw cards file does not exist raise an error
        if not self.raw_cards_file.exists():
            raise FileNotFoundError(f"Raw cards file not found: {self.raw_cards_file}")

        # open the raw cards data and convert to python object from JSON
        with open(self.raw_cards_file, 'r', encoding='utf-8') as f:
            raw_cards = json.load(f)

        # Empty list to hold all processed cards
        processed_cards = []

        # Go through each card in the raw cards
        # this is a list of dicts, so iterate through each item in list and access dict keys to gather needed information/values
        # for each card we extract the relevant information and append to the list of processed cards
        for card in raw_cards:
            # Skip tokens and non-game cards as these do not have any relevances to overall gameplay
            # Use .get to avoid error if some cards are missing layout value
            if card.get('layout') in ['token', 'emblem', 'art_series']:
                continue

            # Extract relevant information for building the text and metadata
            # If a field is missing, give a default value '' (empty string) or list
            # These fields can be found here: https://scryfall.com/docs/api/cards
            name = card.get('name')
            type_line = card.get('type_line')
            oracle_text = card.get('oracle_text', '')
            mana_cost = card.get('mana_cost', '')
            keywords = card.get('keywords', [])
            power = card.get('power')
            toughness = card.get('toughness')
            loyalty = card.get('loyalty')

            # Create searchable text for RAG
            # Combine relevant fields into a single text block
            text_parts = [
                f"Card Name: {name}",
                f"Type: {type_line}",
            ]

            # Check for optional fields, if they are present add them to the text parts for each card
            if mana_cost:
                text_parts.append(f"Mana Cost: {mana_cost}")

            if oracle_text:
                text_parts.append(f"Oracle Text: {oracle_text}")

            if power and toughness:
                text_parts.append(f"P/T: {power}/{toughness}")

            if loyalty:
                text_parts.append(f"Loyalty: {loyalty}")

            if keywords:
                text_parts.append(f"Keywords: {', '.join(keywords)}")

            # Build the processed card with clean structure: text + metadata
            # Text will be used for embeddings into vectore store and metadata for retrieval/filtering
            processed_card = {
                'text': '\n'.join(text_parts),
                'metadata': {
                    'source': 'scryfall',
                    'type': 'card',
                    'id': card.get('id'),
                    'name': name,
                    'type_line': type_line,
                    'oracle_text': oracle_text,
                    'mana_cost': mana_cost,
                    'cmc': card.get('cmc', 0),
                    'colors': card.get('colors', []),
                    'color_identity': card.get('color_identity', []),
                    'keywords': keywords,
                    'power': power,
                    'toughness': toughness,
                    'loyalty': loyalty,
                    'set_name': card.get('set_name'),
                    'rarity': card.get('rarity')
                }
            }

            # append each processed card to the list
            processed_cards.append(processed_card)

        # Save processed cards
        with open(self.processed_cards_file, 'w', encoding='utf-8') as f:
            json.dump(processed_cards, f, indent=2)

        # return list of cards
        return processed_cards

    
    def process_rules(self) -> List[Dict[str, Any]]:
        """
        Process the comprehensive rules into chunks suitable for RAG.
        Simply chunks the entire rules document using LangChain text splitter.
        """
        print("Processing comprehensive rules")

        # if the raw rules does not exist (not downloaded) raise an error
        if not self.raw_rules_file.exists():
            raise FileNotFoundError(f"Raw rules file not found: {self.raw_rules_file}")

        # Open the rules and read the text
        with open(self.raw_rules_file, 'r', encoding='utf-8') as f:
            rules_text = f.read()

        # Use LangChain text splitter to chunk the entire rules document
        chunks = self.text_splitter.split_text(rules_text)

        # Process each chunk into the format expected for RAG
        processed_rules = []
        for i, chunk in enumerate(chunks):
            processed_rules.append({
                'text': chunk,
                'metadata': {
                    'source': 'comprehensive_rules',
                    'type': 'rule',
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
            })

        # Save processed rules
        with open(self.processed_rules_file, 'w', encoding='utf-8') as f:
            json.dump(processed_rules, f, indent=2)

        print(f"Processed {len(processed_rules)} rule chunks")
        print(f"Saved to: {self.processed_rules_file}")

        return processed_rules

    def run_full_ingestion(self) -> None:
        """
        Run complete data ingestion pipeline.
        Downloads rules and card data, processes (and chunks where needed), then saves to directory.
        
        """

        print("Starting MTG Data Ingestion Pipeline")


        try:
            # fetch the scryfall card data
            print("\nFetching Scryfall card data")
            self.fetch_scryfall_cards()

            # fetch MTG comprehensive rules
            print("\nFetching MTG Comprehensive Rules")
            self.fetch_comprehensive_rules()

            # process scryfall card data
            print("\nProcessing card data")
            cards = self.process_cards()

            # process comprehensive rules
            print("\nProcessing rules data")
            rules = self.process_rules()

            # Summary of ingetsiong inclduing total card and chunks
            print(f"Total cards processed: {len(cards)}")
            print(f"Total rule chunks: {len(rules)}")
            print(f"\nProcessed data saved to: {PROCESSED_DATA_DIR}")

        except Exception as e:
            print(f"\nError during ingestion: {e}")
            raise


if __name__ == "__main__":
    ingestion = MTGDataIngestion()
    ingestion.run_full_ingestion()
