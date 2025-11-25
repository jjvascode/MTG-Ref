import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ChromaDB settings
CHROMA_COLLECTION_NAME = "mtg-knowledge-base"

# Model settings
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Scryfall API
SCRYFALL_BULK_DATA_URL = "https://api.scryfall.com/bulk-data"
SCRYFALL_CARDS_TYPE = "oracle_cards"  # Oracle cards (one version per card)

# MTG Comprehensive Rules
MTG_RULES_URL = "https://media.wizards.com/2024/downloads/MagicCompRules%2020241108.txt"
