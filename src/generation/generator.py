
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from src.config import OPENAI_API_KEY, LLM_MODEL


class MTGGenerator:
    """
    Generates responses using LLMs
    Basing responses on provided context
    """
    def __init__(self, retriever):
        """ Initialize generator with a retriever"""

        self.retriever = retriever

        # initialize the LLM
        self.llm = ChatOpenAI()

