
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from langchain_core.output_parsers import StrOutputParser

from config import OPENAI_API_KEY, LLM_MODEL

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

class MTGGenerator:
    """
    Generates responses using LLMs
    Basing responses on provided context
    """
    def __init__(self, retriever):
        """ Initialize generator with a retriever"""

        # retriever instance to get context 
        self.retriever = retriever

        # initialize the LLM
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key = OPENAI_API_KEY,
            temperature=0.3 
        )

        # Load the prompt template and create prompt
        prompt_file = PROMPTS_DIR / "mtg_prompt_template.txt"
        with open(prompt_file, 'r', encoding = 'utf-8') as f: 
            system_prompt = f.read()
    
        # Create chat prompt template.
        # Used tuples to define role of each message 
        # system -> instructions from prompt template, user -> question/query and context
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt), 
            ("user", "{question}\n\nContext:\n{context}")
        ])

        # Implement an LCEL chain from link below:
        # https://www.pinecone.io/learn/series/langchain/langchain-expression-language/
        self.chain = self.prompt | self.llm | StrOutputParser()

    
    def generate_response(self, question: str) -> str:
        """
        Function to generate a response using the compelte RAG pipeline

        Arguments: 
            str: question/query from user
        Returns: 
            str: final generate response from LLM
        """

        # Retrieve context using retriever 
        results = self.retriever.retrieve(question)

        # Formate the result
        context = self.retriever.format_hybrid_context(results)

        # Run LCEL chain to generate response
        answer = self.chain.invoke({
            "question": question, 
            "context": context
        })

        return answer





