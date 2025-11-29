from vectorstore.vector_store import MTGVectorStore
from retrieval.retriever import MTGRetriever
from generation.generator import MTGGenerator


print("Loading MTG Ref")
# Intialize vector store, retriever and generator
vector_store = MTGVectorStore()
retriever = MTGRetriever(vector_store)
generator = MTGGenerator(retriever)

def main():

    print("\nWelcome to MTG Ref\n")
    print("Ask me any questions about Magic: The Gathering cards or rules.\n")
    print("If you no longer have any questions, type 'exit' to quit.\n")
    
    while True: 
        question = input("Your Question: ").strip()

        if question.lower() == 'exit':
            print("exiting MTG Ref. Goodbye")
            break

        try: 
            response = generator.generate_response(question)
            print(f"MTG Ref: {response}\n")
        except Exception as e: 
            print(f"An error has occured, please try again.\n Error: {e}\n")




if __name__ == "__main__": 
    main()