from dotenv import load_dotenv
from app.graph.workflow import app

load_dotenv()

REPO_URL = "https://github.com/RonaldForte/icebreaker.git"
BRANCH_NAME = "dev_amit"

def run_assistant():
    print("Git RAG Assistant (LangGraph) - Type 'exit' to quit.")
    while True:
        user_query = input("\nUser: ")
        if user_query.lower() == "exit":
            break
            
        inputs = {
            "query": user_query,
            "repo_url": REPO_URL,
            "branch": BRANCH_NAME
        }
        # Run the graph synchronously
        result = app.invoke(inputs)
        
        print(f"\n[Filters applied: Branch={result['branch']}, File={result['filename']}]")
        print(f"Assistant: {result['answer']}")

if __name__ == "__main__":
    run_assistant()