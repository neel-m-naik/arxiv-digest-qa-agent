from agent import (
    AgentState,
    node_query_understanding,
    node_arxiv_retrieval,
    node_parse_pdf,
    node_chunk_and_embed,
    node_generate_briefing,
    answer_question
)

def run_agent(user_query: str) -> AgentState:
    state = AgentState(raw_query=user_query)
    
    stages = [
        ("Query Understanding", node_query_understanding),
        ("arXiv Retrieval & Download", node_arxiv_retrieval),
        ("PDF Parsing", node_parse_pdf),
        ("Chunking & Vector Store Setup", node_chunk_and_embed),
        ("Generating Executive Briefing", node_generate_briefing)
    ]
    
    print("\n--- Starting Autonomous arXiv Agent Pipeline ---")
    for stage_name, node_fn in stages:
        print(f"[*] Executing: {stage_name}...")
        state = node_fn(state)
        if state.error:
            print(f"[!] Pipeline halted at '{stage_name}': {state.error}")
            return state
            
    print("[✓] Pipeline execution finished successfully.\n")
    return state

def main():
    print("=" * 60)
    print("     Autonomous arXiv Paper Digest & QA Agent")
    print("=" * 60)
    
    query = input("\nEnter arXiv ID, paper URL, or research topic:\n> ").strip()
    if not query:
        print("Input cannot be empty. Exiting.")
        return
        
    state = run_agent(query)
    
    if state.error:
        print(f"\nExecution failed: {state.error}")
        return
        
    # Print the Executive Briefing
    print("\n" + "=" * 60)
    print(state.briefing)
    print("=" * 60)
    
    # Interactive QA Loop
    print("\n[QA Loop Active] Ask questions about this paper (type 'exit' or 'quit' to stop):")
    while True:
        try:
            user_question = input("\nYour Question: ").strip()
            if user_question.lower() in ["exit", "quit"]:
                print("Exiting QA session. Goodbye!")
                break
            if not user_question:
                continue
                
            print("\nSearching paper context and generating response...")
            ans = answer_question(state, user_question)
            print(f"\nAnswer:\n{ans}\n")
            print("-" * 60)
        except KeyboardInterrupt:
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()