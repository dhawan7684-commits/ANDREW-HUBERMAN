import sys
from langchain_core.messages import HumanMessage, AIMessage
from digital_twin_engine import HubermanDigitalTwin

def run_interactive_loop():
    print("Initializing systems and mapping analytical models...")
    try:
        twin = HubermanDigitalTwin()
    except ValueError as e:
        print(e)
        sys.exit(1)

    SESSION_ID = "evaluation_session_alpha"
    chat_history = twin.load_session_history(SESSION_ID)

    print(f"\n========================================================")
    print(f"🧠 Digital Twin Session Loaded: '{SESSION_ID}'")
    if chat_history:
        print(f"🔄 Recovered {len(chat_history)} dialogue context units from persistent storage.")
    else:
        print("✨ No previous session trace found. Initializing blank context array.")
    print("========================================================\n")
    print("Dr. Andrew Huberman: Welcome. Let's discuss science and science-related tools.")
    print("Type 'exit' or 'quit' to save session traces to storage.\n")

    while True:
        try:
            user_query = input("You: ")
        except (KeyboardInterrupt, EOFError):
            user_query = "exit"

        if user_query.strip().lower() in ['exit', 'quit']:
            twin.save_session_history(SESSION_ID, chat_history)
            print("\n💾 Session context written successfully to persistent storage disk file! System offline.")
            break
            
        if not user_query.strip():
            continue
            
        print("\n[Twin analyzing neural parameters...]")
        
        # Directly calling our updated LCEL pipeline string output engine
        response_text = twin.rag_chain.invoke({
            "input": user_query,
            "chat_history": chat_history
        })
        
        print(f"\n🧠 Dr. Andrew Huberman:\n{response_text}\n" + "-"*60 + "\n")
        
        chat_history.append(HumanMessage(content=user_query))
        chat_history.append(AIMessage(content=response_text))

if __name__ == "__main__":
    run_interactive_loop()