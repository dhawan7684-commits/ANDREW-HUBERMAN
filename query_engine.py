import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Ensure API Key
if not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = "YOUR_ACTUAL_GEMINI_API_KEY_HERE"

DB_DIR = os.path.join(".", "dataset", "chroma_db")

def ask_huberman_twin(user_question: str):
    # 1. Initialize our setup
    embeddings = GoogleGenAIEmbeddings(model="models/text-embedding-004")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    
    # Connect back to the saved database
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # 2. Configure Retreiver Engine (Pull top 4 most relevant chunks from YouTube/Newsletters)
    retriever = db.as_retriever(search_kwargs={"k": 4})
    
    # 3. Create Persona Prompt Structure
    system_prompt = (
        "You are the AI Digital Twin of Dr. Andrew Huberman.\n"
        "Answer the user's questions with scientific precision, high clarity, and a supportive tone based "
        "only on your provided YouTube transcripts and Newsletters contexts below.\n"
        "If you do not know the answer from the context, state that you don't find it in your current log files.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 4. Tie Retrieval and Generation Together (Step 4: Retrieving)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # Fire query
    response = rag_chain.invoke({"input": user_question})
    
    print("\n--- 🧠 HUBERMAN TWIN RESPONSE ---")
    print(response["answer"])
    print("\n--- 🔬 SOURCES USED ---")
    for doc in response["context"]:
        print(f"📍 Source: {doc.metadata.get('source')} | Type: {doc.metadata.get('type')}")

if __name__ == "__main__":
    # Test query
    query = input("Ask your Digital Twin a question: ")
    ask_huberman_twin(query)