import streamlit as st
from digital_twin_engine import HubermanDigitalTwin  # Imports your existing class

# 1. Page Configuration & Styling
st.set_page_config(
    page_title="Huberman Digital Twin",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Dr. Andrew Huberman Digital Twin")
st.caption("Discussing science, underlying biological mechanisms, and human optimization.")
st.markdown("---")

# 2. Initialize the Backend Engine (Cached so it only runs once)
@st.cache_resource
def init_twin():
    twin = HubermanDigitalTwin()
    if hasattr(twin, 'retriever') and hasattr(twin.retriever, 'search_kwargs'):
        twin.retriever.search_kwargs["k"] = 5
    return twin

with st.spinner("Loading neural parameters and vector databases..."):
    twin_engine = init_twin()

# 3. Handle Short-Term Session Memory (Conversational Welcome Message)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome. Let's discuss science and science-related tools. What's on your mind today regarding human biology or behavior?"}
    ]

# 4. Render Conversation History on Screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Handle User Input & RAG Execution
if user_query := st.chat_input("Message your Digital Twin..."):
    
    # Display the user's message immediately
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # Generate the Twin's Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("Processing neural data..."):
            try:
                from langchain_core.messages import HumanMessage, AIMessage
                
                # Re-map memory to proper LangChain Message Objects
                formatted_history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        formatted_history.append(HumanMessage(content=msg["content"]))
                    else:
                        formatted_history.append(AIMessage(content=msg["content"]))
                
                # Execute pipeline invocation
                raw_response = twin_engine.rag_chain.invoke({
                    "input": user_query, 
                    "chat_history": formatted_history
                })
                
                response_text = raw_response if isinstance(raw_response, str) else raw_response.get("text", str(raw_response))
                
            except Exception as e:
                response_text = f"An execution error occurred: {str(e)}"
            
            # Print response to screen
            response_placeholder.markdown(response_text)
            
    # Save response to history
    st.session_state.messages.append({"role": "assistant", "content": response_text})