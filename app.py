import os
import streamlit as st
from dotenv import load_dotenv
from digital_twin_engine import HubermanDigitalTwin

# Load configuration values for local runtime
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

# 1. Page Configuration & Styling
st.set_page_config(
    page_title="Huberman Digital Twin",
    page_icon="🧠",
    layout="centered"
)

# Polished 2-Column Layout: Left (Huberman Profile) | Right (Title Text Stack)
col_left, col_text = st.columns([1, 5])

with col_left:
    try:
        # Dynamically updates your profile image
        st.image("left_image.jpg", use_container_width=True)
    except Exception:
        st.caption("🖼️ [Huberman]")

with col_text:
    st.title("Dr. Andrew Huberman Digital Twin")
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
    try:
        twin_engine = init_twin()
    except Exception as initialization_error:
        st.error(f"Initialization Failed: {str(initialization_error)}")
        st.stop()

# 3. Handle Short-Term Session Memory (Conversational Welcome Message)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome. Let's discuss science and science-related tools. What's on your mind today regarding human biology or behavior?"}
    ]

# 4. Render Conversation History on Screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Handle User Input & Conversational Execution
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
                
                # Re-map short term memory directly into LangChain Base Objects
                formatted_history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        formatted_history.append(HumanMessage(content=msg["content"]))
                    else:
                        formatted_history.append(AIMessage(content=msg["content"]))
                
                # Inject a strict formatting wrapper onto the query dynamically
                concise_query = (
                    f"{user_query}\n\n"
                    "[Constraint: Provide a highly concise, punchy response (max 3-4 sentences). "
                    "Deliver an organic, conversational insight or biological mechanism matching your persona. "
                    "Do not provide massive breakdowns, long preambles, or unprompted multi-step lists unless explicitly asked.]"
                )
                
                # Execute pipeline execution
                raw_response = twin_engine.rag_chain.invoke({
                    "input": concise_query, 
                    "chat_history": formatted_history
                })
                
                response_text = raw_response if isinstance(raw_response, str) else raw_response.get("text", str(raw_response))
                
            except Exception as e:
                response_text = f"An execution error occurred: {str(e)}"
            
            # Print response output to screen
            response_placeholder.markdown(response_text)
            
    # Save response cleanly back into state history
    st.session_state.messages.append({"role": "assistant", "content": response_text})