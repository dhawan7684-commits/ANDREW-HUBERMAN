import os
import streamlit as st
from dotenv import load_dotenv
from digital_twin_engine import HubermanDigitalTwin

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
PROFILE_FILE_PATH = "user_profile.txt"

# Page Configuration
st.set_page_config(
    page_title="Huberman Digital Twin",
    page_icon="🧠",
    layout="wide"  # Keeps the wide panel layout
)

# ─── AUTOMATIC SESSION CLEANUP (Wipes memory when chat ends/resets) ───
if "messages" not in st.session_state:
    if os.path.exists(PROFILE_FILE_PATH):
        try:
            os.remove(PROFILE_FILE_PATH)
        except Exception:
            pass # Failsafe if file is temporarily locked by another process

    # Initialize the chat with the conversational greeting
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome. Let's discuss science and science-related tools. What's on your mind today regarding human biology or behavior?"}
    ]

# ─── MASTER LAYOUT DIVISION ───
master_left_col, master_right_col = st.columns([1, 3], gap="large")

# 📊 PANEL 1: LEFT SIDE (Dynamic User Profile Panel)
with master_left_col:
    st.markdown("### 🗂️ User Insights & Memory")
    st.caption("Key facts extracted dynamically from your current session:")
    
    # Read and display facts live from your standalone text file
    if os.path.exists(PROFILE_FILE_PATH) and os.path.getsize(PROFILE_FILE_PATH) > 0:
        with open(PROFILE_FILE_PATH, "r", encoding="utf-8") as f:
            profile_facts = f.read()
        st.markdown(profile_facts)
    else:
        st.info("Start chatting! As you share habits or preferences, they will appear right here.")
        
    # Manual clear button fallback that forces a session state reset
    if st.button("🗑️ Reset Chat & Wipe Memory"):
        if os.path.exists(PROFILE_FILE_PATH):
            try:
                os.remove(PROFILE_FILE_PATH)
            except Exception:
                pass
        # Clear Streamlit conversation state completely
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# 🧠 PANEL 2: RIGHT SIDE (Main Core Application Interface)
with master_right_col:
    # Header Sub-Layout: Left (Portrait Badge) | Right (Title Text Stack)
    header_pic, header_txt = st.columns([1, 5])
    
    with header_pic:
        try:
            st.image("left_image.jpg", use_container_width=True)
        except Exception:
            st.caption("🖼️ [Huberman]")
            
    with header_txt:
        st.title("Dr. Andrew Huberman Digital Twin")
        st.caption("Discussing science, underlying biological mechanisms, and human optimization.")
        
    st.markdown("---")

    # Initialize Backend Engine
    @st.cache_resource
    def init_twin():
        twin = HubermanDigitalTwin()
        if hasattr(twin, 'retriever') and hasattr(twin.retriever, 'search_kwargs'):
            twin.retriever.search_kwargs["k"] = 5
        return twin

    with st.spinner("Loading neural parameters and vector databases..."):
        try:
            twin_engine = init_twin()
        except Exception as err:
            st.error(f"Initialization Failed: {str(err)}")
            st.stop()

    # Render Conversation Feed
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle User Query Input
    if user_query := st.chat_input("Message your Digital Twin..."):
        
        # Display human input
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        # Trigger background extraction BEFORE generating the response
        twin_engine.extract_and_save_profile_facts(user_query)
            
        # Generate and show assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("Processing neural data..."):
                try:
                    from langchain_core.messages import HumanMessage, AIMessage
                    
                    formatted_history = []
                    for msg in st.session_state.messages[:-1]:
                        if msg["role"] == "user":
                            formatted_history.append(HumanMessage(content=msg["content"]))
                        else:
                            formatted_history.append(AIMessage(content=msg["content"]))
                    
                    concise_query = (
                        f"{user_query}\n\n"
                        "[Constraint: Provide a highly concise, punchy response (max 3-4 sentences). "
                        "Deliver an organic, conversational insight or biological mechanism matching your persona.]"
                    )
                    
                    raw_response = twin_engine.rag_chain.invoke({
                        "input": concise_query, 
                        "chat_history": formatted_history
                    })
                    
                    response_text = raw_response if isinstance(raw_response, str) else raw_response.get("text", str(raw_response))
                    
                except Exception as e:
                    response_text = f"An execution error occurred: {str(e)}"
                
                response_placeholder.markdown(response_text)
                
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.rerun()  # Forces a UI refresh to show the newly extracted facts immediately