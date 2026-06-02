import os
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from digital_twin_engine import HubermanDigitalTwin

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
PROFILE_FILE_PATH = "user_profile.txt"

# Page Configuration
st.set_page_config(
    page_title="Huberman Digital Twin",
    page_icon="🧠",
    layout="wide"
)

# ─── AUTOMATIC SESSION CLEANUP ───
if "messages" not in st.session_state:
    if os.path.exists(PROFILE_FILE_PATH):
        try:
            os.remove(PROFILE_FILE_PATH)
        except Exception:
            pass

    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome. Let's discuss science and science-related tools. What's on your mind today regarding human biology or behavior?", "time": ""}
    ]

# ─── MASTER LAYOUT DIVISION ───
master_left_col, master_right_col = st.columns([1, 3], gap="large")

# 📊 PANEL 1: LEFT SIDE (Dynamic User Profile Panel)
with master_left_col:
    st.markdown("### 🗂️ User Insights & Memory")
    st.caption("Key facts extracted dynamically from your current session:")
    
    if os.path.exists(PROFILE_FILE_PATH) and os.path.getsize(PROFILE_FILE_PATH) > 0:
        with open(PROFILE_FILE_PATH, "r", encoding="utf-8") as f:
            profile_facts = f.read()
        st.markdown(profile_facts)
    else:
        st.info("Start chatting! As you share habits or preferences, they will appear right here.")
        
    if st.button("🗑️ Reset Chat & Wipe Memory"):
        if os.path.exists(PROFILE_FILE_PATH):
            try:
                os.remove(PROFILE_FILE_PATH)
            except Exception:
                pass
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# 🧠 PANEL 2: RIGHT SIDE (Main Core Application Interface)
with master_right_col:
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

    # Render Conversation Feed with Time Labels
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("time"):
                st.caption(f"🕒 Time Sent: {message['time']}")
            st.markdown(message["content"])

    # Handle User Query Input
    if user_query := st.chat_input("Message your Digital Twin..."):
        
        # Capture precise real-world execution timestamp
        current_now = datetime.now()
        timestamp_str = current_now.strftime("%I:%M %p (%A)") # e.g. "08:30 AM (Monday)"
        
        # Display human input with time
        st.session_state.messages.append({"role": "user", "content": user_query, "time": timestamp_str})
        with st.chat_message("user"):
            st.caption(f"🕒 Time Sent: {timestamp_str}")
            st.markdown(user_query)
            
        # Trigger timeline-aware background extraction
        twin_engine.extract_and_save_profile_facts(user_query, timestamp_str)
            
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
                    
                    # Wrap query inside a contextual timeline tracking array dynamically
                    concise_query = (
                        f"[CURRENT TIME CONTEXT: {timestamp_str}]\n"
                        f"User Message: {user_query}\n\n"
                        "[Constraint: Provide a highly concise, punchy response (max 3-4 sentences). "
                        "Leverage the current time context naturally if the query implies a time-sensitive biological routine.]"
                    )
                    
                    raw_response = twin_engine.rag_chain.invoke({
                        "input": concise_query, 
                        "chat_history": formatted_history
                    })
                    
                    response_text = raw_response if isinstance(raw_response, str) else raw_response.get("text", str(raw_response))
                    
                except Exception as e:
                    response_text = f"An execution error occurred: {str(e)}"
                
                response_placeholder.markdown(response_text)
                
        st.session_state.messages.append({"role": "assistant", "content": response_text, "time": ""})
        st.rerun()