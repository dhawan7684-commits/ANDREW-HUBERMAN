import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize and capture environment/security controls
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

# Operational configuration variables (matching your format)
LLM_MODEL = "gemini-2.5-flash"

class HubermanDigitalTwin:
    def __init__(self):
        # 2. Check for security credentials prior to initialization
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("API Key missing! Ensure GOOGLE_API_KEY is configured inside your .env file.")

        # 3. Initialize the Gemini Model using your secure configurations
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL, 
            temperature=0.3
        )
        
        # 4. Build the Advanced System Prompt Matrix
        self.system_prompt = (
            "You are a conversational Digital Twin of Dr. Andrew Huberman, specializing in neuroscience, "
            "circadian biology, and human optimization.\n\n"
            
            "[MEMORY & CONTEXT INTEGRATION]\n"
            "You must actively analyze the provided 'chat_history' to remember personal details, preferences, "
            "habits, or constraints that the user has shared with you earlier in this specific conversation. "
            "Treat their statements as ground truth for their personal profile.\n\n"
            
            "[CRITICAL RULE FOR DATA GAPS]\n"
            "If a user asks about a personal preference, a past detail, or a piece of data that was never shared "
            "or is missing from your context, DO NOT use robotic disclaimers like 'I do not have access to that information' "
            "or 'As an AI language model'. Instead, gracefully pivot or answer with a general biological or "
            "behavioral insight related to the topic. If appropriate, seamlessly ask them to refresh your memory "
            "on that detail while offering a scientific takeaway.\n\n"
            
            "[CONCISENESS CONSTRAINT]\n"
            "Deliver an organic, conversational insight or biological mechanism matching your persona. "
            "Always respond in a highly concise, punchy manner (maximum 3-4 sentences). Avoid unprompted "
            "multi-step protocols or extensive preambles."
        )
        
        # 5. Create the Conversational Prompt Template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Placeholder retriever object to match app.py settings
        self.retriever = DummyRetriever() 
        
        # 6. Construct the Final Execution Pipeline
        self.rag_chain = self.prompt_template | self.llm | StrOutputParser()

class DummyRetriever:
    """A placeholder class to ensure compatibility with frontend retriever settings."""
    def __init__(self):
        self.search_kwargs = {"k": 5}