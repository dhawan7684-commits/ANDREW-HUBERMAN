import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# Initialize and capture environment/security controls
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

# Operational configuration variables
LLM_MODEL = "gemini-2.5-flash"
PROFILE_FILE_PATH = "user_profile.txt"

class HubermanDigitalTwin:
    def __init__(self):
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("API Key missing! Ensure GOOGLE_API_KEY is configured inside your .env file.")

        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL, 
            temperature=0.3
        )
        
        # Core conversational brain
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
            "behavioral insight related to the topic.\n\n"
            "[CONCISENESS CONSTRAINT]\n"
            "Deliver an organic, conversational insight or biological mechanism matching your persona. "
            "Always respond in a highly concise, punchy manner (maximum 3-4 sentences)."
        )
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        self.retriever = DummyRetriever() 
        self.rag_chain = self.prompt_template | self.llm | StrOutputParser()

    def extract_and_save_profile_facts(self, user_input):
        """Scans user input broadly for ANY personal facts (name, habits, preferences) and logs them."""
        extraction_prompt = (
            f"Analyze this user message: '{user_input}'\n\n"
            "Extract ANY personal identity detail or data point mentioned by the user. This includes:\n"
            "- Their name (e.g., 'Name: John Doe')\n"
            "- Lifestyle habits, sleep metrics, screen time, or daily routines\n"
            "- Dietary preferences, favorite foods, or restrictions\n"
            "- Physical limitations, workout structures, or fitness goals\n\n"
            "Format your extraction as a short bullet point written in the third person starting with an asterisk (e.g., '* Name: Laksh Dhawan'). "
            "If the message contains absolutely no personal user data, facts, or identity traits, reply ONLY with 'NONE'. "
            "Do not include any conversational introduction, justification, or extra text."
        )
        
        try:
            extracted_fact = self.llm.invoke(extraction_prompt).content.strip()
            
            # Make sure we got a valid fact and not a 'NONE' fallback response
            if extracted_fact and "NONE" not in extracted_fact.upper():
                existing_facts = []
                if os.path.exists(PROFILE_FILE_PATH):
                    with open(PROFILE_FILE_PATH, "r", encoding="utf-8") as f:
                        existing_facts = f.read().splitlines()
                
                # Append to file if it hasn't been saved yet
                if extracted_fact not in existing_facts:
                    with open(PROFILE_FILE_PATH, "a", encoding="utf-8") as f:
                        f.write(f"{extracted_fact}\n")
        except Exception:
            pass  # Failsafe background execution execution

class DummyRetriever:
    def __init__(self):
        self.search_kwargs = {"k": 5}