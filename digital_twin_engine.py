import os
import re
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-2-preview"
CHROMA_DB_DIR = "./chroma_db"
PROFILE_FILE_PATH = "user_profile.txt"

class HubermanDigitalTwin:
    def __init__(self):
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("Key missing! Ensure GOOGLE_API_KEY is configured inside your .env file.")

        self.llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.3)
        self.embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        
        if os.path.exists(CHROMA_DB_DIR):
            self.vector_store = Chroma(
                persist_directory=CHROMA_DB_DIR, 
                embedding_function=self.embeddings
            )
        else:
            self.vector_store = None
            print("Warning: Combined database not found. Run ingest_dataset.py first.")

        self.system_prompt = (
            "You are a conversational Digital Twin of Dr. Andrew Huberman, specializing in neuroscience "
            "and human optimization. You have access to your historical YouTube video transcripts and your written Newsletters.\n\n"
            "[TIMELINE AWARENESS]\n"
            "You are supplied with a dynamic timestamp tracking the user's local system time on every turn. "
            "Cross-reference this clock data with any lifestyle habits or choices they mention.\n\n"
            "[MEMORY & CONTEXT INTEGRATION]\n"
            "Analyze the active 'chat_history' array meticulously. Treat any personal information shared by the user as absolute fact.\n\n"
            "[CRITICAL RULE FOR DATA GAPS]\n"
            "If a user asks about a personal preference or data missing from your context, you must NEVER "
            "state 'I do not have access to that information' or state you are an AI. Gracefully pivot to a general biological mechanism.\n\n"
            "[CONCISENESS CONSTRAINT]\n"
            "Always respond in a highly concise, punchy manner (maximum 3-4 sentences)."
        )
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}\n\n[Retrieved Mixed Database Context]:\n{context}")
        ])
        
        self.rag_chain = self.prompt_template | self.llm | StrOutputParser()

    def determine_search_filters(self, user_query):
        """Analyzes query parameters to isolate year or source specific instructions."""
        analysis_prompt = (
            f"Analyze this raw user question: '{user_query}'\n\n"
            "Determine if the user is asking specifically about a certain year or database source type.\n"
            "Output a valid JSON dictionary representing database metadata constraints. Valid parameters are:\n"
            "- 'year': integer type\n"
            "- 'source_type': string type (MUST be either 'youtube' or 'newsletter')\n"
            "- 'category': string type ('guest_episode' or 'essentials')\n\n"
            "If the user says words like 'email', 'newsletter', 'article', or 'written', map 'source_type' to 'newsletter'.\n"
            "If the user says 'video', 'youtube', 'channel', or 'watch', map 'source_type' to 'youtube'.\n"
            "If no specific filters are detected, return ONLY '{}'. Do not add markdown backticks."
        )
        
        try:
            raw_output = self.llm.invoke(analysis_prompt).content.strip()
            # Clean out any accidental markdown code blocks without breaking string rules
            cleaned_json = raw_output.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_json) if cleaned_json else {}
        except Exception:
            return {}

    def query_twin_with_rag(self, user_query, formatted_history, timestamp_str):
        """Queries the integrated database utilizing dynamic multi-source filtering options."""
        context_string = "No matching cross-platform document context found."
        
        if self.vector_store:
            active_filters = self.determine_search_filters(user_query)
            
            # Requesting 5 top document matches from the unified dataset
            search_kwargs = {"k": 5}
            if active_filters:
                search_kwargs["filter"] = active_filters
                
            try:
                retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
                matching_docs = retriever.invoke(user_query)
                if matching_docs:
                    context_string = ""
                    for d in matching_docs:
                        source_tag = d.metadata.get('source_type', 'unknown').upper()
                        context_string += f"\n[{source_tag} SOURCE]:\n{d.page_content}\n---"
            except Exception as e:
                context_string = f"Database parsing error: {str(e)}"

        concise_input = (
            f"[CURRENT TIME CONTEXT: {timestamp_str}]\n"
            f"User Message: {user_query}\n\n"
            "[Constraint: Keep response limited to 3-4 sentences total.]"
        )
        
        return self.rag_chain.invoke({
            "input": concise_input,
            "chat_history": formatted_history,
            "context": context_string
        })

    def extract_and_save_profile_facts(self, user_input, current_time_str):
        """Captures traits from user input to save in the dashboard file."""
        extraction_prompt = (
            f"Analyze this user message: '{user_input}'\n"
            f"Contextual Current Time: {current_time_str}\n\n"
            "Extract ANY personal identity detail mentioned. Format your extraction as a short "
            "bullet point written in the third person starting with an asterisk (e.g., '* Name: Laksh Dhawan'). "
            "If the message contains no personal profile details, reply ONLY with 'NONE'."
        )
        try:
            extracted_fact = self.llm.invoke(extraction_prompt).content.strip()
            if extracted_fact and "NONE" not in extracted_fact.upper():
                existing_facts = []
                if os.path.exists(PROFILE_FILE_PATH):
                    with open(PROFILE_FILE_PATH, "r", encoding="utf-8") as f:
                        existing_facts = f.read().splitlines()
                if extracted_fact not in existing_facts:
                    with open(PROFILE_FILE_PATH, "a", encoding="utf-8") as f:
                        f.write(f"{extracted_fact}\n")
        except Exception:
            pass