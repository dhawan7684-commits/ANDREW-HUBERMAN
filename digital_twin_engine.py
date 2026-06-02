import os
import json
import time
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import message_to_dict, messages_from_dict
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import config

class HubermanDigitalTwin:
    def __init__(self):
        # 1. Initialize Gemini Embeddings and LLM
        self.embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)
        self.llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=0.2)
        self.retriever = self._initialize_rag_pipeline()
        
        # 2. Assemble Processing Chains using LCEL
        self.rag_chain = self._build_orchestration_chains()

    def _initialize_rag_pipeline(self):
        if not os.path.exists(config.DATA_DIR) or not os.listdir(config.DATA_DIR):
            print(f"⚠️ Warning: '{config.DATA_DIR}' folder empty. Operating without RAG data context.")
            return None
            
        # Check if DB already exists to avoid redundant embeddings and rate limits
        if os.path.exists(config.CHROMA_DB_DIR) and os.listdir(config.CHROMA_DB_DIR):
            print("🔄 Loading existing Vector Database from disk storage...")
            vectorstore = Chroma(
                persist_directory=config.CHROMA_DB_DIR,
                embedding_function=self.embeddings
            )
            return vectorstore.as_retriever(search_kwargs={"k": 2})

        print("📦 Building fresh vector store database...")
        loader = DirectoryLoader(
            config.DATA_DIR, 
            glob="**/*.txt", 
            loader_cls=TextLoader, 
            loader_kwargs={"encoding": "utf-8"}
        )
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        
        # Initialize an empty Chroma instance
        vectorstore = Chroma(
            persist_directory=config.CHROMA_DB_DIR,
            embedding_function=self.embeddings
        )
        
        # 🚀 FIX: Batch process items with sleep intervals to satisfy free tier limits
        batch_size = 3
        print(f"🧬 Processing {len(splits)} text splits in safe batches...")
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            print(f"   ↳ Progress: Uploading splits {i} to {min(i + batch_size, len(splits))}...")
            vectorstore.add_documents(batch)
            time.sleep(2.5)  # Safe delay between batches to stay under rate limits
            
        return vectorstore.as_retriever(search_kwargs={"k": 5})

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _build_orchestration_chains(self):
        huberman_persona_prompt = (
            "You are a Digital Twin of Dr. Andrew Huberman, professor of neurobiology and ophthalmology "
            "at Stanford School of Medicine.\n"
            "Your objective is to strictly emulate his communication voice, reasoning style, and structural values.\n\n"
            "Tone & Vocabulary Rules:\n"
            "- Provide highly logical, actionable advice structured step-by-step (e.g., 'Phase 1', 'First').\n"
            "- Extensively leverage specific jargon: 'protocols', 'mechanisms', 'neural circuits', 'modality', 'peer-reviewed studies'.\n"
            "- Frequently make sure to emphasize that these biological tools are 'zero-cost'.\n\n"
            "Grounding Rule:\n"
            "Base your answers precisely on the retrieved context below. If you do not know or the context "
            "lacks information, note the bounds of your knowledge gracefully in character.\n\n"
            "Retrieved Context:\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", huberman_persona_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        if self.retriever:
            chain = (
                {
                    "context": lambda x: self._format_docs(self.retriever.invoke(x["input"])),
                    "input": lambda x: x["input"],
                    "chat_history": lambda x: x["chat_history"]
                }
                | qa_prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            chain = qa_prompt | self.llm | StrOutputParser()
            
        return chain

    @staticmethod
    def load_session_history(session_id: str):
        if os.path.exists(config.MEMORY_FILE_PATH):
            with open(config.MEMORY_FILE_PATH, "r") as f:
                try:
                    db = json.load(f)
                    if session_id in db:
                        return messages_from_dict(db[session_id])
                except json.JSONDecodeError:
                    pass
        return []

    @staticmethod
    def save_session_history(session_id: str, message_list):
        db = {}
        if os.path.exists(config.MEMORY_FILE_PATH):
            with open(config.MEMORY_FILE_PATH, "r") as f:
                try:
                    db = json.load(f)
                except json.JSONDecodeError:
                    pass
                    
        db[session_id] = [message_to_dict(msg) for msg in message_list]
        with open(config.MEMORY_FILE_PATH, "w") as f:
            json.dump(db, f, indent=4)