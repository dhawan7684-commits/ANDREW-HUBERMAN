import os
import time
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Ensure your API key is configured
if not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = "YOUR_ACTUAL_GEMINI_API_KEY_HERE"

# --- CONFIGURATION PATHS ---
DB_DIR = os.path.join(".", "dataset", "chroma_db")
YOUTUBE_DIR = r"c:\Users\Laksh Dhawan\Desktop\ANDREW HUBERMAN DIGITAL TWIN\dataset\youtube_history"
NEWSLETTER_DIR = r"c:\Users\Laksh Dhawan\Desktop\ANDREW HUBERMAN DIGITAL TWIN\dataset\newsletters"

def load_and_chunk_data():
    all_chunks = []
    
    # Setup our text splitter (Step 1: Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    
    # --- 1. PROCESS YOUTUBE TRANSCRIPTS ---
    yt_files = glob.glob(os.path.join(YOUTUBE_DIR, "*.txt"))
    print(f"📋 Found {len(yt_files)} YouTube transcripts.")
    for file_path in yt_files:
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            chunks = text_splitter.split_documents(docs)
            for chunk in chunks:
                chunk.metadata["source"] = os.path.basename(file_path)
                chunk.metadata["type"] = "youtube"
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"⚠️ Error chunking YouTube video {os.path.basename(file_path)}: {e}")

    # --- 2. PROCESS NEWSLETTERS ---
    if os.path.exists(NEWSLETTER_DIR):
        nl_files = glob.glob(os.path.join(NEWSLETTER_DIR, "*.txt"))
        print(f"📋 Found {len(nl_files)} Newsletters.")
        for file_path in nl_files:
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                chunks = text_splitter.split_documents(docs)
                for chunk in chunks:
                    chunk.metadata["source"] = os.path.basename(file_path)
                    chunk.metadata["type"] = "newsletter"
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"⚠️ Error chunking Newsletter {os.path.basename(file_path)}: {e}")
    else:
        print("ℹ️ Newsletter directory not found. Skipping...")

    return all_chunks

def compile_vector_db():
    print("🚀 Initializing Pipeline...")
    chunks = load_and_chunk_data()
    total_chunks = len(chunks)
    print(f"✅ Step 1 Complete: Total chunks generated: {total_chunks}")
    
    if total_chunks == 0:
        print("❌ No data chunks to process. Exiting.")
        return

    # Step 2: Initialize Gemini Embedding Engine
    print("\n🧠 Step 2: Waking up Gemini Text Embedding Engine...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    print(f"\n💾 Step 3: Saving embeddings into Chroma DB folder at: {DB_DIR}")
    print("⏳ Processing batches with safety delays to avoid API rate blocks...")

    # Upload configuration constraints (Safety valves for large datasets)
    BATCH_SIZE = 50 
    PAUSE_DURATION = 60 

    for start_idx in range(0, total_chunks, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, total_chunks)
        batch_chunks = chunks[start_idx:end_idx]
        
        print(f"   📤 Uploading chunks {start_idx} to {end_idx}...")
        
        if start_idx == 0:
            db = Chroma.from_documents(
                documents=batch_chunks,
                embedding=embeddings,
                persist_directory=DB_DIR
            )
        else:
            db.add_documents(documents=batch_chunks)
            
        if end_idx < total_chunks:
            print(f"   💤 API rate-limit protection active. Sleeping for {PAUSE_DURATION} seconds...")
            time.sleep(PAUSE_DURATION)

    print(f"\n🎉 Success! Database built and locked at: {DB_DIR}")

if __name__ == "__main__":
    compile_vector_db()