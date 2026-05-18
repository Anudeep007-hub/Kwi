import os
import numpy as np
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from supabase import create_client, Client 
from groq import Groq
from sentence_transformers import SentenceTransformer 
import requests
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. Initialization & Clients
# ==========================================
app = FastAPI(title="KWI Real-Time RAG Backend")

# FIXED: Changed back to "*" so your HTML file on port 5500 can connect!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase (Use Service Role key for backend operations)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") 
GROQ_API_KEY = os.environ.get("ROQ_API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Groq for Sub-1.5s LLM Inference
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize local embedding model (Runs in RAM, extremely fast, 384 dimensions)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ==========================================
# 2. Pydantic Models (Data Validation)
# ==========================================
class ChunkPayload(BaseModel):
    text: str

# FIXED: Added filename here so the backend knows what to call the document
class IngestRequest(BaseModel):
    file_id: str
    filename: str 
    file_hash: str
    chunks: List[ChunkPayload]

class AskRequest(BaseModel):
    query: str
    selected_file_ids: List[str]
    
class CheckFileRequest(BaseModel):
    file_hash: str

# ==========================================
# 3. Helper: Binary Quantization
# ==========================================
def float_to_binary_string(vector: np.ndarray) -> str:
    """Converts a float32 vector into a bit string ('1010...') for pgvector bit types."""
    binary_vector = np.where(vector > 0, 1, 0)
    return "".join(binary_vector.astype(str))

# ==========================================
# 4. API Endpoints
# ========================================== 

@app.post("/check-file")
async def check_file(
    request: CheckFileRequest, 
    user_id: str = Header(..., description="Simulated JWT/User ID for testing")
):
    """Checks if the exact file has already been uploaded by this user."""
    response = supabase.table("files").select("file_id, filename").eq("user_id", user_id).eq("file_hash", request.file_hash).execute()
    
    if response.data:
        # File exists! Return the existing data.
        return {"exists": True, "file_id": response.data[0]["file_id"], "filename": response.data[0]["filename"]}
    
    return {"exists": False}

@app.post("/ingest")
async def ingest_document(
    request: IngestRequest, 
    user_id: str = Header(..., description="Simulated JWT/User ID for testing")
):
    """
    Receives extracted text chunks from the browser, embeds them, 
    binarizes them, and stores them in PostgreSQL.
    """
    # 1. The Bouncer: Check the 10-PDF Limit
    response = supabase.table("chunks").select("file_id", count="exact").eq("user_id", user_id).execute()
    
    unique_files = set(item['file_id'] for item in response.data)
    if len(unique_files) >= 10 and request.file_id not in unique_files:
        raise HTTPException(status_code=403, detail="Free tier limit reached: Maximum 10 PDFs allowed.")

    # ---------------------------------------------------------
    # FIXED: Insert into the 'files' table FIRST to satisfy the Foreign Key
    # ---------------------------------------------------------
    try:
        supabase.table("files").insert({
            "file_id": request.file_id,
            "user_id": user_id,
            "filename": request.filename,
            "file_hash": request.file_hash
        }).execute()
    except Exception as e:
        print(f"File registration note: {e}")

    # 2. Process chunks
    texts = [chunk.text for chunk in request.chunks]
    
    # Generate float embeddings in one batch (Fast)
    float_embeddings = embedding_model.encode(texts)
    
    # 3. Apply Binary Quantization and Prepare Insert Payload
    insert_data = []
    for i, text in enumerate(texts):
        bit_string = float_to_binary_string(float_embeddings[i])
        insert_data.append({
            "file_id": request.file_id,
            "user_id": user_id,
            "content": text,
            "binary_embedding": bit_string
        })
    
    # 4. Bulk Insert into Supabase
    supabase.table("chunks").insert(insert_data).execute()
    
    return {"status": "success", "inserted_chunks": len(insert_data)}


@app.post("/ask")
async def ask_question(
    request: AskRequest, 
    user_id: str = Header(..., description="Simulated JWT/User ID for testing")
):
    """
    Takes the user query and selected PDFs, performs a Hamming Distance search,
    and streams the context to Groq (Llama-3) for a blazing fast answer.
    """
    # 1. Embed and binarize the query
    query_float_embedding = embedding_model.encode(request.query)
    query_bit_string = float_to_binary_string(query_float_embedding)

    
    # 2. Fast Retrieval using PostgreSQL RPC (Hamming Distance + Metadata Filtering)
    search_response = supabase.rpc(
        "match_binary_chunks",
        {
            "query_embedding": query_bit_string,
            "match_count": 5,  # Top-K strict limit for latency
            "user_uid": user_id,
            "selected_file_ids": request.selected_file_ids # FIXED: Matched exact SQL argument name
        }
    ).execute()

    retrieved_chunks = search_response.data
    
    if not retrieved_chunks:
        return {"answer": "No relevant context found in the selected documents."}

    # 3. Construct the LLM Prompt
    context_text = "\n\n".join([chunk["content"] for chunk in retrieved_chunks])
    
    system_prompt = (
        "You only have access to a small portion of the document. If the user asks you to count items, summarize the entire document,"
        "or answer global questions, politely state that you can only answer specific factual questions based on the retrieved snippets."
    )

    # 4. Sub-1.5s Generation using Groq Llama-3
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {request.query}"}
        ],
        model=MODEL_NAME, 
        max_tokens=250,        
        temperature=0.1
    )

    return {
        "answer": chat_completion.choices[0].message.content,
        "retrieved_chunks_used": len(retrieved_chunks)
    } 

    
@app.get("/")
def test_app():
    return {"message": "I'm alive"}