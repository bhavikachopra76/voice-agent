import os
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables are required.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Initializing local embedding model (bge-small-en-v1.5)...")
embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
print("Model loaded successfully!")

def upload_chunks_to_supabase(chunks):
    print(f"\nStarting embedding and upload process for {len(chunks)} chunks...")
    
    for i, chunk in enumerate(chunks):
        print(f"-> Generating embedding for chunk {i+1}/{len(chunks)}...")
        embedding = embedding_model.encode(chunk).tolist() 
        data = {
            "content": chunk,
            "embedding": embedding
        }
        
        try:
            response = supabase.table("document_chunks").insert(data).execute()
            print(f"   Successfully uploaded chunk {i+1} to Supabase.")
        except Exception as e:
            print(f"   Error uploading chunk {i+1}: {e}")
            return False
            
    print("\n All chunks successfully embedded and stored in Supabase!")
    return True

if __name__ == "__main__":
    from main import extract_and_chunk_pdf
    
    pdf_filename = "rag_test_document.pdf"
    
    if os.path.exists(pdf_filename):
        chunks = extract_and_chunk_pdf(pdf_filename)
        
        upload_chunks_to_supabase(chunks)
    else:
        print(f"Error: Could not find '{pdf_filename}'. Please ensure it is in this folder.")