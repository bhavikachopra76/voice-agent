import os
import asyncio
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import edge_tts
import pygame

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not CEREBRAS_API_KEY:
    raise ValueError("SUPABASE_URL, SUPABASE_KEY, and CEREBRAS_API_KEY environment variables are required.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
cerebras_client = OpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=CEREBRAS_API_KEY
)

print("Loading embedding model on CPU...")
embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
print("Model loaded successfully!")

pygame.mixer.init()

async def speak_text(text):
    """Generates premium neural speech and plays it."""
    print("\n[Generating high-quality audio with Edge-TTS...]")

    voice = "en-US-AriaNeural"
    output_file = "response.mp3"
    
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)
    
    print("[Speaking...]")

    pygame.mixer.music.load(output_file)
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def retrieve_relevant_chunks(user_query, match_count=2):
    query_vector = embedding_model.encode(user_query).tolist()
    response = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_vector,
            "match_threshold": 0.4,
            "match_count": match_count
        }
    ).execute()
    return response.data

def generate_answer(user_query, retrieved_chunks):
    context_text = "\n\n---\n\n".join([chunk["content"] for chunk in retrieved_chunks])
    
    system_prompt = (
        "You are a helpful voice assistant. Answer the user's question using ONLY the provided text context. "
        "If the answer cannot be found in the context, say 'I am sorry, but that information is not in the manual.' "
        "Do not invent facts or use outside knowledge. Keep your response clear, conversational, short, and direct."
    )
    
    user_prompt = f"Context from the manual:\n{context_text}\n\nUser Question: {user_query}"
    
    print("Sending context to Cerebras AI for ultra-fast generation...")
    
    completion = cerebras_client.chat.completions.create(
        model="zai-glm-4.7", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )
    
    return completion.choices[0].message.content

async def main():
    test_question = "What should I do if the unit begins drifting to the left during initial spin-up?"
    print(f"\nUser Question: '{test_question}'")

    matched_data = retrieve_relevant_chunks(test_question)
    print(f"Retrieved {len(matched_data)} relevant chunks from Supabase.")
    
    ai_response = generate_answer(test_question, matched_data)
    print("\n--- Cerebras AI Response ---")
    print(ai_response)
    
    await speak_text(ai_response)

if __name__ == "__main__":
    asyncio.run(main())