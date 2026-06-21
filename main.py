import os
from dotenv import load_dotenv
import fitz  
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def extract_and_chunk_pdf(pdf_path):
    print(f"Opening PDF: {pdf_path}...")
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text += str(page.get_text())
        
    print(f"Extraction complete. Total characters extracted: {len(full_text)}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(full_text)
    print(f"Splitting complete. Created {len(chunks)} text chunks.")
    
    return chunks

if __name__ == "__main__":
    pdf_filename = "rag_test_document.pdf" 
    
    if os.path.exists(pdf_filename):
        document_chunks = extract_and_chunk_pdf(pdf_filename)
        print("\n--- Sample First Chunk ---")
        print(document_chunks[0])
    else:
        print(f"Could not find '{pdf_filename}' in this directory. Please place your PDF here.")