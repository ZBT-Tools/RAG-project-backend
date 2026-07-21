import re
import fitz  
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

embeddings_model = OllamaEmbeddings(model="qwen3-embedding:4b")

def construct_vectorstore():
    
    text_splitter = RecursiveCharacterTextSplitter(
      
        chunk_size=2000,      
        chunk_overlap=150     
    )

    # 3. Read the PDF and track headings sequentially
    # Regex matching MDPI structural headers (e.g., "1. Introduction", "2.1. Aquifer...", "4. Conclusions")
    HEADING_REGEX = r'^(?:\d+\.)+(?:\d+)?\s+[A-Z][a-zA-Z\s,]{3,}'

    doc = fitz.open("pdfs/energies-18-00645.pdf") # Replace with your downloaded file path

    sections = []
    current_heading = "Abstract"
    current_buffer = ""

    for page in doc:
        text_lines = page.get_text("text").split('\n')
        
        for line in text_lines:
            line_clean = line.strip()            
            # Check if line matches an academic heading pattern
            if re.match(HEADING_REGEX, line_clean):
                # Save the gathered buffer to its corresponding heading before shifting states
                if current_buffer.strip():
                    sections.append({"heading": current_heading, "text": current_buffer})
                    current_buffer = ""
                current_heading = line_clean
            else:
                current_buffer += line + "\n"

  
    if current_buffer.strip():
        sections.append({"heading": current_heading, "text": current_buffer})

  
    token_chunks = []
    for section in sections:
        heading_title = section["heading"]
        section_text = section["text"]
        
        document_chunk = Document(page_content=section_text, metadata={"heading": heading_title})
        token_chunks.extend(text_splitter.split_documents([document_chunk]))
    



    vector_store = Chroma.from_documents(
        documents=token_chunks, embedding=embeddings_model, persist_directory="langchain_kb")

    
    return vector_store


construct_vectorstore()