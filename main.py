import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langsmith import traceable
import uvicorn
from langchain.chat_models import init_chat_model
from app.agent import ProductionAgent
from app.cache import ResponseCache
from app.models import ChatRequest, ChatResponse
from fastapi.middleware.cors import CORSMiddleware


cache = ResponseCache(ttl_seconds=3000)
print(cache.stats)
app = FastAPI(title="Agent API")
embeddings_model = OllamaEmbeddings(model="qwen3-embedding:4b")

llm = init_chat_model(model="qwen3.5:9b", temperature=0.2, model_provider="ollama")
ollama_instance = ProductionAgent(llm=llm)
vectordb = Chroma(
    persist_directory="RAG_utils/langchain_kb", embedding_function=embeddings_model
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
@traceable(name="chat_endpoint")
async def chat(request: Request, body: ChatRequest):
    query = body.message

    start = time.time()
    cached_response = cache.get(query)
    
    if cached_response is not None:
        print("Cache hit")

        return ChatResponse(
            response=cached_response,
            thread_id=body.thread_id,
            model_used="cache",
            cached=True,
            processing_time_ms=0,
        )
    
    initial_state = {
        "query": query,
        "rewritten_query": "",
        "documents": [],
        "generation": "",
        "relevance_score": 0.0,
        "retry_count": 0,
        "max_retries": 2,
        "vectordb": vectordb,  # Pass vectorstore via state
    }
    
    
    #ollama_app = build_agentic_rag_graph()

    
    result = ollama_instance.build_agentic_rag_graph().invoke(initial_state)
    

    response_text = result["generation"]

    cache.set(body.message, response_text)

    input_tokens = int(len(body.message.split()) * 1.3)
    output_tokens = int(len(response_text.split()) * 1.3)
    end = time.time()

    return ChatResponse(
        response=response_text,
        thread_id=body.thread_id,
        model_used="qwen:9b",
        cached=False,
        processing_time_ms=round(((end - start) * 1000), 2),
    )


uvicorn.run(app, host="0.0.0.0", port=5000)