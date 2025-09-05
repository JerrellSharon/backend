from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uuid
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage

# Initialize FastAPI app
app = FastAPI(title="AI Chatbot API", version="1.0.0")

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the chatbot (moved from your app.py)
api_key = "AIzaSyDdK0bKg6e9M-tYTUWUQsPFuR40hF1eTd4"
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=api_key)

# Simple in-memory storage for conversation history
conversation_history: Dict[str, List] = {}

# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    thread_id: str


@app.get("/")
async def root():
    return {"message": "AI Chatbot API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Generate thread_id if not provided
        thread_id = request.thread_id or str(uuid.uuid4())
        
        # Get or create conversation history
        if thread_id not in conversation_history:
            conversation_history[thread_id] = []
        
        # Add user message to history
        conversation_history[thread_id].append(HumanMessage(content=request.message))
        
        # Get response from chatbot with full conversation history
        response = model.invoke(conversation_history[thread_id])
        
        # Add AI response to history
        conversation_history[thread_id].append(response)
        
        return ChatResponse(
            message=response.content,
            thread_id=thread_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/threads/{thread_id}/history")
async def get_chat_history(thread_id: str):
    """Get chat history for a specific thread"""
    try:
        if thread_id not in conversation_history:
            return {"messages": []}
        
        messages = []
        for msg in conversation_history[thread_id]:
            if hasattr(msg, 'content'):
                role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
                messages.append({
                    "role": role,
                    "content": msg.content
                })
        
        return {"messages": messages}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
