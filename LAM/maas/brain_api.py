#!/usr/bin/env python3
"""
🧠 BRAIN API - Connect Your Personal Brain to Everything

This API lets you:
1. Query your brain from ChatGPT/Claude
2. Add documents, emails, files
3. Get context for any LLM
4. Search across all your data

Start: uvicorn brain_api:app --host 0.0.0.0 --port 5000

Then use:
- http://localhost:5000/docs (API documentation)
- http://localhost:5000/ask (ask questions)
- http://localhost:5000/teach (add knowledge)
- http://localhost:5000/context (get context for LLMs)
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import sys

# Add maas to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autonomous_brain import AutonomousBrain

# ============================================================
# CONFIGURATION
# ============================================================

BRAIN_FILE = os.environ.get("BRAIN_FILE", "willie.said")
AUTO_SAVE = True

# ============================================================
# LOAD BRAIN
# ============================================================

print(f"🧠 Loading brain from {BRAIN_FILE}...")
if os.path.exists(BRAIN_FILE):
    brain = AutonomousBrain.load(BRAIN_FILE)
else:
    print(f"   Creating new brain: {BRAIN_FILE}")
    brain = AutonomousBrain(BRAIN_FILE.replace(".said", ""))

brain.start()

# ============================================================
# API
# ============================================================

app = FastAPI(
    title="🧠 Personal Brain API",
    description="""
Connect your personal brain to ChatGPT, Claude, Cursor, and more!

## Features
- **Ask**: Query your brain for answers
- **Teach**: Add new knowledge
- **Context**: Get relevant context for LLMs
- **Search**: Find anything in your data
- **Learn**: Add documents, emails, files

## Usage with ChatGPT/Claude
1. Before asking ChatGPT, call `/context` with your question
2. Include the context in your prompt
3. After ChatGPT responds, call `/teach` to remember the conversation

## Usage with Cursor
1. Configure Cursor to call `/context` before each request
2. Inject the context into your prompts
3. All discussions are remembered!
    """,
    version="1.0.0",
)

# CORS for browser extensions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# MODELS
# ============================================================

class AskRequest(BaseModel):
    question: str
    
class AskResponse(BaseModel):
    answer: str
    source: str = "brain"
    
class TeachRequest(BaseModel):
    content: str
    category: str = "general"
    source: Optional[str] = None

class ContextRequest(BaseModel):
    query: str
    max_length: int = 2000
    
class ContextResponse(BaseModel):
    context: str
    interests: List[str]
    instruction: str

class SearchRequest(BaseModel):
    query: str
    limit: int = 10

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root():
    """Brain status."""
    return {
        "status": "🧠 Brain is active!",
        "brain_file": BRAIN_FILE,
        "memories": len(brain.memory.text_index),
        "interests": [t for t, _ in brain.get_interests()[:5]],
        "is_learning": brain.learning_loop.is_running,
    }

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Ask your brain a question.
    
    Example: "What was the email about the doctor bill?"
    """
    answer = brain.ask(req.question)
    
    if AUTO_SAVE:
        brain.save(BRAIN_FILE)
    
    return AskResponse(answer=answer)

@app.post("/teach")
def teach(req: TeachRequest):
    """
    Teach your brain something new.
    
    Example: Add a conversation, note, or fact.
    """
    result = brain.teach(req.content, category=req.category)
    
    if AUTO_SAVE:
        brain.save(BRAIN_FILE)
    
    return {"success": True, "memories": len(brain.memory.text_index)}

@app.post("/context", response_model=ContextResponse)
def get_context(req: ContextRequest):
    """
    Get relevant context from your brain for LLM prompts.
    
    Use this before sending to ChatGPT/Claude:
    1. Call /context with your question
    2. Include the returned context in your prompt
    3. The LLM will have your personal context!
    """
    # Get relevant answer from brain
    relevant = brain.ask(req.query)
    
    # Truncate if needed
    if len(relevant) > req.max_length:
        relevant = relevant[:req.max_length] + "..."
    
    # Get current interests
    interests = [topic for topic, _ in brain.get_interests()[:5]]
    
    return ContextResponse(
        context=relevant,
        interests=interests,
        instruction=f"""
[PERSONAL CONTEXT from your brain]
The user has these interests: {', '.join(interests)}

Relevant information from their personal memory:
{relevant}

Use this context to provide a personalized response.
        """.strip()
    )

@app.post("/search")
def search(req: SearchRequest):
    """
    Search across all your data.
    
    Searches: emails, files, documents, notes, conversations.
    """
    results = []
    
    # Search text store
    for entry in brain.memory.text_index:
        text_id = entry['id']
        full_text = brain.memory.text_store.get(text_id, "")
        
        if req.query.lower() in full_text.lower():
            results.append({
                'preview': full_text[:200],
                'source': entry.get('source', 'unknown'),
                'timestamp': entry.get('timestamp', ''),
            })
    
    return {"query": req.query, "results": results[:req.limit]}

@app.post("/learn-document")
async def learn_document(file: UploadFile = File(...)):
    """
    Upload and learn from a document.
    
    Supports: PDF, TXT, MD, etc.
    """
    # Save uploaded file temporarily
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Learn from document
    try:
        result = brain.learn_document(temp_path)
        
        if AUTO_SAVE:
            brain.save(BRAIN_FILE)
        
        return {"success": True, "result": result}
    finally:
        os.remove(temp_path)

@app.get("/stats")
def stats():
    """Get brain statistics."""
    return brain.stats()

@app.get("/interests")
def interests():
    """Get detected interests."""
    return {"interests": brain.get_interests()}

@app.get("/personality")
def personality():
    """Get brain personality summary."""
    return {"personality": brain.personality()}

@app.post("/save")
def save():
    """Force save brain to disk."""
    brain.save(BRAIN_FILE)
    return {"success": True, "file": BRAIN_FILE}

# ============================================================
# LLM INTEGRATION HELPERS
# ============================================================

@app.post("/chatgpt-context")
def chatgpt_context(req: ContextRequest):
    """
    Get context formatted for ChatGPT.
    
    Copy this into ChatGPT's system prompt or your message.
    """
    context = get_context(req)
    
    return {
        "system_prompt": f"""
You are a personal assistant with access to the user's personal memory.

{context.instruction}

Always consider this context when answering questions.
        """.strip(),
        "context_to_inject": context.context,
    }

@app.post("/cursor-context")
def cursor_context(req: ContextRequest):
    """
    Get context formatted for Cursor.
    
    Use this to inject personal context into Cursor conversations.
    """
    context = get_context(req)
    
    return {
        "context": f"""
[Personal Brain Context]
Previous relevant discussions and knowledge:
{context.context}

User's current interests: {', '.join(context.interests)}
        """.strip()
    }

@app.post("/remember-conversation")
def remember_conversation(messages: List[dict]):
    """
    Remember a conversation from ChatGPT/Claude/Cursor.
    
    Call this after each conversation to remember it.
    """
    conversation = "\n".join([
        f"{m.get('role', 'unknown')}: {m.get('content', '')}"
        for m in messages
    ])
    
    brain.teach(conversation, category="conversation")
    
    if AUTO_SAVE:
        brain.save(BRAIN_FILE)
    
    return {"success": True, "remembered": len(messages), "messages": "saved"}

# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup():
    print("=" * 60)
    print("🧠 PERSONAL BRAIN API STARTED")
    print("=" * 60)
    print(f"   Brain file: {BRAIN_FILE}")
    print(f"   Memories: {len(brain.memory.text_index)}")
    print(f"   Auto-save: {AUTO_SAVE}")
    print()
    print("   📖 API Docs: http://localhost:5000/docs")
    print("   🔍 Ask: POST http://localhost:5000/ask")
    print("   📝 Teach: POST http://localhost:5000/teach")
    print("   🎯 Context: POST http://localhost:5000/context")
    print("=" * 60)

@app.on_event("shutdown")
def shutdown():
    brain.stop()
    brain.save(BRAIN_FILE)
    print("🧠 Brain saved and stopped")

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)









