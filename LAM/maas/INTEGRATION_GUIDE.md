# 🧠 Personal Brain Integration Guide

## The Vision: Your Brain Connected to Everything

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        YOUR PERSONAL BRAIN ECOSYSTEM                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  DATA SOURCES                    .SAID BRAIN                    INTERFACES  │
│  ────────────                    ──────────                     ──────────  │
│                                                                              │
│  📧 Emails ──────────┐                              ┌──────── 🤖 ChatGPT    │
│  📁 Local Files ─────┤                              │                        │
│  🔍 Web Search ──────┤      ┌──────────────┐       ├──────── 💻 Cursor     │
│  📄 Documents ───────┼─────▶│  willie.said │◀──────┤                        │
│  💬 Chat History ────┤      │              │       ├──────── 🌐 Web App    │
│  📝 Notes ───────────┤      │  LEANN Index │       │                        │
│  🗓️ Calendar ────────┤      └──────────────┘       └──────── 📱 Mobile     │
│  📷 Photos ──────────┘            │                                          │
│                                   │                                          │
│                          ┌────────▼────────┐                                │
│                          │ UNIFIED SEARCH  │                                │
│                          │ "Find my wife's │                                │
│                          │  doctor email"  │                                │
│                          └─────────────────┘                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔌 Part 1: Data Connectors

### Email Connector (Gmail/Outlook)

```python
# email_connector.py
from autonomous_brain import AutonomousBrain
import imaplib
import email

class EmailConnector:
    """Connects your email to the brain."""
    
    def __init__(self, brain: AutonomousBrain):
        self.brain = brain
    
    def connect_gmail(self, email_addr: str, app_password: str):
        """Connect to Gmail and learn from emails."""
        mail = imaplib.IMAP4_SSL('imap.gmail.com')
        mail.login(email_addr, app_password)
        mail.select('inbox')
        
        # Get recent emails
        _, messages = mail.search(None, 'ALL')
        
        for num in messages[0].split()[-100:]:  # Last 100 emails
            _, msg = mail.fetch(num, '(RFC822)')
            email_body = email.message_from_bytes(msg[0][1])
            
            subject = email_body['subject']
            sender = email_body['from']
            date = email_body['date']
            body = self._get_body(email_body)
            
            # Teach to brain
            content = f"Email from {sender} on {date}\nSubject: {subject}\n{body}"
            self.brain.teach(content, category="email")
    
    def _get_body(self, msg):
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    return part.get_payload(decode=True).decode()
        return msg.get_payload(decode=True).decode()
```

### Local Files Connector

```python
# file_connector.py
import os
from pathlib import Path

class FileConnector:
    """Indexes your local files."""
    
    def __init__(self, brain: AutonomousBrain):
        self.brain = brain
    
    def index_folder(self, folder_path: str, extensions: list = None):
        """Index all files in a folder."""
        extensions = extensions or ['.txt', '.md', '.pdf', '.docx', '.py']
        
        for root, dirs, files in os.walk(folder_path):
            # Skip hidden folders
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in extensions:
                    path = os.path.join(root, file)
                    try:
                        self.brain.learn_document(path)
                    except:
                        pass  # Skip files that can't be read
```

### Browser History Connector

```python
# browser_connector.py
import sqlite3
import os

class BrowserConnector:
    """Learns from your browser history."""
    
    def __init__(self, brain: AutonomousBrain):
        self.brain = brain
    
    def index_chrome_history(self):
        """Index Chrome browsing history."""
        # Chrome history location
        history_path = os.path.expanduser(
            "~/.config/google-chrome/Default/History"
        )
        
        # Copy to avoid lock
        import shutil
        temp_path = "/tmp/chrome_history"
        shutil.copy2(history_path, temp_path)
        
        conn = sqlite3.connect(temp_path)
        cursor = conn.cursor()
        
        # Get recent history
        cursor.execute("""
            SELECT url, title, visit_count, last_visit_time 
            FROM urls 
            ORDER BY last_visit_time DESC 
            LIMIT 1000
        """)
        
        for url, title, visits, last_visit in cursor.fetchall():
            content = f"Visited: {title}\nURL: {url}\nVisits: {visits}"
            self.brain.teach(content, category="browsing")
```

---

## 🔍 Part 2: LEANN-Style Unified Search

```python
# unified_search.py
class UnifiedSearch:
    """Search across ALL your data instantly."""
    
    def __init__(self, brain: AutonomousBrain):
        self.brain = brain
        
    def search(self, query: str) -> list:
        """
        Search across everything:
        - Emails
        - Files
        - Documents
        - Notes
        - Browser history
        """
        # Brain's neural memory finds relevant content
        results = []
        
        # Search text store
        for text_id, full_text in self.brain.memory.text_store.items():
            if self._matches(query, full_text):
                # Get metadata
                meta = self._get_metadata(text_id)
                results.append({
                    'text': full_text[:500],
                    'source': meta.get('source', 'unknown'),
                    'relevance': self._score(query, full_text)
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance'], reverse=True)
        
        return results[:10]
    
    def natural_search(self, query: str) -> str:
        """
        Natural language search.
        
        Example: "Find the email from my wife about the doctor bill"
        """
        return self.brain.ask(query)
```

---

## 🤖 Part 3: LLM Integration (ChatGPT, Claude, etc.)

### Option A: Browser Extension

```javascript
// brain_extension.js - Browser extension for ChatGPT/Claude

class BrainExtension {
    constructor() {
        this.brainAPI = 'http://localhost:5000';
    }
    
    // Inject brain context into LLM prompts
    async enhancePrompt(userMessage) {
        // Get relevant context from brain
        const response = await fetch(`${this.brainAPI}/api/v1/context`, {
            method: 'POST',
            body: JSON.stringify({ query: userMessage })
        });
        
        const context = await response.json();
        
        // Inject as system context
        return `
[Personal Context from your brain (willie.said):]
${context.relevant_memories}

[Your Question:]
${userMessage}
        `;
    }
    
    // Save conversation to brain
    async saveConversation(messages) {
        await fetch(`${this.brainAPI}/api/v1/learn`, {
            method: 'POST',
            body: JSON.stringify({
                content: messages.map(m => `${m.role}: ${m.content}`).join('\n'),
                source: 'chatgpt_conversation'
            })
        });
    }
}
```

### Option B: API Middleware

```python
# brain_middleware.py
from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()

# Load your brain
from autonomous_brain import AutonomousBrain
willie = AutonomousBrain.load("willie.said")
willie.start()

@app.post("/chat")
async def chat_with_context(message: str):
    """
    Chat with any LLM, but with YOUR brain's context.
    """
    # Get relevant context from brain
    context = willie.ask(message)
    
    # Send to LLM with context
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"""
You are a personal assistant with access to the user's personal brain.

Relevant context from their memory:
{context}

Use this context to provide personalized answers.
            """},
            {"role": "user", "content": message}
        ]
    )
    
    # Save the conversation to brain
    willie.teach(f"Q: {message}\nA: {response.choices[0].message.content}")
    willie.save("willie.said")
    
    return {"response": response.choices[0].message.content}
```

---

## 💻 Part 4: Cursor/IDE Integration

### Cursor Extension

```python
# cursor_brain_plugin.py
"""
Cursor plugin that remembers all your development discussions.
"""

class CursorBrainPlugin:
    def __init__(self):
        from autonomous_brain import AutonomousBrain
        self.brain = AutonomousBrain.load("cursor_brain.said")
        self.brain.start()
    
    def on_conversation(self, messages: list):
        """Called when a conversation happens in Cursor."""
        # Learn from the conversation
        conversation = "\n".join([
            f"{m['role']}: {m['content']}" 
            for m in messages
        ])
        
        self.brain.teach(conversation, category="cursor_discussion")
    
    def get_context(self, query: str) -> str:
        """Get relevant past discussions."""
        return self.brain.ask(query)
    
    def inject_context(self, prompt: str) -> str:
        """Inject relevant past context into prompt."""
        relevant = self.get_context(prompt)
        
        return f"""
[Relevant past discussions:]
{relevant}

[Current request:]
{prompt}
        """
```

### How to use in Cursor:

1. **Save discussions automatically:**
   ```python
   # After each conversation
   plugin.on_conversation(current_messages)
   ```

2. **Recall past discussions:**
   ```python
   # "Remember when we discussed the memory architecture?"
   context = plugin.get_context("memory architecture discussion")
   ```

3. **Full context of 100+ discussions:**
   ```python
   # Brain remembers ALL discussions
   # Just ask naturally
   plugin.brain.ask("What did we decide about the chunking approach?")
   ```

---

## 🌐 Part 5: Web Interface

```python
# brain_web.py
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Personal Brain Interface")

# Load brain
from autonomous_brain import AutonomousBrain
brain = AutonomousBrain.load("willie.said")
brain.start()

class TeachRequest(BaseModel):
    content: str
    category: str = "general"

class AskRequest(BaseModel):
    question: str

@app.post("/teach")
def teach(req: TeachRequest):
    result = brain.teach(req.content, category=req.category)
    brain.save("willie.said")
    return {"success": True, "result": result}

@app.post("/ask")
def ask(req: AskRequest):
    answer = brain.ask(req.question)
    return {"answer": answer}

@app.get("/stats")
def stats():
    return brain.stats()

@app.get("/interests")
def interests():
    return {"interests": brain.get_interests()}

@app.post("/learn-document")
def learn_document(path: str):
    result = brain.learn_document(path)
    brain.save("willie.said")
    return result

# Run: uvicorn brain_web:app --host 0.0.0.0 --port 5000
```

---

## 🏃 Quick Start: Connect Everything

```python
# setup_my_brain.py
"""
One script to connect your entire digital life to your brain.
"""

from autonomous_brain import AutonomousBrain

# Create or load your brain
import os
if os.path.exists("willie.said"):
    brain = AutonomousBrain.load("willie.said")
else:
    brain = AutonomousBrain("willie")

brain.start()

# 1. Index your documents
print("📁 Indexing documents...")
brain.learn_document("/path/to/important/documents")

# 2. Connect email (optional)
# from email_connector import EmailConnector
# EmailConnector(brain).connect_gmail("you@gmail.com", "app_password")

# 3. Index browser history (optional)
# from browser_connector import BrowserConnector
# BrowserConnector(brain).index_chrome_history()

# 4. Save
brain.save("willie.said")

print("✅ Brain connected to your digital life!")
print("🌐 Start web interface: uvicorn brain_web:app --port 5000")
```

---

## 🔮 The Complete Flow

```
YOU: "Find the email from my wife about the doctor bill"
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PERSONAL BRAIN (willie.said)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. UNDERSTAND (Neural Memory)                                  │
│     "wife" + "email" + "doctor" + "bill" → semantic search     │
│                                                                  │
│  2. SEARCH (LEANN-style)                                        │
│     ├── Emails: Found 3 matches                                 │
│     ├── Files: Found 1 match (doctor_receipt.pdf)              │
│     └── Notes: Found 0 matches                                  │
│                                                                  │
│  3. RANK (Your interests/preferences)                           │
│     Best match: Email from sarah@gmail.com on March 15         │
│     Subject: "Doctor appointment bill"                         │
│                                                                  │
│  4. RETURN                                                       │
│     "I found an email from Sarah on March 15 about..."         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
         ANSWER DELIVERED IN <100ms
```

---

## 📋 Summary: Integration Points

| Integration | How | Effort |
|-------------|-----|--------|
| **Local Files** | FileConnector + folder indexing | Easy |
| **Emails** | IMAP connector | Medium |
| **Browser History** | SQLite reader | Easy |
| **ChatGPT/Claude** | Browser extension or API middleware | Medium |
| **Cursor** | Plugin + context injection | Medium |
| **Web Interface** | FastAPI server | Easy |
| **Mobile** | REST API + mobile app | Hard |

---

## 🚀 Next Steps

1. **Start the web API** for your brain
2. **Index your important folders**
3. **Install browser extension** (coming soon)
4. **Use with ChatGPT/Claude** via API
5. **Remember: Brain gets smarter over time!**

