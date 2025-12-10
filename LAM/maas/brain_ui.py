#!/usr/bin/env python3
"""
🧠 BRAIN UI - Simple Web Interface for Your Personal Brain

A beautiful web interface to:
1. UPLOAD files from YOUR LOCAL computer (drag & drop!)
2. Ask questions
3. Teach new things
4. Trigger research on topics
5. See your interests

Start: python brain_ui.py
Open: http://localhost:8080
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autonomous_brain import AutonomousBrain
from data_connectors import FileConnector

# ============================================================
# CONFIGURATION
# ============================================================

BRAIN_FILE = os.environ.get("BRAIN_FILE", "willie_fresh.said")

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
file_connector = FileConnector(brain)

# ============================================================
# APP
# ============================================================

app = FastAPI(title="🧠 Willie Brain UI")

# ============================================================
# HTML UI - With LOCAL file upload support
# ============================================================

HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 Willie - Personal Brain</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0a0a0f;
            --bg-card: #12121a;
            --bg-hover: #1a1a25;
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.3);
            --text: #e2e8f0;
            --text-dim: #64748b;
            --success: #22c55e;
            --warning: #f59e0b;
            --border: #2a2a3a;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Space Grotesk', sans-serif;
            background: var(--bg-dark);
            color: var(--text);
            min-height: 100vh;
            background-image: 
                radial-gradient(ellipse at top, rgba(99, 102, 241, 0.1) 0%, transparent 50%),
                radial-gradient(ellipse at bottom right, rgba(236, 72, 153, 0.05) 0%, transparent 50%);
        }
        
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
        
        header { text-align: center; margin-bottom: 3rem; padding: 2rem 0; }
        
        h1 {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #6366f1 0%, #ec4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .subtitle { color: var(--text-dim); font-size: 1.1rem; }
        
        .stats-bar { display: flex; justify-content: center; gap: 2rem; margin-top: 1.5rem; flex-wrap: wrap; }
        
        .stat {
            background: var(--bg-card);
            padding: 1rem 1.5rem;
            border-radius: 12px;
            border: 1px solid var(--border);
            text-align: center;
        }
        
        .stat-value { font-size: 1.5rem; font-weight: 600; color: var(--accent); }
        .stat-label { font-size: 0.8rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1px; }
        
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
        
        .card {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid var(--border);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3); }
        
        .card-title { font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }
        .card-title .icon { font-size: 1.4rem; }
        
        input[type="text"], textarea {
            width: 100%;
            padding: 0.8rem 1rem;
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text);
            font-family: inherit;
            font-size: 1rem;
            margin-bottom: 0.8rem;
            transition: border-color 0.2s;
        }
        
        input[type="text"]:focus, textarea:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }
        
        textarea { min-height: 100px; resize: vertical; }
        
        button {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            border: none;
            padding: 0.8rem 1.5rem;
            border-radius: 8px;
            font-family: inherit;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            width: 100%;
        }
        
        button:hover { transform: translateY(-1px); box-shadow: 0 4px 20px var(--accent-glow); }
        button:active { transform: translateY(0); }
        
        button.secondary {
            background: var(--bg-dark);
            border: 1px solid var(--border);
        }
        button.secondary:hover { background: var(--bg-hover); }
        
        button.research {
            background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
        }
        
        .result {
            margin-top: 1rem;
            padding: 1rem;
            background: var(--bg-dark);
            border-radius: 8px;
            border-left: 3px solid var(--accent);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            line-height: 1.6;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .result.success { border-left-color: var(--success); }
        
        .interests-list { display: flex; flex-wrap: wrap; gap: 0.5rem; }
        
        .interest-tag {
            background: var(--bg-dark);
            padding: 0.4rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
            cursor: pointer;
            transition: background 0.2s;
        }
        .interest-tag:hover { background: var(--accent); }
        
        .interest-count {
            background: var(--accent);
            color: white;
            padding: 0.1rem 0.4rem;
            border-radius: 10px;
            font-size: 0.75rem;
        }
        
        .section-title { font-size: 1.5rem; font-weight: 600; margin: 2rem 0 1rem 0; color: var(--text); }
        
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .pulse { animation: pulse 1.5s infinite; }
        
        .full-width { grid-column: 1 / -1; }
        
        /* DRAG AND DROP UPLOAD ZONE */
        .upload-zone {
            border: 2px dashed var(--border);
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s;
            cursor: pointer;
            margin-bottom: 1rem;
        }
        
        .upload-zone:hover, .upload-zone.dragover {
            border-color: var(--accent);
            background: rgba(99, 102, 241, 0.1);
        }
        
        .upload-zone .icon { font-size: 3rem; margin-bottom: 1rem; }
        .upload-zone .text { color: var(--text-dim); margin-bottom: 0.5rem; }
        .upload-zone .hint { font-size: 0.85rem; color: var(--text-dim); }
        
        .upload-zone input { display: none; }
        
        .file-list { max-height: 150px; overflow-y: auto; margin-top: 1rem; }
        .file-item {
            padding: 0.5rem;
            background: var(--bg-dark);
            border-radius: 6px;
            margin-bottom: 0.3rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            display: flex;
            justify-content: space-between;
        }
        
        .uploaded-files { margin-top: 1rem; }
        
        .btn-group { display: flex; gap: 0.5rem; margin-top: 0.5rem; }
        .btn-group button { flex: 1; }
        
        .alert {
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .alert.info { background: rgba(99, 102, 241, 0.2); border: 1px solid var(--accent); }
        .alert.success { background: rgba(34, 197, 94, 0.2); border: 1px solid var(--success); }
        .alert.warning { background: rgba(245, 158, 11, 0.2); border: 1px solid var(--warning); }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 Willie</h1>
            <p class="subtitle">Your Personal AI Brain - Upload from YOUR Computer!</p>
            
            <div class="stats-bar" id="stats-bar">
                <div class="stat">
                    <div class="stat-value" id="stat-memories">--</div>
                    <div class="stat-label">Memories</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="stat-qa">--</div>
                    <div class="stat-label">Q&A Pairs</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="stat-research">--</div>
                    <div class="stat-label">Researched</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="stat-status">🟢</div>
                    <div class="stat-label">Learning</div>
                </div>
            </div>
        </header>
        
        <div class="grid">
            <!-- ASK -->
            <div class="card">
                <div class="card-title"><span class="icon">🔍</span> Ask Your Brain</div>
                <input type="text" id="ask-input" placeholder="What do you want to know?" onkeypress="if(event.key==='Enter')askQuestion()">
                <button onclick="askQuestion()">Ask</button>
                <div class="result" id="ask-result" style="display: none;"></div>
            </div>
            
            <!-- TEACH -->
            <div class="card">
                <div class="card-title"><span class="icon">📝</span> Teach Your Brain</div>
                <textarea id="teach-input" placeholder="What do you want to remember? (e.g., My son likes cycling and rode for team Picusa in Spain)"></textarea>
                <button onclick="teach()">Remember This</button>
                <div class="result success" id="teach-result" style="display: none;"></div>
            </div>
            
            <!-- UPLOAD FROM YOUR COMPUTER -->
            <div class="card full-width">
                <div class="card-title"><span class="icon">📤</span> Upload from YOUR Computer</div>
                
                <div class="alert info">
                    📁 <strong>Drag & drop files here</strong> or click to browse YOUR local computer. 
                    Supports: PDF, TXT, MD, DOCX, and more!
                </div>
                
                <div class="upload-zone" id="upload-zone" onclick="document.getElementById('file-input').click()">
                    <div class="icon">📂</div>
                    <div class="text">Drag & Drop Files Here</div>
                    <div class="hint">or click to browse your computer</div>
                    <input type="file" id="file-input" multiple accept=".pdf,.txt,.md,.docx,.doc,.py,.js,.json,.html,.css,.xml,.csv">
                </div>
                
                <div class="uploaded-files" id="uploaded-files" style="display: none;">
                    <strong>Files to upload:</strong>
                    <div class="file-list" id="file-list"></div>
                </div>
                
                <button onclick="uploadFiles()" id="upload-btn" style="display: none;">🚀 Upload & Learn from Files</button>
                <div class="result" id="upload-result" style="display: none;"></div>
            </div>
            
            <!-- RESEARCH TOPIC -->
            <div class="card">
                <div class="card-title"><span class="icon">🔬</span> Research a Topic</div>
                <p style="color: var(--text-dim); margin-bottom: 1rem; font-size: 0.9rem;">
                    Tell the brain to go learn about something from the internet!
                </p>
                <input type="text" id="research-input" placeholder="e.g., Picusa cycling team, Python programming" onkeypress="if(event.key==='Enter')researchTopic()">
                <button class="research" onclick="researchTopic()">🌐 Research This Now</button>
                <div class="result" id="research-result" style="display: none;"></div>
            </div>
            
            <!-- AUTONOMOUS LEARNING STATUS -->
            <div class="card">
                <div class="card-title"><span class="icon">🤖</span> Autonomous Learning</div>
                <div id="auto-learn-status">
                    <p style="color: var(--text-dim); margin-bottom: 1rem;">
                        The brain automatically researches topics you're interested in.
                    </p>
                </div>
                <div class="btn-group">
                    <button class="secondary" onclick="toggleAutoLearn()">Toggle Auto-Learn</button>
                    <button onclick="forceResearchCycle()">🔄 Force Research Cycle</button>
                </div>
                <div class="result" id="auto-result" style="display: none;"></div>
            </div>
        </div>
        
        <h2 class="section-title">🎯 Your Interests (click to research)</h2>
        <div class="card">
            <div class="interests-list" id="interests-list">
                <span class="pulse">Loading interests...</span>
            </div>
        </div>
        
        <h2 class="section-title">📚 What Willie Has Learned</h2>
        <div class="card">
            <div id="learned-summary">
                <span class="pulse">Loading...</span>
            </div>
        </div>
    </div>
    
    <script>
        const API = '';
        let selectedFiles = [];
        
        // ===== DRAG & DROP =====
        const uploadZone = document.getElementById('upload-zone');
        const fileInput = document.getElementById('file-input');
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(event => {
            uploadZone.addEventListener(event, e => { e.preventDefault(); e.stopPropagation(); });
        });
        
        ['dragenter', 'dragover'].forEach(event => {
            uploadZone.addEventListener(event, () => uploadZone.classList.add('dragover'));
        });
        
        ['dragleave', 'drop'].forEach(event => {
            uploadZone.addEventListener(event, () => uploadZone.classList.remove('dragover'));
        });
        
        uploadZone.addEventListener('drop', e => {
            const files = [...e.dataTransfer.files];
            handleFiles(files);
        });
        
        fileInput.addEventListener('change', e => {
            const files = [...e.target.files];
            handleFiles(files);
        });
        
        function handleFiles(files) {
            selectedFiles = files;
            
            const fileList = document.getElementById('file-list');
            const uploadedFiles = document.getElementById('uploaded-files');
            const uploadBtn = document.getElementById('upload-btn');
            
            if (files.length > 0) {
                uploadedFiles.style.display = 'block';
                uploadBtn.style.display = 'block';
                
                fileList.innerHTML = files.map(f => `
                    <div class="file-item">
                        <span>📄 ${f.name}</span>
                        <span>${formatSize(f.size)}</span>
                    </div>
                `).join('');
            } else {
                uploadedFiles.style.display = 'none';
                uploadBtn.style.display = 'none';
            }
        }
        
        function formatSize(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
            return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        }
        
        async function uploadFiles() {
            if (selectedFiles.length === 0) return;
            
            const result = document.getElementById('upload-result');
            result.style.display = 'block';
            result.innerHTML = `<span class="pulse">Uploading ${selectedFiles.length} files from your computer...</span>`;
            
            let success = 0;
            let errors = 0;
            
            for (const file of selectedFiles) {
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const res = await fetch(API + '/upload-file', { method: 'POST', body: formData });
                    const data = await res.json();
                    if (data.success) success++;
                    else errors++;
                } catch (e) {
                    errors++;
                }
            }
            
            result.innerHTML = `✅ Learned from ${success} files! (${errors} errors)`;
            result.classList.add('success');
            
            selectedFiles = [];
            document.getElementById('file-list').innerHTML = '';
            document.getElementById('uploaded-files').style.display = 'none';
            document.getElementById('upload-btn').style.display = 'none';
            
            fetchStats();
        }
        
        // ===== STATS & INTERESTS =====
        async function fetchStats() {
            try {
                const res = await fetch(API + '/stats');
                const data = await res.json();
                
                document.getElementById('stat-memories').textContent = data.texts_stored || 0;
                document.getElementById('stat-qa').textContent = data.qa_pairs || 0;
                document.getElementById('stat-research').textContent = data.topics_researched || 0;
                document.getElementById('stat-status').textContent = data.is_learning ? '🟢' : '🔴';
                
                // Update learned summary
                document.getElementById('learned-summary').innerHTML = `
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
                        <div><strong>${data.texts_stored}</strong> memories</div>
                        <div><strong>${data.qa_pairs}</strong> Q&A pairs</div>
                        <div><strong>${data.teachings}</strong> teachings</div>
                        <div><strong>${data.topics_researched}</strong> topics researched</div>
                        <div><strong>${data.knowledge_from_internet}</strong> facts from internet</div>
                    </div>
                `;
            } catch (e) { console.error(e); }
        }
        
        async function fetchInterests() {
            try {
                const res = await fetch(API + '/interests');
                const data = await res.json();
                
                const container = document.getElementById('interests-list');
                if (data.interests && data.interests.length > 0) {
                    container.innerHTML = data.interests.slice(0, 20).map(([topic, count]) => `
                        <div class="interest-tag" onclick="researchInterest('${topic}')">
                            ${topic}
                            <span class="interest-count">${count}</span>
                        </div>
                    `).join('');
                } else {
                    container.innerHTML = '<span style="color: var(--text-dim)">No interests detected yet. Teach me something!</span>';
                }
            } catch (e) { console.error(e); }
        }
        
        // ===== ACTIONS =====
        async function askQuestion() {
            const input = document.getElementById('ask-input');
            const result = document.getElementById('ask-result');
            if (!input.value.trim()) return;
            
            result.style.display = 'block';
            result.innerHTML = '<span class="pulse">Thinking...</span>';
            
            try {
                const res = await fetch(API + '/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: input.value})
                });
                const data = await res.json();
                result.innerHTML = data.answer || 'No answer found';
            } catch (e) { result.innerHTML = 'Error: ' + e.message; }
            
            fetchStats();
            fetchInterests();
        }
        
        async function teach() {
            const input = document.getElementById('teach-input');
            const result = document.getElementById('teach-result');
            if (!input.value.trim()) return;
            
            result.style.display = 'block';
            result.innerHTML = '<span class="pulse">Learning...</span>';
            
            try {
                const res = await fetch(API + '/teach', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({content: input.value, category: 'manual'})
                });
                const data = await res.json();
                result.innerHTML = `✅ Remembered! Total memories: ${data.memories}`;
                
                // Check if there are detected interests
                if (data.detected_interests && data.detected_interests.length > 0) {
                    result.innerHTML += `<br>🎯 Detected interests: ${data.detected_interests.join(', ')}`;
                }
                
                input.value = '';
            } catch (e) { result.innerHTML = 'Error: ' + e.message; }
            
            fetchStats();
            fetchInterests();
        }
        
        async function researchTopic() {
            const input = document.getElementById('research-input');
            const result = document.getElementById('research-result');
            if (!input.value.trim()) return;
            
            result.style.display = 'block';
            result.innerHTML = `<span class="pulse">🔍 Researching "${input.value}" on the internet...</span>`;
            
            try {
                const res = await fetch(API + '/research', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({topic: input.value})
                });
                const data = await res.json();
                
                if (data.facts_learned > 0) {
                    result.innerHTML = `✅ Learned ${data.facts_learned} facts about "${input.value}"!<br><br>Sample: ${data.sample || ''}`;
                    result.classList.add('success');
                } else {
                    result.innerHTML = `⚠️ Couldn't find information about "${input.value}". Try a more specific term.`;
                }
                
                input.value = '';
            } catch (e) { result.innerHTML = 'Error: ' + e.message; }
            
            fetchStats();
            fetchInterests();
        }
        
        async function researchInterest(topic) {
            document.getElementById('research-input').value = topic;
            researchTopic();
        }
        
        async function toggleAutoLearn() {
            try {
                const res = await fetch(API + '/toggle-auto-learn', { method: 'POST' });
                const data = await res.json();
                
                const result = document.getElementById('auto-result');
                result.style.display = 'block';
                result.innerHTML = data.enabled ? '🟢 Auto-learning ENABLED' : '🔴 Auto-learning DISABLED';
                
                fetchStats();
            } catch (e) { console.error(e); }
        }
        
        async function forceResearchCycle() {
            const result = document.getElementById('auto-result');
            result.style.display = 'block';
            result.innerHTML = '<span class="pulse">🔄 Running research cycle...</span>';
            
            try {
                const res = await fetch(API + '/force-research-cycle', { method: 'POST' });
                const data = await res.json();
                result.innerHTML = `✅ Researched ${data.topics_researched} topics, learned ${data.facts_learned} facts!`;
            } catch (e) { result.innerHTML = 'Error: ' + e.message; }
            
            fetchStats();
            fetchInterests();
        }
        
        // Initialize
        fetchStats();
        fetchInterests();
        setInterval(() => { fetchStats(); fetchInterests(); }, 10000);
    </script>
</body>
</html>
"""

# ============================================================
# ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_UI

@app.get("/stats")
def stats():
    s = brain.stats()
    s['interests'] = brain.get_interests()[:10]
    return s

@app.get("/interests")
def interests():
    return {"interests": brain.get_interests()}

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(req: AskRequest):
    answer = brain.ask(req.question)
    brain.save(BRAIN_FILE)
    return {"answer": answer}

class TeachRequest(BaseModel):
    content: str
    category: str = "manual"

@app.post("/teach")
def teach(req: TeachRequest):
    # Observe for interest detection
    brain.learning_loop.observe(req.content)
    
    # Get interests before teaching
    interests_before = set([t for t, _ in brain.get_interests()])
    
    # Teach
    brain.teach(req.content, category=req.category)
    brain.save(BRAIN_FILE)
    
    # Get new interests
    interests_after = set([t for t, _ in brain.get_interests()])
    new_interests = list(interests_after - interests_before)
    
    return {
        "success": True, 
        "memories": len(brain.memory.text_index),
        "detected_interests": new_interests
    }

class ResearchRequest(BaseModel):
    topic: str

@app.post("/research")
def research_topic(req: ResearchRequest):
    """Manually trigger research on a topic."""
    from autonomous_brain import WEB_AVAILABLE
    
    if not WEB_AVAILABLE:
        return {"facts_learned": 0, "error": "Web search not available (install requests and beautifulsoup4)"}
    
    # Use the searcher directly
    results = brain.learning_loop.searcher.search_topic(req.topic)
    
    if results:
        # Learn from results
        combined = f"About {req.topic}:\n" + "\n".join(results)
        brain.memory.learn_text(
            text=combined,
            source=f"manual_research:{req.topic}",
            key_proj=brain.key_proj,
            value_proj=brain.value_proj,
            embedder=brain.embedder,
            device=brain.memory.M_personal.device,
        )
        brain.learning_loop.topics_researched += 1
        brain.learning_loop.knowledge_added += len(results)
        brain.save(BRAIN_FILE)
        
        return {
            "facts_learned": len(results),
            "sample": results[0][:200] if results else ""
        }
    
    return {"facts_learned": 0, "sample": ""}

@app.post("/toggle-auto-learn")
def toggle_auto_learn():
    brain.learning_loop.autonomous_search_enabled = not brain.learning_loop.autonomous_search_enabled
    return {"enabled": brain.learning_loop.autonomous_search_enabled}

@app.post("/force-research-cycle")
def force_research_cycle():
    """Force a research cycle now."""
    before_topics = brain.learning_loop.topics_researched
    before_knowledge = brain.learning_loop.knowledge_added
    
    brain.learning_loop._learning_cycle()
    
    topics_researched = brain.learning_loop.topics_researched - before_topics
    facts_learned = brain.learning_loop.knowledge_added - before_knowledge
    
    brain.save(BRAIN_FILE)
    
    return {
        "topics_researched": topics_researched,
        "facts_learned": facts_learned
    }

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file from the user's LOCAL computer."""
    # Save to temp directory
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        print(f"📤 Uploaded file: {file.filename} ({len(content)} bytes)")
        
        # Learn from document
        try:
            result = brain.learn_document(temp_path)
            brain.save(BRAIN_FILE)
            return {
                "success": True, 
                "filename": file.filename,
                "result": result
            }
        except ImportError as e:
            if "pandas" in str(e) or "openpyxl" in str(e):
                return {
                    "success": False, 
                    "error": f"Excel file support requires: pip install pandas openpyxl",
                    "filename": file.filename
                }
            raise
        except Exception as e:
            return {
                "success": False, 
                "error": f"Failed to process file: {str(e)}",
                "filename": file.filename
            }
    except Exception as e:
        return {"success": False, "error": f"Upload failed: {str(e)}"}
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except:
                pass

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🧠 WILLIE BRAIN UI")
    print("=" * 60)
    print(f"   Brain: {BRAIN_FILE}")
    print(f"   Open: http://localhost:8080")
    print("   ✅ Upload files from YOUR computer via drag & drop!")
    print("   ✅ Research topics on demand!")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8080)
