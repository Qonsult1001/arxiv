#!/usr/bin/env python3
"""
.SAID Memory API - Production FastAPI with .SAID Protocol Support

Complete production-ready API with:
- /api/v1/ versioning
- .SAID Protocol v1.0.0 endpoints
- Prometheus metrics
- OpenTelemetry tracing support
- All Phase 1 endpoints
- LLM integration helpers

.SAID Protocol Endpoints:
- POST /api/v1/said/upload - Upload .SAID memory domain
- GET  /api/v1/said/download - Download .SAID memory domain
- POST /api/v1/said/query - Query specific .SAID domain

Start the API:
    uvicorn memory_api_production:app --host 0.0.0.0 --port 5000 --workers 4

Access:
    API Docs: http://localhost:5000/docs
    Metrics:  http://localhost:5000/metrics
    Health:   http://localhost:5000/health

.SAID — Where AI Memory Lives
"""

from fastapi import FastAPI, HTTPException, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import os
import logging
import time
import sys; sys.path.insert(0, "../core"); from simple_memory_wrapper import MyBrain

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== Prometheus Metrics ==========
# Request metrics
request_count = Counter(
    'memory_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'memory_api_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

# Business metrics
memory_total = Gauge('memory_api_memories_total', 'Total memories stored')
document_total = Gauge('memory_api_documents_total', 'Total documents stored')
tokens_total = Gauge('memory_api_tokens_total', 'Total conversation tokens')
s_slow_magnitude = Gauge('memory_api_s_slow_magnitude', 'S_slow magnitude (long-term memory strength)')

# Operation metrics
remember_operations = Counter('memory_api_remember_operations_total', 'Total remember operations')
recall_operations = Counter('memory_api_recall_operations_total', 'Total recall operations')
search_operations = Counter('memory_api_search_operations_total', 'Total search operations')

# ========== FastAPI App ==========
app = FastAPI(
    title="Personal AI Memory API - Production",
    description="Production-ready memory system with monitoring and LLM integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Global brain instance (.SAID domain)
brain: Optional[MyBrain] = None
BRAIN_NAME = "api_brain"  # Creates api_brain.said file


def get_brain() -> MyBrain:
    """Get or create the brain instance (.SAID domain)."""
    global brain
    if brain is None:
        if os.path.exists(f"{BRAIN_NAME}.said"):
            brain = MyBrain.load(BRAIN_NAME)
            logger.info(f"Loaded .SAID domain with {len(brain.brain.memory_index)} memories")
        else:
            brain = MyBrain(name=BRAIN_NAME, auto_save=True)
            logger.info("Created new .SAID memory domain")
        update_metrics()
    return brain


def update_metrics():
    """Update Prometheus metrics from brain state."""
    b = get_brain()
    memory_total.set(len(b.brain.memory_index))
    document_total.set(len(b.brain.documents))
    tokens_total.set(b.brain.total_conversation_tokens)
    s_slow_magnitude.set(b.brain.S_slow.norm().item())


# ========== Middleware for Metrics ==========
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Track request metrics."""
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time

    # Track metrics
    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)

    return response


# ========== Pydantic Models ==========
class DocumentUploadRequest(BaseModel):
    text: str = Field(..., description="Document text", min_length=1)
    doc_id: Optional[str] = Field(None, description="Optional document ID")
    title: Optional[str] = Field(None, description="Optional document title")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "This is the API documentation...",
                "doc_id": "api_docs_v1",
                "title": "API Documentation v1.0",
                "metadata": {"version": "1.0", "author": "Team"}
            }
        }


class DocumentResponse(BaseModel):
    doc_id: str
    title: Optional[str]
    num_chunks: int
    approx_tokens: int
    created_at: str


class DocumentDetailResponse(BaseModel):
    doc_id: str
    title: Optional[str]
    full_text: str
    num_chunks: int
    chunks: List[str]
    metadata: Optional[Dict[str, Any]]


class QueryRequest(BaseModel):
    query: str = Field(..., description="Search query", min_length=1)
    doc_id: Optional[str] = Field(None, description="Optional document ID to search within")
    top_k: Optional[int] = Field(3, description="Number of results", ge=1, le=20)

    class Config:
        json_schema_extra = {
            "example": {
                "query": "How does authentication work?",
                "top_k": 5
            }
        }


class QueryResult(BaseModel):
    text: str
    doc_id: str
    score: float
    chunk_id: int


class QueryResponse(BaseModel):
    query: str
    results: List[QueryResult]
    total_results: int
    latency_ms: float


class SynthesizeRequest(BaseModel):
    question: str = Field(..., description="Question to answer", min_length=1)
    context_size: Optional[int] = Field(5, description="Number of context chunks to use")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the recommended authentication method?",
                "context_size": 5
            }
        }


class SynthesizeResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    source: str
    context_used: List[str]
    latency_ms: float


class CacheStats(BaseModel):
    total_memories: int
    total_documents: int
    total_tokens: int
    s_slow_magnitude: float
    step_count: int
    cache_size_bytes: int
    brain_name: str


# ========== Phase 1 API Endpoints (/api/v1) ==========

@app.post("/api/v1/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(request: DocumentUploadRequest):
    """
    Upload a document for storage and indexing.

    The document will be:
    1. Chunked into manageable pieces
    2. Indexed for fast keyword search
    3. Embedded for semantic search
    4. Stored persistently

    **Example**:
    ```json
    {
      "text": "API Documentation\\n\\nAuthentication uses JWT tokens...",
      "doc_id": "api_docs",
      "title": "API Documentation"
    }
    ```
    """
    start_time = time.time()

    b = get_brain()
    result = b.add_document(
        request.text,
        doc_id=request.doc_id,
        title=request.title
    )

    update_metrics()

    latency = (time.time() - start_time) * 1000
    logger.info(f"Document uploaded: {result['doc_id']} ({result['num_chunks']} chunks, {latency:.1f}ms)")

    return DocumentResponse(
        doc_id=result['doc_id'],
        title=request.title,
        num_chunks=result['num_chunks'],
        approx_tokens=result['approx_tokens'],
        created_at=time.strftime('%Y-%m-%d %H:%M:%S')
    )


@app.get("/api/v1/documents/{doc_id}", response_model=DocumentDetailResponse)
async def get_document(doc_id: str):
    """
    Retrieve a specific document by ID.

    Returns the full document text, all chunks, and metadata.

    **Example**:
    ```
    GET /api/v1/documents/api_docs
    ```
    """
    b = get_brain()

    if doc_id not in b.brain.documents:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{doc_id}' not found"
        )

    doc = b.brain.documents[doc_id]

    return DocumentDetailResponse(
        doc_id=doc_id,
        title=doc.get('title'),
        full_text=doc.get('full_text', ''),
        num_chunks=len(doc['chunks']),
        chunks=doc['chunks'],
        metadata=doc.get('metadata', {})
    )


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents using hybrid search (keyword + semantic).

    Returns ranked chunks from documents matching the query.

    **Example**:
    ```json
    {
      "query": "authentication methods",
      "top_k": 5
    }
    ```
    """
    start_time = time.time()

    b = get_brain()
    results = b.search_documents(
        request.query,
        doc_id=request.doc_id,
        top_k=request.top_k
    )

    search_operations.inc()
    update_metrics()

    latency = (time.time() - start_time) * 1000

    formatted_results = [
        QueryResult(
            text=r['text'],
            doc_id=r['doc_id'],
            score=r['score'],
            chunk_id=r.get('chunk_id', 0)
        )
        for r in results
    ]

    logger.info(f"Query: '{request.query}' -> {len(results)} results ({latency:.1f}ms)")

    return QueryResponse(
        query=request.query,
        results=formatted_results,
        total_results=len(results),
        latency_ms=latency
    )


@app.post("/api/v1/query/synthesize", response_model=SynthesizeResponse)
async def query_synthesize(request: SynthesizeRequest):
    """
    Query with answer synthesis.

    1. Retrieves relevant context from memory
    2. Synthesizes a coherent answer
    3. Returns answer with confidence and sources

    **Example**:
    ```json
    {
      "question": "What authentication method should I use?",
      "context_size": 5
    }
    ```
    """
    start_time = time.time()

    b = get_brain()

    # Get context from documents
    search_results = b.search_documents(request.question, top_k=request.context_size)
    context_chunks = [r['text'] for r in search_results]

    # Recall from memory (synthesizes answer)
    result = b.brain.recall(request.question, compute_semantic_score=True)

    recall_operations.inc()
    update_metrics()

    latency = (time.time() - start_time) * 1000

    logger.info(f"Synthesize: '{request.question}' -> confidence {result.get('confidence', 0):.2f} ({latency:.1f}ms)")

    return SynthesizeResponse(
        question=request.question,
        answer=result['recalled_content'],
        confidence=result.get('confidence', 0.0),
        source=result.get('source', 'memory'),
        context_used=context_chunks,
        latency_ms=latency
    )


@app.get("/api/v1/cache/stats", response_model=CacheStats)
async def cache_stats():
    """
    Get cache and brain statistics.

    Returns:
    - Total memories stored
    - Total documents indexed
    - Total tokens processed
    - Memory strength (S_slow magnitude)
    - Cache size
    """
    b = get_brain()

    # Calculate cache size (.SAID file)
    cache_size = 0
    if os.path.exists(f"{BRAIN_NAME}.said"):
        cache_size = os.path.getsize(f"{BRAIN_NAME}.said")

    update_metrics()

    return CacheStats(
        total_memories=len(b.brain.memory_index),
        total_documents=len(b.brain.documents),
        total_tokens=b.brain.total_conversation_tokens,
        s_slow_magnitude=b.brain.S_slow.norm().item(),
        step_count=b.brain.step_count,
        cache_size_bytes=cache_size,
        brain_name=b.name
    )


# ========== .SAID Protocol Endpoints ==========

@app.post("/api/v1/said/upload", status_code=status.HTTP_201_CREATED)
async def upload_said_domain(file: bytes, domain: Optional[str] = None):
    """
    Upload a .SAID memory domain file.

    .SAID Protocol endpoint for uploading portable memory domains.

    Args:
        file: .SAID file content (multipart/form-data)
        domain: Optional domain name override

    Returns:
        - domain: Loaded domain name
        - memories: Number of memories loaded
        - documents: Number of documents loaded
    """
    import tempfile
    import torch

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(suffix=".said", delete=False) as tmp:
        tmp.write(file)
        tmp_path = tmp.name

    try:
        # Load the .SAID file
        checkpoint = torch.load(tmp_path)

        # Extract domain info
        domain_name = checkpoint.get('said_domain', domain or 'uploaded_domain')
        owner_info = checkpoint.get('owner', {})

        # Load into current brain (merge)
        b = get_brain()
        # Merge memories from uploaded domain
        if 'memory_index' in checkpoint:
            b.brain.memory_index.extend(checkpoint['memory_index'])
        if 'documents' in checkpoint:
            b.brain.documents.update(checkpoint['documents'])

        # Update model state if needed
        if 'model_state_dict' in checkpoint:
            b.brain.load_state_dict(checkpoint['model_state_dict'], strict=False)

        update_metrics()

        return {
            "success": True,
            "domain": domain_name,
            "subdomain": owner_info.get('subdomain', f"{domain_name}.saidhome.ai"),
            "owner_type": owner_info.get('type', 'unknown'),
            "memories_loaded": len(checkpoint.get('memory_index', [])),
            "documents_loaded": len(checkpoint.get('documents', {})),
            "said_version": checkpoint.get('said_version', 'unknown')
        }
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


@app.get("/api/v1/said/download")
async def download_said_domain(domain: Optional[str] = None):
    """
    Download current memory as a .SAID domain file.

    .SAID Protocol endpoint for exporting portable memory domains.

    Args:
        domain: Domain name for the .SAID file (default: current brain)

    Returns:
        .SAID file for download
    """
    b = get_brain()

    # Create temp file for export
    import tempfile
    domain_name = domain or "exported_domain.said"
    if not domain_name.endswith('.said'):
        domain_name += '.said'

    with tempfile.NamedTemporaryFile(suffix=".said", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Save current brain as .SAID file
        b.brain.save_checkpoint(tmp_path, domain=domain_name)

        # Return file for download
        return FileResponse(
            path=tmp_path,
            filename=domain_name,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{domain_name}"'
            }
        )
    except Exception as e:
        # Clean up on error
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@app.post("/api/v1/said/query")
async def query_said_domain(domain: str, question: str, top_k: int = 5):
    """
    Query a specific .SAID memory domain.

    .SAID Protocol endpoint for querying memory domains.

    Args:
        domain: .SAID domain name (e.g., "alice.said")
        question: Query question
        top_k: Number of results to return

    Returns:
        - domain: Queried domain
        - answer: Synthesized answer
        - results: Top-k relevant memories
    """
    b = get_brain()

    # For now, query the current brain
    # In future: support loading multiple domains
    result = b.brain.recall(question, compute_semantic_score=True)

    recall_operations.inc()
    update_metrics()

    return {
        "domain": domain,
        "question": question,
        "answer": result['recalled_content'],
        "confidence": result.get('confidence', 0.0),
        "source": result.get('source', 'unknown'),
        "said_protocol": "v1.0.0"
    }


# ========== Legacy Endpoints (Backward Compatibility) ==========

@app.post("/remember")
async def remember_legacy(request: dict):
    """Legacy endpoint - use /api/v1/documents instead."""
    b = get_brain()
    result = b.brain.memorize(request['text'], memory_type=request.get('memory_type', 'general'))
    remember_operations.inc()
    update_metrics()
    return {
        "success": True,
        "memory_id": len(b.brain.memory_index) - 1,
        "novelty": result['psi']
    }


@app.post("/recall")
async def recall_legacy(request: dict):
    """Legacy endpoint - use /api/v1/query/synthesize instead."""
    b = get_brain()
    result = b.brain.recall(request['question'])
    recall_operations.inc()
    update_metrics()
    return {
        "success": True,
        "answer": result['recalled_content']
    }


# ========== Monitoring Endpoints ==========

@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus format:
    - Request counts and latencies
    - Memory/document counts
    - Business metrics
    """
    update_metrics()
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/health")
async def health():
    """
    Health check endpoint.

    Returns:
    - status: "healthy" or "unhealthy"
    - brain_loaded: true/false
    - memory_count: number of memories
    """
    try:
        b = get_brain()
        return {
            "status": "healthy",
            "brain_loaded": True,
            "memory_count": len(b.brain.memory_index),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


# ========== UI Endpoints ==========

@app.get("/", response_class=FileResponse)
async def index():
    """Serve the main web interface."""
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return JSONResponse(
        status_code=404,
        content={"error": "Web interface not found"}
    )


@app.get("/admin", response_class=FileResponse)
async def admin():
    """Serve the admin panel."""
    if os.path.exists("static/admin.html"):
        return FileResponse("static/admin.html")
    return JSONResponse(
        status_code=404,
        content={"error": "Admin panel not found"}
    )


# ========== Startup Event ==========

@app.on_event("startup")
async def startup_event():
    """Initialize brain on startup."""
    logger.info("="*60)
    logger.info("🧠 Personal AI Memory API - Production")
    logger.info("="*60)
    logger.info("Starting API server...")
    logger.info("API Documentation: http://localhost:5000/docs")
    logger.info("Metrics: http://localhost:5000/metrics")
    logger.info("Health: http://localhost:5000/health")
    logger.info("="*60)

    # Initialize brain
    get_brain()


# Run with: uvicorn memory_api_production:app --host 0.0.0.0 --port 5000 --workers 4
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)