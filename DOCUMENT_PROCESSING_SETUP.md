# Document Processing Setup

## Prerequisites
1. **Redis Server**: Required for Celery message broker
   ```bash
   # Install Redis (Windows)
   # Download from: https://github.com/microsoftarchive/redis/releases
   # Or use WSL/Docker
   
   # Start Redis server
   redis-server
   ```

2. **Qdrant Server**: Optional for vector storage (document search)
   ```bash
   # Using Docker
   docker run -p 6333:6333 qdrant/qdrant
   
   # Or start existing local Qdrant
   # (Already configured in app/database/qdrant/)
   ```

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -e .
   ```

2. **Environment Variables**:
   ```bash
   # Add to .env file
   CELERY_BROKER_URL=redis://localhost:6379/0
   CELERY_RESULT_BACKEND=redis://localhost:6379/0
   ```

## Running the System

1. **Start the API Server** (in terminal 1):
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Start Redis Server** (in terminal 2):
   ```bash
   redis-server
   ```

3. **Start Celery Worker** (in terminal 3):
   ```bash
   # Option 1: Using the startup script
   python -m app.workers.start_worker
   
   # Option 2: Direct celery command
   celery -A app.workers.celery_app worker --loglevel=info --concurrency=2
   ```

## Testing Document Upload

1. Use the demo interface or API to upload a document
2. Check Celery worker logs for processing status
3. Document status will update from 'processing' → 'ready' or 'failed'
4. Vector embeddings will be stored in Qdrant (if running)

## Features Implemented

✅ **Document Upload Endpoint**: Enhanced `/documents/upload` with background processing
✅ **Background Task Queue**: Celery with Redis broker
✅ **Document Processing Pipeline**: Extract text, chunk, and embed
✅ **Status Management**: Tracks processing/ready/failed states
✅ **User Messaging**: Returns "document is being processed" during processing
✅ **Error Handling**: Retry logic and status updates
✅ **Qdrant Integration**: Vector storage for document chunks
✅ **RAG Agent Integration**: Uses existing document processing capabilities

## Monitoring

- **Worker Status**: Check terminal 3 for worker logs
- **Task Status**: Monitor Redis for task queues
- **Document Status**: Check database DOCUMENTS table
- **API Response**: Upload endpoint confirms background processing started