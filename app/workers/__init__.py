"""
Background Task Worker Implementation

This module provides background task processing capabilities for document upload and processing using Celery as the task queue system.

Requirements:
- Redis server for Celery broker
- Celery worker for background processing
- Integration with existing RAG agent for document processing

Features:
- Async document processing
- Status tracking in database
- Error handling and retry logic
- Qdrant vector storage integration
- Chunk extraction and embedding

Dependencies:
- celery[redis]==5.3.4
- redis==5.0.1

Usage:
1. Start Redis server: redis-server
2. Start Celery worker: celery -A app.workers.celery_app worker --loglevel=info
3. Use upload endpoint to trigger background processing

Note: Task modules (document_processor, cleanup_tasks) are automatically imported
by Celery via celery_app.conf.imports configuration to avoid circular import issues.
"""

# Only export celery_app to avoid circular imports
# Task modules are registered via celery_app.conf.imports
__all__ = ["celery_app"]
