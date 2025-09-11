from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.database.session import get_db

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check(db: Session = Depends(get_db)) -> dict:
    """
    Health check endpoint to verify API and database connectivity
    """
    try:
        # Test database connectivity
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": db_status,
        "message": "Sample Chatbot API is running",
    }


@router.get("/db")
async def database_health(db: Session = Depends(get_db)) -> dict:
    """
    Detailed database health check
    """
    try:
        # Test basic connectivity
        db.execute(text("SELECT 1"))

        return {
            "status": "healthy",
            "database_connection": "connected",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")
