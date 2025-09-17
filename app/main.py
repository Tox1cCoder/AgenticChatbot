from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.container import Container
from app.api import (
    health_router,
    users_router,
    conversations_router,
    messages_router,
    feedback_router,
)
from app.api.auth import router as auth_router


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""

    # Initialize the dependency injection container
    container = Container()
    container.wire(
        modules=[
            "app.api.auth",
            "app.api.users",
            "app.api.conversations",
            "app.api.messages",
            "app.api.feedback",
        ]
    )

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=settings.app_description,
        debug=settings.api_debug,
    )

    # Attach container to app for dependency injection
    app.container = container

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(auth_router)
    app.include_router(health_router)
    app.include_router(users_router)
    app.include_router(conversations_router)
    app.include_router(messages_router)
    app.include_router(feedback_router)

    return app


# Create the FastAPI app instance
app = create_app()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to the Sample Chatbot API",
        "version": settings.app_version,
        "docs_url": "/docs",
        "health_check": "/health",
    }
