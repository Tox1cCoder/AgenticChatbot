"""
Base validation utility class for consistent patterns
"""

from abc import ABC


class BaseValidationUtils(ABC):
    """Base validation utility class for consistent initialization patterns"""

    def __init__(self, session_factory: callable):
        """
        Initialize validation utils with session factory for dependency injection.

        Args:
            session_factory: Callable that returns database session context managers
        """
        self.session_factory = session_factory
        self._init_repositories()

    def _init_repositories(self):
        """Initialize repositories - to be implemented by subclasses"""
        pass
