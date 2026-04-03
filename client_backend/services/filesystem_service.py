"""
Filesystem service for local file operations.

Provides secure file read/write/search operations with workspace sandboxing.
"""

import asyncio
import mimetypes
import re
from pathlib import Path

from client_backend.core.config import client_settings
from client_backend.core.logging import get_logger
from client_backend.core.paths import (
    validate_workspace_path,
)

logger = get_logger(__name__)


class FileInfo:
    """Information about a file or directory."""

    def __init__(self, path: Path):
        self.path = path
        self.name = path.name
        self.is_file = path.is_file()
        self.is_dir = path.is_dir()
        self.size = path.stat().st_size if self.is_file else 0
        self.modified_time = path.stat().st_mtime
        self.mime_type = mimetypes.guess_type(str(path))[0] if self.is_file else None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "path": str(self.path),
            "name": self.name,
            "is_file": self.is_file,
            "is_dir": self.is_dir,
            "size": self.size,
            "modified_time": self.modified_time,
            "mime_type": self.mime_type,
        }


class FilesystemError(Exception):
    """Raised when filesystem operations fail."""

    pass


class FilesystemService:
    """
    Service for filesystem operations with security controls.

    All operations are sandboxed to configured workspace roots.
    """

    def __init__(self):
        self.workspace_roots = client_settings.workspace_roots
        self.max_file_size = 50 * 1024 * 1024  # 50MB limit for safety

    async def read_file(
        self,
        file_path: str | Path,
        encoding: str = "utf-8",
        max_size: int | None = None,
    ) -> str:
        """
        Read a text file.

        Args:
            file_path: Path to the file.
            encoding: Text encoding (default: utf-8).
            max_size: Maximum size in bytes (defaults to configured limit).

        Returns:
            File contents as string.

        Raises:
            FilesystemError: If read fails or path is outside workspace.
        """
        try:
            validated_path = validate_workspace_path(file_path)

            if not validated_path.exists():
                raise FilesystemError(f"File not found: {file_path}")

            if not validated_path.is_file():
                raise FilesystemError(f"Path is not a file: {file_path}")

            # Check file size
            size_limit = max_size or self.max_file_size
            file_size = validated_path.stat().st_size
            if file_size > size_limit:
                raise FilesystemError(f"File too large: {file_size} bytes (limit: {size_limit})")

            # Read file
            content = await asyncio.to_thread(validated_path.read_text, encoding=encoding)

            logger.debug(f"Read file: {validated_path} ({file_size} bytes)")
            return content

        except FilesystemError:
            raise
        except Exception as e:
            raise FilesystemError(f"Failed to read file: {e}")

    async def read_file_bytes(
        self,
        file_path: str | Path,
        max_size: int | None = None,
    ) -> bytes:
        """
        Read a file as bytes.

        Args:
            file_path: Path to the file.
            max_size: Maximum size in bytes (defaults to configured limit).

        Returns:
            File contents as bytes.

        Raises:
            FilesystemError: If read fails or path is outside workspace.
        """
        try:
            validated_path = validate_workspace_path(file_path)

            if not validated_path.exists():
                raise FilesystemError(f"File not found: {file_path}")

            if not validated_path.is_file():
                raise FilesystemError(f"Path is not a file: {file_path}")

            # Check file size
            size_limit = max_size or self.max_file_size
            file_size = validated_path.stat().st_size
            if file_size > size_limit:
                raise FilesystemError(f"File too large: {file_size} bytes (limit: {size_limit})")

            # Read file
            content = await asyncio.to_thread(validated_path.read_bytes)

            logger.debug(f"Read file (bytes): {validated_path} ({file_size} bytes)")
            return content

        except FilesystemError:
            raise
        except Exception as e:
            raise FilesystemError(f"Failed to read file: {e}")

    async def write_file(
        self,
        file_path: str | Path,
        content: str,
        encoding: str = "utf-8",
        create_dirs: bool = True,
    ) -> FileInfo:
        """
        Write text content to a file.

        Args:
            file_path: Path to the file.
            content: Content to write.
            encoding: Text encoding (default: utf-8).
            create_dirs: Create parent directories if they don't exist.

        Returns:
            FileInfo for the written file.

        Raises:
            FilesystemError: If write fails or path is outside workspace.
        """
        try:
            validated_path = validate_workspace_path(file_path)

            # Create parent directories if needed
            if create_dirs:
                validated_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            await asyncio.to_thread(validated_path.write_text, content, encoding=encoding)

            logger.info(f"Wrote file: {validated_path} ({len(content)} chars)")
            return FileInfo(validated_path)

        except FilesystemError:
            raise
        except Exception as e:
            raise FilesystemError(f"Failed to write file: {e}")

    async def write_file_bytes(
        self,
        file_path: str | Path,
        content: bytes,
        create_dirs: bool = True,
    ) -> FileInfo:
        """
        Write bytes to a file.

        Args:
            file_path: Path to the file.
            content: Bytes to write.
            create_dirs: Create parent directories if they don't exist.

        Returns:
            FileInfo for the written file.

        Raises:
            FilesystemError: If write fails or path is outside workspace.
        """
        try:
            validated_path = validate_workspace_path(file_path)

            # Create parent directories if needed
            if create_dirs:
                validated_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            await asyncio.to_thread(validated_path.write_bytes, content)

            logger.info(f"Wrote file (bytes): {validated_path} ({len(content)} bytes)")
            return FileInfo(validated_path)

        except FilesystemError:
            raise
        except Exception as e:
            raise FilesystemError(f"Failed to write file: {e}")

    async def list_directory(
        self,
        dir_path: str | Path,
        recursive: bool = False,
        pattern: str | None = None,
    ) -> list[FileInfo]:
        """
        List files and directories.

        Args:
            dir_path: Directory path to list.
            recursive: Recursively list subdirectories.
            pattern: Optional glob pattern to filter results.

        Returns:
            List of FileInfo objects.

        Raises:
            FilesystemError: If listing fails or path is outside workspace.
        """
        try:
            validated_path = validate_workspace_path(dir_path)

            if not validated_path.exists():
                raise FilesystemError(f"Directory not found: {dir_path}")

            if not validated_path.is_dir():
                raise FilesystemError(f"Path is not a directory: {dir_path}")

            # List entries
            if recursive and pattern:
                entries = list(validated_path.rglob(pattern))
            elif recursive:
                entries = list(validated_path.rglob("*"))
            elif pattern:
                entries = list(validated_path.glob(pattern))
            else:
                entries = list(validated_path.iterdir())

            # Convert to FileInfo
            result = [FileInfo(entry) for entry in sorted(entries)]

            logger.debug(f"Listed directory: {validated_path} ({len(result)} entries)")
            return result

        except FilesystemError:
            raise
        except Exception as e:
            raise FilesystemError(f"Failed to list directory: {e}")

    async def create_directory(
        self,
        dir_path: str | Path,
        parents: bool = True,
    ) -> FileInfo:
        """
        Create a directory.

        Args:
            dir_path: Directory path to create.
            parents: Create parent directories if needed.

        Returns:
            FileInfo for the created directory.

        Raises:
            FilesystemError: If creation fails or path is outside workspace.
        """
        try:
            validated_path = validate_workspace_path(dir_path)

            await asyncio.to_thread(validated_path.mkdir, parents=parents, exist_ok=True)

            logger.info(f"Created directory: {validated_path}")
            return FileInfo(validated_path)

        except FilesystemError:
            raise
        except Exception as e:
            raise FilesystemError(f"Failed to create directory: {e}")

    async def delete_file(self, file_path: str | Path) -> None:
        """
        Delete a file.

        Args:
            file_path: Path to the file to delete.

        Raises:
            FilesystemError: If deletion fails or path is outside workspace.
        """
        try:
            validated_path = validate_workspace_path(file_path)

            if not validated_path.exists():
                raise FilesystemError(f"File not found: {file_path}")

            if not validated_path.is_file():
                raise FilesystemError(f"Path is not a file: {file_path}")

            await asyncio.to_thread(validated_path.unlink)

            logger.info(f"Deleted file: {validated_path}")

        except FilesystemError:
            raise
        except Exception as e:
            raise FilesystemError(f"Failed to delete file: {e}")

    async def delete_directory(
        self,
        dir_path: str | Path,
        recursive: bool = False,
    ) -> None:
        """
        Delete a directory.

        Args:
            dir_path: Path to the directory to delete.
            recursive: Delete directory and all contents (dangerous!).

        Raises:
            FilesystemError: If deletion fails or path is outside workspace.
        """
        try:
            validated_path = validate_workspace_path(dir_path)

            if not validated_path.exists():
                raise FilesystemError(f"Directory not found: {dir_path}")

            if not validated_path.is_dir():
                raise FilesystemError(f"Path is not a directory: {dir_path}")

            if recursive:
                import shutil

                await asyncio.to_thread(shutil.rmtree, validated_path)
            else:
                await asyncio.to_thread(validated_path.rmdir)

            logger.info(f"Deleted directory: {validated_path} (recursive={recursive})")

        except FilesystemError:
            raise
        except Exception as e:
            raise FilesystemError(f"Failed to delete directory: {e}")

    async def search_files(
        self,
        root_path: str | Path,
        pattern: str,
        content_pattern: str | None = None,
        max_results: int = 100,
    ) -> list[tuple[FileInfo, list[str] | None]]:
        """
        Search for files by name and optionally by content.

        Args:
            root_path: Root directory to search from.
            pattern: Glob pattern for file names (e.g., "*.py").
            content_pattern: Optional regex pattern to search file contents.
            max_results: Maximum number of results to return.

        Returns:
            List of (FileInfo, matching_lines) tuples.
            matching_lines is None if content_pattern is not provided.

        Raises:
            FilesystemError: If search fails or path is outside workspace.
        """
        try:
            validated_path = validate_workspace_path(root_path)

            if not validated_path.exists():
                raise FilesystemError(f"Root path not found: {root_path}")

            if not validated_path.is_dir():
                raise FilesystemError(f"Root path is not a directory: {root_path}")

            # Search for files matching pattern
            matching_files = list(validated_path.rglob(pattern))[:max_results]

            results = []
            content_regex = re.compile(content_pattern) if content_pattern else None

            for file_path in matching_files:
                if not file_path.is_file():
                    continue

                file_info = FileInfo(file_path)
                matching_lines = None

                # Search content if pattern provided
                if content_regex:
                    try:
                        content = await self.read_file(file_path, max_size=10 * 1024 * 1024)
                        matching_lines = [
                            line for line in content.splitlines() if content_regex.search(line)
                        ][:50]  # Limit matching lines per file

                        if not matching_lines:
                            continue  # Skip files with no content match

                    except Exception as e:
                        logger.warning(f"Could not search content of {file_path}: {e}")
                        continue

                results.append((file_info, matching_lines))

                if len(results) >= max_results:
                    break

            logger.debug(f"Search completed: {len(results)} results for pattern '{pattern}'")
            return results

        except FilesystemError:
            raise
        except Exception as e:
            raise FilesystemError(f"Failed to search files: {e}")

    async def get_file_info(self, file_path: str | Path) -> FileInfo:
        """
        Get information about a file or directory.

        Args:
            file_path: Path to the file or directory.

        Returns:
            FileInfo object.

        Raises:
            FilesystemError: If path is invalid or outside workspace.
        """
        try:
            validated_path = validate_workspace_path(file_path)

            if not validated_path.exists():
                raise FilesystemError(f"Path not found: {file_path}")

            return FileInfo(validated_path)

        except FilesystemError:
            raise
        except Exception as e:
            raise FilesystemError(f"Failed to get file info: {e}")


# Global singleton
_filesystem_service: FilesystemService | None = None


def get_filesystem_service() -> FilesystemService:
    """Get the global filesystem service."""
    global _filesystem_service
    if _filesystem_service is None:
        _filesystem_service = FilesystemService()
    return _filesystem_service
