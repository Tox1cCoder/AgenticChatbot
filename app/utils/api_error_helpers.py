from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import HTTPException, status


@contextmanager
def translate_service_errors(
    *,
    action: str,
    value_error_status: int | None = None,
) -> Iterator[None]:
    try:
        yield
    except HTTPException:
        raise
    except ValueError as exc:
        if value_error_status is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to {action}: {exc}",
            ) from exc
        raise HTTPException(status_code=value_error_status, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to {action}: {exc}",
        ) from exc
