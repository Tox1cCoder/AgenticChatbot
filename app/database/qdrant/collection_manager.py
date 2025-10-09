"""Centralized Qdrant collection management."""

import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

logger = logging.getLogger(__name__)


def ensure_collection(
    qdrant_client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance: Distance = Distance.COSINE,
) -> bool:
    """
    Ensure a Qdrant collection exists, creating it if necessary.
    """
    try:
        collections = qdrant_client.get_collections()

        if any(c.name == collection_name for c in collections.collections):
            logger.debug(f"Qdrant collection '{collection_name}' already exists")
            return True

        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )
        logger.info(
            f"Created Qdrant collection '{collection_name}' with vector size {vector_size}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to ensure Qdrant collection '{collection_name}': {e}")
        return False
