"""Alembic migration helpers used during application startup."""

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ALEMBIC_INI = _PROJECT_ROOT / "alembic.ini"
_SCRIPT_LOCATION = _PROJECT_ROOT / "app" / "alembic"


def _build_alembic_config() -> Config:
    config = Config(str(_ALEMBIC_INI))
    config.set_main_option("script_location", str(_SCRIPT_LOCATION))
    config.attributes["configure_logger"] = False
    return config


def upgrade_database() -> None:
    logger.info("Applying database migrations")
    command.upgrade(_build_alembic_config(), "head")
    logger.info("Database migrations are current")
