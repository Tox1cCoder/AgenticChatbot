import logging
from pathlib import Path

from client_backend.core import logging as logging_module
from client_backend.core.config import ClientSettings, initialize_client_environment


def test_initialize_client_environment_defers_profile_side_effects(tmp_path):
    profile_root = tmp_path / "profile"
    settings = ClientSettings(profile_root=str(profile_root), local_session_secret="")

    assert not profile_root.exists()
    assert settings.local_session_secret == ""

    initialize_client_environment(settings)

    assert profile_root.is_dir()
    assert settings.local_session_secret
    assert (profile_root / ".local_session_secret").exists()


def test_setup_logging_defers_log_directory_creation_until_called(tmp_path, monkeypatch):
    profile_root = tmp_path / "profile"

    monkeypatch.setattr(logging_module.client_settings, "profile_root", str(profile_root))
    monkeypatch.setattr(logging_module.client_settings, "log_to_file", True)
    monkeypatch.setattr(logging_module.client_settings, "log_level", "INFO")

    logger = logging_module.get_logger(__name__)
    assert logger.name.startswith("client_backend")
    assert not (profile_root / "logs").exists()

    logging_module.setup_logging()

    assert (Path(profile_root) / "logs").is_dir()


def test_setup_logging_closes_replaced_handlers(monkeypatch):
    class TrackingHandler(logging.Handler):
        explicitly_closed = False

        def close(self) -> None:
            self.explicitly_closed = True
            super().close()

    logger = logging.getLogger("client_backend")
    original_handlers = list(logger.handlers)
    replaced = TrackingHandler()
    logger.handlers = [replaced]
    monkeypatch.setattr(logging_module.client_settings, "log_to_file", False)
    monkeypatch.setattr(logging_module.client_settings, "log_level", "INFO")

    try:
        logging_module.setup_logging()
        assert replaced.explicitly_closed is True
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.handlers = original_handlers
