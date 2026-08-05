import logging
from pathlib import Path

import pytest

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


def test_skills_root_defaults_to_the_profile(tmp_path):
    """Unset means "resolve per user under the profile", not "no skills"."""
    assert ClientSettings(profile_root=str(tmp_path)).skills_root == ""


def test_skills_root_rejects_a_list_of_paths(tmp_path):
    """Uploads install into this directory, so several of them have no meaning."""
    with pytest.raises(ValueError, match="single directory"):
        ClientSettings(skills_root=f"{tmp_path},{tmp_path / 'other'}")


def test_skills_root_rejects_a_relative_path():
    """The sidecar's working directory depends on its launcher."""
    with pytest.raises(ValueError, match="absolute"):
        ClientSettings(skills_root="skills")


def test_removed_plural_setting_names_its_replacement(tmp_path, monkeypatch):
    """A stale CLIENT_SKILLS_ROOTS in an env file must say what replaced it.

    The dotenv source is what reports unknown keys -- an unknown *environment*
    variable is ignored by pydantic-settings -- and an operator's env file is
    exactly where the removed name survives an upgrade.
    """
    env_file = tmp_path / "client-settings"
    env_file.write_text("CLIENT_SKILLS_ROOTS=C:/skills\n", encoding="utf-8")
    monkeypatch.setenv("CLIENT_ENV_FILE", str(env_file))

    with pytest.raises(ValueError, match="CLIENT_SKILLS_ROOT to a single absolute"):
        ClientSettings()


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
    replaced._client_backend_owned = True
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


def test_setup_logging_preserves_foreign_handlers(monkeypatch):
    logger = logging.getLogger("client_backend")
    original_handlers = list(logger.handlers)
    foreign = logging.NullHandler()
    logger.handlers = [foreign]
    monkeypatch.setattr(logging_module.client_settings, "log_to_file", False)

    try:
        logging_module.setup_logging()
        assert foreign in logger.handlers
        assert getattr(foreign, "_closed", False) is False
        logging_module.setup_logging()
        assert foreign in logger.handlers
        assert (
            sum(getattr(handler, "_client_backend_owned", False) for handler in logger.handlers)
            == 1
        )
    finally:
        for handler in logger.handlers:
            if handler is not foreign:
                handler.close()
        logger.handlers = original_handlers
