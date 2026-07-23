from app.core.config import _load_or_create_dev_secret_key


def test_dev_secret_key_is_generated_and_persisted(tmp_path):
    key_path = tmp_path / ".dev_secret_key"

    first = _load_or_create_dev_secret_key(key_path)

    assert first
    assert key_path.read_text(encoding="utf-8").strip() == first


def test_dev_secret_key_is_stable_across_calls(tmp_path):
    """A server restart must reuse the persisted key, not mint a new one that
    would invalidate every previously issued token."""
    key_path = tmp_path / ".dev_secret_key"

    first = _load_or_create_dev_secret_key(key_path)
    second = _load_or_create_dev_secret_key(key_path)

    assert first == second


def test_dev_secret_key_reuses_existing_file(tmp_path):
    key_path = tmp_path / ".dev_secret_key"
    key_path.write_text("preexisting-key-value", encoding="utf-8")

    assert _load_or_create_dev_secret_key(key_path) == "preexisting-key-value"
