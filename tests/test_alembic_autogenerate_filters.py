from app.alembic.autogenerate_filters import EXTERNAL_TABLE_NAMES, include_name


def test_external_tables_are_excluded_from_autogenerate():
    for table_name in EXTERNAL_TABLE_NAMES:
        assert include_name(table_name, "table", {}) is False


def test_application_tables_are_included_in_autogenerate():
    assert include_name("messages", "table", {}) is True
    assert include_name("document_chunks", "table", {}) is True
