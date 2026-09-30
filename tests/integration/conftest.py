"""Refuse to run integration tests against the application's own database.

Every module in this directory calls ``Base.metadata.create_all`` and seeds
rows. Pointed at the app database, that is not merely untidy:

* ``create_all`` builds tables and indexes from the *models*, bypassing
  Alembic. That is how ``ix_chat_images_sha256`` came to exist on a database
  whose migrations never created it, and it kept ``alembic check`` dirty until
  ``c9d0e1f2a3b4`` removed it.
* ``test_model_usage_repository_postgres.py`` and
  ``test_conversation_compaction_postgres.py`` assert over aggregates that
  include every row in the table, so real data makes them fail for reasons
  that have nothing to do with the code under test.
* the seed/cleanup fixtures delete by primary key, so a bug in one is a delete
  against production rows.

The guard is unconditional rather than per-module. A module that only touches
its own ``uuid4`` rows today is one edit away from not doing so, and the cost
of being wrong is someone's data.

``tests/conftest.py`` already refuses this at configure time for the whole suite.
The check stays here because the cost of it silently going missing is the same.
It compares against the application database as configured *before* the suite
rebinds ``DATABASE_URL``: after the rebind ``settings.database_url`` is the test
database itself, so comparing against it would refuse every run.
"""

from __future__ import annotations

from tests import database_isolation


def pytest_collection_modifyitems(config, items):
    """Fail the integration suite outright when it is aimed at the app database."""
    test_url = database_isolation.configured_test_database_url()
    if not test_url:
        return
    database_isolation.refuse_application_database(
        test_url, database_isolation.application_database_url()
    )
