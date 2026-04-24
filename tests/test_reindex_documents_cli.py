"""Phase 10 guards: reindex_documents CLI accepts the right selectors and
refuses to run without an explicit target.
"""

from __future__ import annotations

import pytest

from scripts import reindex_documents


def test_cli_accepts_document_id_selector():
    args = reindex_documents._parse_args(["--document-id", "abc"])
    assert args.document_id == "abc"


def test_cli_accepts_conversation_id_selector():
    args = reindex_documents._parse_args(["--conversation-id", "conv-1"])
    assert args.conversation_id == "conv-1"


def test_cli_accepts_all_flag():
    args = reindex_documents._parse_args(["--all"])
    assert args.all is True


def test_cli_rejects_both_document_and_conversation():
    with pytest.raises(SystemExit):
        reindex_documents._parse_args(["--document-id", "d", "--conversation-id", "c"])


def test_cli_accepts_dry_run():
    args = reindex_documents._parse_args(["--dry-run"])
    assert args.dry_run is True


def test_cli_supports_continue_on_error():
    args = reindex_documents._parse_args(["--all", "--continue-on-error"])
    assert args.continue_on_error is True
