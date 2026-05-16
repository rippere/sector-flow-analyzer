"""
Tests for database session configuration — WAL mode and engine setup.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text


def test_wal_mode_enabled():
    """
    SQLite engines created via get_engine() should have WAL journal mode enabled.

    In-memory SQLite (:memory:) silently ignores WAL and returns 'memory' —
    this is expected and safe, since WAL only matters for file-based DBs.
    We test with a temporary file-based DB to confirm WAL is set.
    """
    import tempfile
    import os

    # Use a temporary file-based DB to verify WAL actually sticks
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_path = f.name

    try:
        # Reset module-level singletons to force a fresh engine
        import sector_flow.database.session as sess_mod
        orig_engine = sess_mod._engine
        orig_factory = sess_mod._SessionFactory
        sess_mod._engine = None
        sess_mod._SessionFactory = None

        db_url = f"sqlite:///{tmp_path}"
        engine = sess_mod.get_engine(db_url)

        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA journal_mode")).fetchone()
            journal_mode = result[0].lower()

        assert journal_mode in ("wal", "memory"), (
            f"Expected journal_mode to be 'wal' or 'memory', got '{journal_mode}'"
        )
        # For file-based DBs, must be WAL
        assert journal_mode == "wal", (
            f"File-based SQLite should be WAL, got '{journal_mode}'"
        )

    finally:
        # Restore singletons
        sess_mod._engine = orig_engine
        sess_mod._SessionFactory = orig_factory
        engine.dispose()
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def test_get_session_factory_returns_cached_instance():
    """get_session_factory returns the same object on repeated calls (cached singleton)."""
    import sector_flow.database.session as sess_mod
    orig_engine = sess_mod._engine
    orig_factory = sess_mod._SessionFactory
    sess_mod._engine = None
    sess_mod._SessionFactory = None

    try:
        f1 = sess_mod.get_session_factory("sqlite://")
        f2 = sess_mod.get_session_factory("sqlite://")
        assert f1 is f2
    finally:
        sess_mod._engine = orig_engine
        sess_mod._SessionFactory = orig_factory


def test_get_session_rolls_back_and_reraises_on_exception():
    """get_session rolls back the session and re-raises when an exception occurs."""
    import sector_flow.database.session as sess_mod
    orig_engine = sess_mod._engine
    orig_factory = sess_mod._SessionFactory
    sess_mod._engine = None
    sess_mod._SessionFactory = None

    try:
        with pytest.raises(ValueError, match="test rollback"):
            with sess_mod.get_session("sqlite://") as session:
                raise ValueError("test rollback")
    finally:
        sess_mod._engine = orig_engine
        sess_mod._SessionFactory = orig_factory
