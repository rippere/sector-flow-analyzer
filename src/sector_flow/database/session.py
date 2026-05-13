"""
SQLite session management for Sector Flow Analyzer.

Three database paths are used at different times — do not mix them:

  Development:  DATABASE_URL=sqlite:///./dev.db
                File at project root. Used for local iteration and manual testing.
                Run: sector-flow analyze --db sqlite:///./dev.db

  Production:   DATABASE_URL=sqlite:////home/rippere/.sector_flow/sector_flow.db
                Managed by systemd cron (sector-flow ingest + sector-flow analyze).
                Run: sector-flow ingest --db sqlite:////home/rippere/.sector_flow/sector_flow.db

  Docker:       DATABASE_URL=sqlite:////data/sector_flow.db
                Mounted as a Docker volume. Set automatically by docker-compose.yml.
                Do not set this manually.

WAL mode is enabled automatically for all SQLite connections to allow
concurrent readers during long analysis runs.
"""

from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from sector_flow.config import settings
from sector_flow.database.models import Base

_engine = None
_SessionFactory = None


def get_engine(database_url: str | None = None):
    global _engine
    if _engine is None:
        url = database_url or settings.database_url
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args, echo=False)

        # Enable WAL mode for SQLite — allows concurrent readers during writes.
        # In-memory SQLite (:memory: or sqlite://) silently ignores WAL; this is safe.
        if url.startswith("sqlite"):
            with _engine.connect() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL"))
                conn.execute(text("PRAGMA synchronous=NORMAL"))

    return _engine


def get_session_factory(database_url: str | None = None) -> sessionmaker:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(database_url), expire_on_commit=False)
    return _SessionFactory


def init_db(database_url: str | None = None) -> None:
    Base.metadata.create_all(get_engine(database_url))


@contextmanager
def get_session(database_url: str | None = None):
    factory = get_session_factory(database_url)
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
