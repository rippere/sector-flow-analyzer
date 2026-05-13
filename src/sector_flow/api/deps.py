"""FastAPI dependency injection helpers."""

from __future__ import annotations

from sqlalchemy.orm import Session
from sector_flow.database.session import get_session


def get_db():
    """Yield a SQLAlchemy session for use in a request."""
    with get_session() as session:
        yield session
