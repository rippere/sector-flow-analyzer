from contextlib import contextmanager
from sqlalchemy import create_engine
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
