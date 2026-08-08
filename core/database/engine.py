"""
AFE — Database Engine & Session Management

Provides the SQLAlchemy engine, session factory, and database
initialization helpers.  In development mode the database is SQLite;
in production it can be pointed to PostgreSQL via DATABASE_URL.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from core.config.settings import get_settings
from core.database.base import Base

logger = logging.getLogger(__name__)

_settings = get_settings()

# ──────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────
_connect_args = {}
if _settings.DATABASE_URL.startswith("sqlite"):
    _connect_args["check_same_thread"] = False

db_engine = create_engine(
    _settings.DATABASE_URL,
    echo=_settings.DEBUG and _settings.ENV == "development",
    connect_args=_connect_args,
    pool_pre_ping=True,
)

# Enable WAL mode and foreign keys for SQLite
if _settings.DATABASE_URL.startswith("sqlite"):

    @event.listens_for(db_engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()


# ──────────────────────────────────────────────
# Session factory
# ──────────────────────────────────────────────
SessionFactory = sessionmaker(bind=db_engine, expire_on_commit=False)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a transactional database session that auto-commits on success."""
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ──────────────────────────────────────────────
# Initialization
# ──────────────────────────────────────────────
def init_db() -> None:
    """Create all tables that don't yet exist."""
    # Ensure the data directory exists for SQLite
    if _settings.DATABASE_URL.startswith("sqlite"):
        db_path = _settings.DATABASE_URL.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Import all models so they are registered with Base.metadata
    import core.database.models  # noqa: F401

    try:
        Base.metadata.create_all(bind=db_engine, checkfirst=True)
    except Exception as e:
        logger.warning("Table creation notice: %s", e)
    logger.info("Database tables created / verified.")
