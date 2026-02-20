from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from clearinghouse.storage.models import Base


def create_session_factory(database_url: str):
    engine = create_engine(database_url, future=True)
    return sessionmaker(bind=engine, expire_on_commit=False), engine


def init_db(engine) -> None:
    Base.metadata.create_all(engine)
