from app.database.session import SessionLocal, get_engine, get_session_factory


def test_database_session_factory_is_singleton_provider():
    assert get_session_factory() is SessionLocal


def test_database_engine_is_bound_to_session_factory():
    assert SessionLocal.kw["bind"] is get_engine()
