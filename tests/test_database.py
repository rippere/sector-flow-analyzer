from sector_flow.database.models import SectorETF, PriceData, SECTOR_ETFS
from sector_flow.database.repository import ETFRepository, PriceRepository


def test_seed_etfs_creates_all(db_session):
    repo = ETFRepository(db_session)
    created = repo.seed_etfs()
    db_session.commit()
    assert len(created) == len(SECTOR_ETFS)
    assert db_session.query(SectorETF).count() == len(SECTOR_ETFS)


def test_seed_etfs_idempotent(db_session):
    repo = ETFRepository(db_session)
    repo.seed_etfs()
    db_session.commit()
    created2 = repo.seed_etfs()
    db_session.commit()
    assert len(created2) == 0
    assert db_session.query(SectorETF).count() == len(SECTOR_ETFS)


def test_get_by_ticker(db_session):
    repo = ETFRepository(db_session)
    repo.seed_etfs()
    db_session.commit()
    etf = repo.get_by_ticker("XLK")
    assert etf is not None
    assert etf.sector_name == "Technology"


def test_save_price_data(db_session, mock_ohlcv_data):
    etf_repo = ETFRepository(db_session)
    etf_repo.seed_etfs()
    db_session.commit()
    etf = etf_repo.get_by_ticker("XLK")

    price_repo = PriceRepository(db_session)
    saved = price_repo.save_price_data(etf.id, mock_ohlcv_data)
    db_session.commit()
    assert saved == 5


def test_save_price_data_no_duplicates(db_session, mock_ohlcv_data):
    etf_repo = ETFRepository(db_session)
    etf_repo.seed_etfs()
    db_session.commit()
    etf = etf_repo.get_by_ticker("XLK")

    price_repo = PriceRepository(db_session)
    price_repo.save_price_data(etf.id, mock_ohlcv_data)
    db_session.commit()
    saved2 = price_repo.save_price_data(etf.id, mock_ohlcv_data)
    db_session.commit()
    assert saved2 == 0
    assert db_session.query(PriceData).count() == 5


def test_get_price_data_with_date_filter(db_session, mock_ohlcv_data):
    from datetime import datetime, timedelta
    etf_repo = ETFRepository(db_session)
    etf_repo.seed_etfs()
    db_session.commit()
    etf = etf_repo.get_by_ticker("XLK")

    price_repo = PriceRepository(db_session)
    price_repo.save_price_data(etf.id, mock_ohlcv_data)
    db_session.commit()

    start = datetime(2024, 1, 4)
    rows = price_repo.get_price_data(etf.id, start=start)
    assert len(rows) == 3


def test_get_latest_date(db_session, mock_ohlcv_data):
    from datetime import datetime, timedelta
    etf_repo = ETFRepository(db_session)
    etf_repo.seed_etfs()
    db_session.commit()
    etf = etf_repo.get_by_ticker("XLK")

    price_repo = PriceRepository(db_session)
    price_repo.save_price_data(etf.id, mock_ohlcv_data)
    db_session.commit()

    latest = price_repo.get_latest_date(etf.id)
    assert latest == datetime(2024, 1, 6)
