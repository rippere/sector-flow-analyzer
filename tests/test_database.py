import pytest
from datetime import datetime
from sector_flow.database.models import SectorETF, PriceData, SECTOR_ETFS
from sector_flow.database.repository import ETFRepository, FlowMetricRepository, PriceRepository


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


def test_get_price_data_with_end_filter(db_session, mock_ohlcv_data):
    etf_repo = ETFRepository(db_session)
    etf_repo.seed_etfs()
    db_session.commit()
    etf = etf_repo.get_by_ticker("XLK")

    price_repo = PriceRepository(db_session)
    price_repo.save_price_data(etf.id, mock_ohlcv_data)
    db_session.commit()

    end = datetime(2024, 1, 3)
    rows = price_repo.get_price_data(etf.id, end=end)
    assert len(rows) == 2  # Jan 2, Jan 3


def test_update_flow_fields_creates_skeleton_when_no_row(db_session):
    etf_repo = ETFRepository(db_session)
    etf_repo.seed_etfs()
    db_session.commit()
    etf = db_session.query(SectorETF).filter_by(ticker="XLK").first()

    price_repo = PriceRepository(db_session)
    as_of = datetime(2024, 6, 1)
    result = price_repo.update_flow_fields(etf.id, as_of, shares_outstanding=1_000_000.0)
    db_session.commit()

    assert result is True
    rows = price_repo.get_price_data(etf.id)
    assert len(rows) == 1
    assert rows[0].shares_outstanding == 1_000_000.0
    assert rows[0].open == 0.0  # skeleton row has zero OHLCV


def test_update_flow_fields_updates_existing_row(db_session, mock_ohlcv_data):
    etf_repo = ETFRepository(db_session)
    etf_repo.seed_etfs()
    db_session.commit()
    etf = db_session.query(SectorETF).filter_by(ticker="XLK").first()

    price_repo = PriceRepository(db_session)
    price_repo.save_price_data(etf.id, mock_ohlcv_data)
    db_session.commit()

    as_of = datetime(2024, 1, 2)
    price_repo.update_flow_fields(etf.id, as_of, shares_outstanding=500_000.0, aum_usd=1e9)
    db_session.commit()

    rows = price_repo.get_price_data(etf.id, start=as_of, end=as_of)
    assert rows[0].shares_outstanding == 500_000.0
    assert rows[0].aum_usd == pytest.approx(1e9)


def test_compute_and_store_flows_calculates_net_inflow(db_session):
    etf_repo = ETFRepository(db_session)
    etf_repo.seed_etfs()
    db_session.commit()
    etf = db_session.query(SectorETF).filter_by(ticker="XLK").first()

    day1 = datetime(2024, 1, 2)
    day2 = datetime(2024, 1, 3)
    db_session.add(PriceData(
        etf_id=etf.id, date=day1, open=200.0, high=202.0, low=199.0, close=200.0,
        volume=1e6, adjusted_close=200.0,
        shares_outstanding=1_000_000.0, aum_usd=200_000_000.0,
    ))
    db_session.add(PriceData(
        etf_id=etf.id, date=day2, open=201.0, high=203.0, low=200.0, close=201.0,
        volume=1e6, adjusted_close=201.0, shares_outstanding=1_010_000.0,
    ))
    db_session.commit()

    price_repo = PriceRepository(db_session)
    updated = price_repo.compute_and_store_flows(etf.id)
    db_session.commit()

    assert updated == 1
    rows = price_repo.get_price_data(etf.id)
    # flow = (1_010_000 - 1_000_000) * (200_000_000 / 1_000_000) = 10_000 * 200 = 2_000_000
    assert rows[1].net_inflow_usd == pytest.approx(2_000_000.0)


def test_compute_and_store_flows_skips_prev_row_missing_aum(db_session):
    """prev row with None aum_usd is skipped without error."""
    etf_repo = ETFRepository(db_session)
    etf_repo.seed_etfs()
    db_session.commit()
    etf = db_session.query(SectorETF).filter_by(ticker="XLK").first()

    day1 = datetime(2024, 1, 2)
    day2 = datetime(2024, 1, 3)
    db_session.add(PriceData(
        etf_id=etf.id, date=day1, open=200.0, high=202.0, low=199.0, close=200.0,
        volume=1e6, adjusted_close=200.0, shares_outstanding=1_000_000.0,
        # aum_usd intentionally omitted (None)
    ))
    db_session.add(PriceData(
        etf_id=etf.id, date=day2, open=201.0, high=203.0, low=200.0, close=201.0,
        volume=1e6, adjusted_close=201.0, shares_outstanding=1_010_000.0,
    ))
    db_session.commit()

    price_repo = PriceRepository(db_session)
    updated = price_repo.compute_and_store_flows(etf.id)
    assert updated == 0


def test_compute_and_store_flows_skips_zero_shares_outstanding(db_session):
    """prev row with shares_outstanding == 0 causes division-by-zero guard — skipped."""
    etf_repo = ETFRepository(db_session)
    etf_repo.seed_etfs()
    db_session.commit()
    etf = db_session.query(SectorETF).filter_by(ticker="XLK").first()

    day1 = datetime(2024, 1, 2)
    day2 = datetime(2024, 1, 3)
    db_session.add(PriceData(
        etf_id=etf.id, date=day1, open=200.0, high=202.0, low=199.0, close=200.0,
        volume=1e6, adjusted_close=200.0, shares_outstanding=0.0, aum_usd=0.0,
    ))
    db_session.add(PriceData(
        etf_id=etf.id, date=day2, open=201.0, high=203.0, low=200.0, close=201.0,
        volume=1e6, adjusted_close=201.0, shares_outstanding=1_010_000.0,
    ))
    db_session.commit()

    price_repo = PriceRepository(db_session)
    updated = price_repo.compute_and_store_flows(etf.id)
    assert updated == 0


def test_compute_and_store_flows_skips_already_computed_rows(db_session):
    etf_repo = ETFRepository(db_session)
    etf_repo.seed_etfs()
    db_session.commit()
    etf = db_session.query(SectorETF).filter_by(ticker="XLK").first()

    day1 = datetime(2024, 1, 2)
    day2 = datetime(2024, 1, 3)
    db_session.add(PriceData(
        etf_id=etf.id, date=day1, open=200.0, high=202.0, low=199.0, close=200.0,
        volume=1e6, adjusted_close=200.0, shares_outstanding=1_000_000.0, aum_usd=200_000_000.0,
    ))
    db_session.add(PriceData(
        etf_id=etf.id, date=day2, open=201.0, high=203.0, low=200.0, close=201.0,
        volume=1e6, adjusted_close=201.0, shares_outstanding=1_010_000.0,
        net_inflow_usd=99.0,
    ))
    db_session.commit()

    price_repo = PriceRepository(db_session)
    updated = price_repo.compute_and_store_flows(etf.id)
    db_session.commit()

    assert updated == 0
    rows = price_repo.get_price_data(etf.id)
    assert rows[1].net_inflow_usd == pytest.approx(99.0)


def test_get_series_with_start_filter(db_session):
    etf_repo = ETFRepository(db_session)
    etf_repo.seed_etfs()
    db_session.commit()
    etf = db_session.query(SectorETF).filter_by(ticker="XLK").first()

    flow_repo = FlowMetricRepository(db_session)
    flow_repo.save_metric(etf.id, datetime(2024, 1, 1), "momentum", 0.5)
    flow_repo.save_metric(etf.id, datetime(2024, 1, 5), "momentum", 0.8)
    flow_repo.save_metric(etf.id, datetime(2024, 1, 10), "momentum", 1.0)
    db_session.commit()

    series = flow_repo.get_series(etf.id, "momentum", start=datetime(2024, 1, 5))
    assert len(series) == 2
    assert series[0].value == pytest.approx(0.8)
