from datetime import datetime, timezone
from typing import Optional
import math


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)
from sqlalchemy.orm import Session
from sqlalchemy import and_
from sector_flow.database.models import SectorETF, PriceData, CovarianceMatrix, FlowMetric, SECTOR_ETFS


class ETFRepository:
    def __init__(self, session: Session):
        self.session = session

    def seed_etfs(self) -> list[SectorETF]:
        existing = {e.ticker for e in self.session.query(SectorETF).all()}
        created = []
        for ticker, sector_name, sector_code in SECTOR_ETFS:
            if ticker not in existing:
                etf = SectorETF(ticker=ticker, sector_name=sector_name, sector_code=sector_code)
                self.session.add(etf)
                created.append(etf)
        self.session.flush()
        return created

    def get_by_ticker(self, ticker: str) -> Optional[SectorETF]:
        return self.session.query(SectorETF).filter_by(ticker=ticker).first()

    def get_all(self) -> list[SectorETF]:
        return self.session.query(SectorETF).all()


class PriceRepository:
    def __init__(self, session: Session):
        self.session = session

    def save_price_data(self, etf_id: int, records: list[dict]) -> int:
        saved = 0
        for rec in records:
            existing = (
                self.session.query(PriceData)
                .filter_by(etf_id=etf_id, date=rec["date"])
                .first()
            )
            if existing is None:
                self.session.add(PriceData(etf_id=etf_id, **rec))
                saved += 1
            elif existing.close == 0.0 and rec.get("close"):
                # Fill SSGA skeleton rows (created pre-market with zeroed OHLCV) with
                # the real bar — left at zero they poison momentum/regime computation.
                for col in ("open", "high", "low", "close", "volume", "adjusted_close"):
                    if rec.get(col) is not None:
                        setattr(existing, col, rec[col])
                saved += 1
        self.session.flush()
        return saved

    def get_price_data(
        self,
        etf_id: int,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> list[PriceData]:
        q = self.session.query(PriceData).filter_by(etf_id=etf_id)
        if start:
            q = q.filter(PriceData.date >= start)
        if end:
            q = q.filter(PriceData.date <= end)
        return q.order_by(PriceData.date).all()

    def get_latest_date(self, etf_id: int) -> Optional[datetime]:
        row = (
            self.session.query(PriceData.date)
            .filter_by(etf_id=etf_id)
            .order_by(PriceData.date.desc())
            .first()
        )
        return row[0] if row else None

    def update_flow_fields(
        self,
        etf_id: int,
        as_of: datetime,
        shares_outstanding: Optional[float] = None,
        aum_usd: Optional[float] = None,
    ) -> bool:
        """Write SSGA snapshot fields onto an existing PriceData row for as_of date.
        Creates a skeleton row if one doesn't exist yet (SSGA runs before market close)."""
        row = self.session.query(PriceData).filter_by(etf_id=etf_id, date=as_of).first()
        if row is None:
            row = PriceData(
                etf_id=etf_id,
                date=as_of,
                open=0.0, high=0.0, low=0.0, close=0.0,
                volume=0.0, adjusted_close=0.0,
            )
            self.session.add(row)

        if shares_outstanding is not None:
            row.shares_outstanding = shares_outstanding
        if aum_usd is not None:
            row.aum_usd = aum_usd
            if shares_outstanding and shares_outstanding > 0:
                row.net_inflow_usd = None  # will be computed in compute_and_store_flows
        self.session.flush()
        return True

    def compute_and_store_flows(self, etf_id: int) -> int:
        """
        Compute net_inflow_usd for all rows that have shares_outstanding
        but no net_inflow_usd yet, using the industry formula:
            flow = (shares_today - shares_yesterday) * nav_yesterday
        Returns count of rows updated.
        """
        rows = (
            self.session.query(PriceData)
            .filter(
                PriceData.etf_id == etf_id,
                PriceData.shares_outstanding.isnot(None),
            )
            .order_by(PriceData.date)
            .all()
        )

        updated = 0
        for i in range(1, len(rows)):
            curr = rows[i]
            prev = rows[i - 1]
            if curr.net_inflow_usd is not None:
                continue
            if prev.shares_outstanding is None or prev.aum_usd is None:
                continue
            if prev.shares_outstanding == 0:
                continue
            nav_yesterday = prev.aum_usd / prev.shares_outstanding
            delta_shares = curr.shares_outstanding - prev.shares_outstanding
            curr.net_inflow_usd = delta_shares * nav_yesterday
            updated += 1

        self.session.flush()
        return updated


class FlowMetricRepository:
    """Repository for per-sector FlowMetric records (regime, momentum, cohesion)."""

    def __init__(self, session: Session):
        self.session = session

    def save_metric(
        self,
        etf_id: int,
        date: datetime,
        metric_name: str,
        value: float,
    ) -> bool:
        """Upsert — update if exists, insert if not."""
        existing = (
            self.session.query(FlowMetric)
            .filter_by(etf_id=etf_id, date=date, metric_name=metric_name)
            .first()
        )
        if existing is not None:
            existing.value = value
            existing.computed_at = _utcnow()
        else:
            self.session.add(
                FlowMetric(
                    etf_id=etf_id,
                    date=date,
                    metric_name=metric_name,
                    value=value,
                    computed_at=_utcnow(),
                )
            )
        self.session.flush()
        return True

    def get_latest(self, etf_id: int, metric_name: str) -> Optional[float]:
        """Most recent value for this metric."""
        row = (
            self.session.query(FlowMetric)
            .filter_by(etf_id=etf_id, metric_name=metric_name)
            .order_by(FlowMetric.date.desc())
            .first()
        )
        return row.value if row else None

    def get_series(
        self,
        etf_id: int,
        metric_name: str,
        start: Optional[datetime] = None,
    ) -> list[FlowMetric]:
        """Time series for charting."""
        q = self.session.query(FlowMetric).filter_by(etf_id=etf_id, metric_name=metric_name)
        if start:
            q = q.filter(FlowMetric.date >= start)
        return q.order_by(FlowMetric.date).all()


class CovarianceRepository:
    """Repository for rolling covariance/correlation pairs."""

    def __init__(self, session: Session):
        self.session = session

    def save_pairs(self, pairs: list[dict]) -> int:
        """
        Upsert covariance pairs into CovarianceMatrix.

        Each dict must have: etf_a_id, etf_b_id, window_days, computed_at,
        covariance, correlation.

        Returns count of rows inserted or updated.
        """
        saved = 0
        for pair in pairs:
            existing = (
                self.session.query(CovarianceMatrix)
                .filter_by(
                    etf_a_id=pair["etf_a_id"],
                    etf_b_id=pair["etf_b_id"],
                    window_days=pair["window_days"],
                    computed_at=pair["computed_at"],
                )
                .first()
            )
            if existing is not None:
                existing.covariance = pair["covariance"]
                existing.correlation = pair["correlation"]
            else:
                self.session.add(
                    CovarianceMatrix(
                        etf_a_id=pair["etf_a_id"],
                        etf_b_id=pair["etf_b_id"],
                        window_days=pair["window_days"],
                        computed_at=pair["computed_at"],
                        covariance=pair["covariance"],
                        correlation=pair["correlation"],
                    )
                )
            saved += 1
        self.session.flush()
        return saved

    def get_latest_matrix(self, window: int = 30) -> list[CovarianceMatrix]:
        """Most recent covariance pairs for all sector pairs with given window."""
        # Find the most recent computed_at timestamp for this window
        latest_row = (
            self.session.query(CovarianceMatrix.computed_at)
            .filter_by(window_days=window)
            .order_by(CovarianceMatrix.computed_at.desc())
            .first()
        )
        if latest_row is None:
            return []
        latest_ts = latest_row[0]
        return (
            self.session.query(CovarianceMatrix)
            .filter_by(window_days=window, computed_at=latest_ts)
            .all()
        )
