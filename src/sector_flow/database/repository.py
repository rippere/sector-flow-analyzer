from datetime import datetime
from typing import Optional
import math
from sqlalchemy.orm import Session
from sector_flow.database.models import SectorETF, PriceData, SECTOR_ETFS


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
