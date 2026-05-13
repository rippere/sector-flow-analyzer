from datetime import datetime
from typing import Optional
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
