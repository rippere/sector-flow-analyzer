from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime,
    ForeignKey, UniqueConstraint, JSON
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

SECTOR_ETFS = [
    ("XLK", "Technology", "GICS-45"),
    ("XLF", "Financials", "GICS-40"),
    ("XLE", "Energy", "GICS-10"),
    ("XLV", "Healthcare", "GICS-35"),
    ("XLY", "Consumer Discretionary", "GICS-25"),
    ("XLP", "Consumer Staples", "GICS-30"),
    ("XLI", "Industrials", "GICS-20"),
    ("XLB", "Materials", "GICS-15"),
    ("XLRE", "Real Estate", "GICS-60"),
    ("XLU", "Utilities", "GICS-55"),
    ("XLC", "Communication Services", "GICS-50"),
]


class SectorETF(Base):
    __tablename__ = "sector_etfs"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), unique=True, nullable=False, index=True)
    sector_name = Column(String(100), nullable=False)
    sector_code = Column(String(20))
    description = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    price_data = relationship("PriceData", back_populates="etf", cascade="all, delete-orphan")
    covariance_matrices = relationship(
        "CovarianceMatrix",
        back_populates="etf_a",
        foreign_keys="CovarianceMatrix.etf_a_id",
    )


class PriceData(Base):
    __tablename__ = "price_data"

    id = Column(Integer, primary_key=True)
    etf_id = Column(Integer, ForeignKey("sector_etfs.id"), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    adjusted_close = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    etf = relationship("SectorETF", back_populates="price_data")

    __table_args__ = (
        UniqueConstraint("etf_id", "date", name="_etf_date_uc"),
    )


class CovarianceMatrix(Base):
    """Phase 2 placeholder — stores rolling covariance between sector pairs."""
    __tablename__ = "covariance_matrices"

    id = Column(Integer, primary_key=True)
    etf_a_id = Column(Integer, ForeignKey("sector_etfs.id"), nullable=False)
    etf_b_id = Column(Integer, ForeignKey("sector_etfs.id"), nullable=False)
    window_days = Column(Integer, nullable=False)
    computed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    covariance = Column(Float)
    correlation = Column(Float)
    metadata_ = Column("metadata", JSON)

    etf_a = relationship("SectorETF", foreign_keys=[etf_a_id], back_populates="covariance_matrices")
    etf_b = relationship("SectorETF", foreign_keys=[etf_b_id])

    __table_args__ = (
        UniqueConstraint("etf_a_id", "etf_b_id", "window_days", "computed_at", name="_cov_uc"),
    )


class FlowMetric(Base):
    """Phase 3 placeholder — stores directional flow indicators."""
    __tablename__ = "flow_metrics"

    id = Column(Integer, primary_key=True)
    etf_id = Column(Integer, ForeignKey("sector_etfs.id"), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    metric_name = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    computed_at = Column(DateTime, default=datetime.utcnow)
    metadata_ = Column("metadata", JSON)

    __table_args__ = (
        UniqueConstraint("etf_id", "date", "metric_name", name="_flow_uc"),
    )
