from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    database_url: str = Field(default="sqlite:///./sector_flow.db")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    data_refresh_interval: int = Field(default=3600)
    log_level: str = Field(default="INFO")

    # --- Market-hours / intraday operation ---
    market_tz: str = Field(default="America/New_York")
    intraday_enabled: bool = Field(default=True)
    intraday_interval_minutes: int = Field(default=20)
    # When True, the intraday refresh no-ops outside regular trading hours.
    rth_only: bool = Field(default=True)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
