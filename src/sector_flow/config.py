from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    database_url: str = Field(default="sqlite:///./sector_flow.db")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    data_refresh_interval: int = Field(default=3600)
    log_level: str = Field(default="INFO")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
