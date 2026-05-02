from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PIPELINE_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://quantuser:quantpass@localhost:5432/quantdb"
    database_url_sync: str = "postgresql://quantuser:quantpass@localhost:5432/quantdb"
    host: str = "0.0.0.0"
    port: int = 18840
    log_dir: str = "~/.mad-lab-train/logs"
    datagen_dir: str = "~/.mad-lab-train/datagen"


settings = Settings()
