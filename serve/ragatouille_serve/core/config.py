from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "RAGatouille API"
    PROJECT_VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    MODEL_NAME: str = "colbert-ir/colbertv2.0"
    INDEX_ROOT: str = "local_store"

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
