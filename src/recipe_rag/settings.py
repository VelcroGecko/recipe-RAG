from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    recipes_path: str = "data/recipes.json"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    top_k: int = 3
    generator_base_url: str = "http://127.0.0.1:8080/v1"
    generator_model: str = "unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL"


settings = Settings()
