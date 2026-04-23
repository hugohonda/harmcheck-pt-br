from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DADA_DATASETS_DIR = Path.home() / "Developer" / "mestrado" / "dada-pt-br" / "datasets"
DADA_TRANSLATED_DIR = Path.home() / "Developer" / "mestrado" / "dada-pt-br" / "output" / "01-translated"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ETHIC_", extra="ignore")

    ollama_host: str = Field(default="http://localhost:11434")
    models: list[str] = Field(default=["gemma4:e4b", "qwen3.5:9b", "llama3.1:8b"])
    datasets_dir: Path = Field(default=DADA_DATASETS_DIR)
    translated_dir: Path = Field(default=DADA_TRANSLATED_DIR)
    output_dir: Path = Field(default=Path("output"))

    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=1024)
    request_timeout: float = Field(default=300.0)

    # Concurrent requests sent per model. Ollama also needs OLLAMA_NUM_PARALLEL
    # on the server side to actually run them in parallel; otherwise it queues.
    num_parallel: int = Field(default=4)
    # Tell Ollama to keep the model resident. Prevents reload between prompts
    # of the same model. Unloaded implicitly when we switch to the next model.
    keep_alive: str = Field(default="30m")

    # Default language for runs. Research is Portuguese-focused.
    default_language: str = Field(default="pt-br")
    # Default LLM-as-judge model. Ideally different from any model under
    # evaluation to avoid self-bias. Uses 0 temperature and a fixed rubric.
    judge_model: str = Field(default="qwen3.5:9b")


settings = Settings()
