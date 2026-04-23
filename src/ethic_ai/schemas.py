from pydantic import BaseModel


class HarmPrompt(BaseModel):
    id: str
    category: str
    text: str
    language: str = "en"


class ModelResponse(BaseModel):
    model: str
    prompt_id: str
    category: str
    language: str
    prompt_text: str
    response: str = ""
    refused: bool = False
    error: str | None = None
    elapsed_seconds: float = 0.0
    eval_count: int = 0
    prompt_eval_count: int = 0
