import json
import random
from collections import defaultdict
from pathlib import Path

from .schemas import HarmPrompt

ALERT_INSTRUCTION = "### Instruction:"
ALERT_RESPONSE = "### Response:"


def _load_raw(path: Path) -> list[dict]:
    """Return the list of entries regardless of whether the file is a bare
    list or the translated `{metadata, data: [...]}` wrapper."""
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("data"), list):
        return obj["data"]
    raise ValueError(f"Unknown dataset shape in {path}: expected list or {{data: [...]}}")


def _clean_alert_wrapper(text: str) -> str:
    """ALERT prompts come as `### Instruction:\\n<body>\\n### Response:\\n`.
    Strip the wrapper so models see the raw request."""
    if ALERT_INSTRUCTION not in text:
        return text
    body = text.split(ALERT_INSTRUCTION, 1)[1]
    if ALERT_RESPONSE in body:
        body = body.split(ALERT_RESPONSE, 1)[0]
    return body.strip()


def _extract_text(item: dict, language: str) -> str:
    raw = item.get(language) or item.get("prompt") or item.get("en") or ""
    return _clean_alert_wrapper(raw.strip())


def load_prompts(
    path: Path,
    language: str = "en",
    limit: int | None = None,
    sample_per_category: int | None = None,
    seed: int = 42,
) -> list[HarmPrompt]:
    """Load prompts from any supported DADA dataset shape.

    - `language` selects the text field (`en`, `pt-br`, etc.); falls back to
      `prompt` then `en` so raw ALERT/AgentHarm files work unchanged.
    - `sample_per_category` does a deterministic stratified sample.
    - `limit` caps the final list (applied after sampling).
    """
    items = _load_raw(path)

    prompts: list[HarmPrompt] = []
    for idx, item in enumerate(items):
        text = _extract_text(item, language)
        if not text:
            continue
        prompts.append(
            HarmPrompt(
                id=str(item.get("id", idx)),
                category=item.get("category", "unknown"),
                text=text,
                language=language,
            )
        )

    if sample_per_category:
        rng = random.Random(seed)
        by_cat: dict[str, list[HarmPrompt]] = defaultdict(list)
        for p in prompts:
            by_cat[p.category].append(p)
        sampled: list[HarmPrompt] = []
        for cat in sorted(by_cat):
            pool = list(by_cat[cat])
            rng.shuffle(pool)
            sampled.extend(pool[:sample_per_category])
        prompts = sampled

    if limit:
        prompts = prompts[:limit]
    return prompts
