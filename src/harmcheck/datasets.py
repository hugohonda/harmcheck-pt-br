import json
import random
from collections import defaultdict
from pathlib import Path

from .config import settings
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


def load_quality_scores(path: Path) -> dict[str, float]:
    """Load `{id -> score}` from a DADA-pt-br 02-evaluated sidecar."""
    with open(path) as f:
        obj = json.load(f)
    rows = obj["data"] if isinstance(obj, dict) and "data" in obj else obj
    return {str(r["id"]): float(r["score"]) for r in rows if "score" in r}


def discover_quality_file(dataset_path: Path) -> Path | None:
    """Find the 02-evaluated sidecar matching the dataset by its substantive
    name (timestamp prefix stripped, since translation and evaluation runs
    have different pipeline_ids). Returns None if ambiguous or missing.
    """
    eval_dir = settings.evaluated_dir
    if not eval_dir.exists():
        return None
    # Strip leading `YYYYMMDD_HHMMSSZ_` timestamp prefix, match the rest.
    parts = dataset_path.stem.split("_")
    core = "_".join(parts[2:]) if len(parts) >= 3 else dataset_path.stem
    if not core:
        return None
    candidates = [p for p in eval_dir.glob("*_evaluated.json") if core in p.stem]
    if len(candidates) == 1:
        return candidates[0]
    return None


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
    min_quality_score: float | None = None,
    quality_file: Path | None = None,
) -> list[HarmPrompt]:
    """Load prompts from any supported DADA dataset shape.

    - `language` selects the text field (`en`, `pt-br`, etc.); falls back to
      `prompt` then `en` so raw ALERT/AgentHarm files work unchanged.
    - `min_quality_score` drops rows whose MetricX translation score in the
      `02-evaluated` sidecar is below the threshold. Rows missing a score
      are also dropped (conservative). If `quality_file` isn't given, tries
      to auto-discover it next to the dataset.
    - `sample_per_category` does a deterministic stratified sample.
    - `limit` caps the final list (applied after filter + sampling).
    """
    items = _load_raw(path)

    quality_map: dict[str, float] | None = None
    if min_quality_score is not None:
        qf = quality_file or discover_quality_file(path)
        if qf is None:
            raise FileNotFoundError(
                f"--min-quality-score set but no evaluated sidecar found for {path}. "
                f"Pass --quality-file explicitly or check ETHIC_EVALUATED_DIR."
            )
        quality_map = load_quality_scores(qf)

    threshold = min_quality_score if min_quality_score is not None else float("-inf")
    prompts: list[HarmPrompt] = []
    dropped_quality = 0
    for idx, item in enumerate(items):
        text = _extract_text(item, language)
        if not text:
            continue
        row_id = str(item.get("id", idx))
        if quality_map is not None:
            score = quality_map.get(row_id)
            if score is None or score < threshold:
                dropped_quality += 1
                continue
        prompts.append(
            HarmPrompt(
                id=row_id,
                category=item.get("category", "unknown"),
                text=text,
                language=language,
            )
        )

    if min_quality_score is not None:
        kept = len(prompts)
        total = kept + dropped_quality
        print(f"quality filter (score >= {min_quality_score}): kept {kept}/{total}, dropped {dropped_quality}")

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
