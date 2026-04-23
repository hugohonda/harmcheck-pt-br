from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import orjson

JUDGE_LABELS = ("REFUSAL", "UNSAFE", "SAFE", "OTHER")


def load_responses(path: Path) -> list[dict]:
    items: list[dict] = []
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(orjson.loads(line))
    return items


def _fmt_pct(n: int, total: int) -> str:
    return f"{(n / total * 100 if total else 0):5.1f}%"


def _has_judge(items: list[dict]) -> bool:
    return any(i.get("judge_label") for i in items)


def _row_marker(row: dict, judged: bool) -> str:
    if judged:
        return row.get("judge_label") or "?"
    return "REFUSED" if row.get("refused") else "ANSWERED"


def summarize(path: Path) -> None:
    items = load_responses(path)
    if not items:
        print(f"{path}: empty")
        return

    total = len(items)
    errors = [i for i in items if i.get("error")]
    ok = [i for i in items if not i.get("error")]
    judged = _has_judge(ok)

    print(f"File         : {path}")
    print(f"Responses    : {total}")
    print(f"Errors       : {len(errors)} ({_fmt_pct(len(errors), total)})")
    print(f"Labelling    : {'LLM-as-judge' if judged else 'heuristic (refused field)'}")

    refusals_h = [i for i in ok if i.get("refused")]
    print(f"Refusals (heur): {len(refusals_h)} ({_fmt_pct(len(refusals_h), len(ok))})")
    if judged:
        counts = Counter(i.get("judge_label") for i in ok)
        for lbl in JUDGE_LABELS:
            print(f"  judge {lbl:<8}: {counts.get(lbl, 0):>6} ({_fmt_pct(counts.get(lbl, 0), len(ok))})")
    print()

    # Per-model summary
    by_model: dict[str, list[dict]] = defaultdict(list)
    for i in items:
        by_model[i["model"]].append(i)

    if judged:
        print("Per model (judge labels):")
        print(
            f"  {'model':<22}{'n':>6}{'err%':>7}"
            f"{'REFUSAL%':>10}{'UNSAFE%':>10}{'SAFE%':>9}{'OTHER%':>9}"
            f"{'len_med':>9}{'t_med':>8}{'tok/s':>8}"
        )
        for m in sorted(by_model):
            rs = by_model[m]
            m_ok = [r for r in rs if not r.get("error")]
            m_err = [r for r in rs if r.get("error")]
            counts = Counter(r.get("judge_label") for r in m_ok)
            n_ok = len(m_ok) or 1
            lens = [len(r.get("response", "")) for r in m_ok]
            times = [r.get("elapsed_seconds", 0.0) for r in m_ok if r.get("elapsed_seconds", 0) > 0]
            toks_per_s = [
                (r["eval_count"] / r["elapsed_seconds"])
                for r in m_ok
                if r.get("eval_count") and r.get("elapsed_seconds", 0) > 0
            ]
            print(
                f"  {m:<22}{len(rs):>6}{_fmt_pct(len(m_err), len(rs)):>7}"
                f"{_fmt_pct(counts.get('REFUSAL', 0), n_ok):>10}"
                f"{_fmt_pct(counts.get('UNSAFE', 0), n_ok):>10}"
                f"{_fmt_pct(counts.get('SAFE', 0), n_ok):>9}"
                f"{_fmt_pct(counts.get('OTHER', 0), n_ok):>9}"
                f"{(median(lens) if lens else 0):>9.0f}"
                f"{(median(times) if times else 0):>8.2f}"
                f"{(median(toks_per_s) if toks_per_s else 0):>8.1f}"
            )
    else:
        print("Per model (heuristic):")
        print(f"  {'model':<22}{'n':>6}{'err%':>8}{'ref%':>8}{'len_med':>10}{'t_med':>10}{'tok/s':>9}")
        for m in sorted(by_model):
            rs = by_model[m]
            m_ok = [r for r in rs if not r.get("error")]
            m_err = [r for r in rs if r.get("error")]
            m_ref = [r for r in m_ok if r.get("refused")]
            lens = [len(r.get("response", "")) for r in m_ok]
            times = [r.get("elapsed_seconds", 0.0) for r in m_ok if r.get("elapsed_seconds", 0) > 0]
            toks_per_s = [
                (r["eval_count"] / r["elapsed_seconds"])
                for r in m_ok
                if r.get("eval_count") and r.get("elapsed_seconds", 0) > 0
            ]
            print(
                f"  {m:<22}{len(rs):>6}"
                f"{_fmt_pct(len(m_err), len(rs)):>8}"
                f"{_fmt_pct(len(m_ref), len(m_ok)):>8}"
                f"{(median(lens) if lens else 0):>10.0f}"
                f"{(median(times) if times else 0):>10.2f}"
                f"{(median(toks_per_s) if toks_per_s else 0):>9.1f}"
            )
    print()

    # Model x category grid
    key = "judge_label" if judged else None
    grid_title = "UNSAFE rate" if judged else "Refusal rate (heuristic)"
    print(f"{grid_title} by model x category  (ok responses only; N in parens):")
    categories = sorted({i.get("category", "") for i in ok})
    models = sorted(by_model)

    col_w = 18
    header = "category".ljust(24) + "".join(m[: col_w - 1].rjust(col_w) for m in models)
    print("  " + header)
    for cat in categories:
        row = cat[:23].ljust(24)
        for m in models:
            cell = [r for r in by_model[m] if r.get("category") == cat and not r.get("error")]
            if not cell:
                row += "—".rjust(col_w)
                continue
            hits = (
                sum(1 for r in cell if r.get(key) == "UNSAFE")
                if key
                else sum(1 for r in cell if r.get("refused"))
            )
            rate = hits / len(cell)
            row += f"{rate * 100:5.1f}% ({len(cell):>3})".rjust(col_w)
        print("  " + row)
    print()

    # Cross-model agreement
    by_prompt: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in ok:
        by_prompt[r["prompt_id"]][r["model"]] = r

    if judged:
        n_multi = 0
        all_refusal = 0
        any_unsafe = 0
        disagreement = 0
        for _pid, per_model in by_prompt.items():
            if len(per_model) < 2:
                continue
            labels = [r.get("judge_label") for r in per_model.values()]
            n_multi += 1
            if all(label == "REFUSAL" for label in labels):
                all_refusal += 1
            if any(label == "UNSAFE" for label in labels):
                any_unsafe += 1
            if len(set(labels)) > 1:
                disagreement += 1
        if n_multi:
            print("Cross-model agreement (prompts answered by all models):")
            print(f"  all REFUSAL        : {all_refusal:>5} ({_fmt_pct(all_refusal, n_multi)})")
            print(f"  any UNSAFE         : {any_unsafe:>5} ({_fmt_pct(any_unsafe, n_multi)})")
            print(f"  label disagreement : {disagreement:>5} ({_fmt_pct(disagreement, n_multi)})")
    else:
        disagreements = 0
        full_refusals = 0
        zero_refusals = 0
        for _pid, per_model in by_prompt.items():
            if len(per_model) < 2:
                continue
            flags = {m: r.get("refused", False) for m, r in per_model.items()}
            if all(flags.values()):
                full_refusals += 1
            elif not any(flags.values()):
                zero_refusals += 1
            else:
                disagreements += 1
        n_multi = full_refusals + zero_refusals + disagreements
        if n_multi:
            print("Cross-model behaviour (prompts answered by all models):")
            print(f"  all refused        : {full_refusals:>5} ({_fmt_pct(full_refusals, n_multi)})")
            print(f"  none refused       : {zero_refusals:>5} ({_fmt_pct(zero_refusals, n_multi)})")
            print(f"  disagreement       : {disagreements:>5} ({_fmt_pct(disagreements, n_multi)})")


def disagreements(path: Path, limit: int = 20) -> None:
    """Prompts where models disagree — on judge_label if present, else on
    the `refused` heuristic. Useful for qualitative inspection."""
    items = [i for i in load_responses(path) if not i.get("error")]
    judged = _has_judge(items)

    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for r in items:
        by_prompt[r["prompt_id"]].append(r)

    rows: list[tuple[str, list[dict]]] = []
    for pid, rs in by_prompt.items():
        if len(rs) < 2:
            continue
        if judged:
            labels = {r.get("judge_label") for r in rs}
            if len(labels) > 1:
                rows.append((pid, rs))
        else:
            flags = {r.get("refused", False) for r in rs}
            if len(flags) > 1:
                rows.append((pid, rs))

    tag = "judge label" if judged else "refusal heuristic"
    print(f"{len(rows)} prompt(s) with cross-model disagreement on {tag}.")
    for pid, rs in rows[:limit]:
        first = rs[0]
        print()
        print(f"--- id={pid}  category={first['category']}  lang={first['language']}")
        print(f"    prompt: {first['prompt_text'][:160]}")
        for r in sorted(rs, key=lambda x: x["model"]):
            marker = _row_marker(r, judged)
            snippet = r.get("response", "").replace("\n", " ")[:140]
            print(f"    [{marker:<8}] {r['model']:<22} {snippet}")
