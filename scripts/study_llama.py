"""Study the llama3.1:8b M-ALERT PT-BR responses file.

The generation pass only sets a heuristic `refused` flag. Before the judge
pass runs we still want a defensible picture of the data: how skewed the
categories are, how confident the heuristic is, which rows are suspicious
(short "answers", long "refusals", ambiguous starts), and what the surface
language of compliant responses looks like per category.

This script reads one JSONL file, stays in the standard library (+ orjson
which is already a project dep), and prints a multi-section report. Pass
`--out-md <path>` to also dump a Markdown version for the thesis notes.

Usage:
    uv run python scripts/study_llama.py output/<file>.jsonl
    uv run python scripts/study_llama.py output/<file>.jsonl --out-md study.md
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median, mean, quantiles
from typing import Iterable

import orjson

# Reuse the exact markers the runner uses as the heuristic, so we can
# measure its behaviour without drift.
from harmcheck.runner import REFUSAL_MARKERS, is_refusal  # type: ignore

# ---------- loading ----------

def load(path: Path) -> list[dict]:
    with open(path, "rb") as fh:
        return [orjson.loads(line) for line in fh if line.strip()]

# ---------- text utilities ----------

_WS = re.compile(r"\s+")
_STOPWORDS_PT = {
    "de","a","o","que","e","do","da","em","um","para","é","com","não","uma",
    "os","no","se","na","por","mais","as","dos","como","mas","foi","ao","ele",
    "das","tem","à","seu","sua","ou","ser","quando","muito","há","nos","já",
    "está","eu","também","só","pelo","pela","até","isso","ela","entre","era",
    "depois","sem","mesmo","aos","ter","seus","suas","numa","pelos","elas",
    "havia","seja","qual","será","nós","tenho","lhe","deles","essas","esses",
    "pelas","este","fosse","dele","tu","te","vocês","vos","lhes","meus","minhas",
    "teu","tua","teus","tuas","nosso","nossa","nossos","nossas","dela","delas",
    "esta","estes","estas","aquele","aquela","aqueles","aquelas","isto","aquilo",
    "estou","está","estamos","estão","estive","esteve","estivemos","estiveram",
    "estava","estávamos","estavam","estivera","estivéramos","estiver","estivermos",
    "estiverem","hei","há","havemos","hão","houve","houvemos","houveram","houvera",
    "houvermos","houverem","houverei","houverá","houveremos","houverão","houveria",
    "houveríamos","houveriam","sou","somos","são","era","éramos","eram","fui",
    "foi","fomos","foram","fora","fôramos","seja","sejamos","sejam","fosse",
    "fôssemos","fossem","for","formos","forem","serei","será","seremos","serão",
    "seria","seríamos","seriam","tenho","tem","temos","tém","tinha","tínhamos",
    "tinham","tive","teve","tivemos","tiveram","tivera","tivéramos","tenha",
    "tenhamos","tenham","tivesse","tivéssemos","tivessem","tiver","tivermos",
    "tiverem","terei","terá","teremos","terão","teria","teríamos","teriam",
    "pode","posso","podem","deve","devem","pra","sobre","então","nem",
    "assim","porque","tudo","isso","aqui","ali","lá","mim","ti","cá","te","me",
    # soft refusal-adjacent noise that otherwise dominates keyword lists
    "desculpe","desculpa","sinto",
}

_QUOTE_TRANSLATION = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))

def tokens(text: str) -> list[str]:
    t = text.translate(_QUOTE_TRANSLATION).lower()
    t = re.sub(r"[^\w\sáéíóúâêôãõàç-]", " ", t, flags=re.UNICODE)
    return [w for w in _WS.split(t) if w and w not in _STOPWORDS_PT and not w.isdigit() and len(w) > 2]

# ---------- formatting ----------

def pct(n: int, d: int) -> str:
    return f"{(n / d * 100 if d else 0):5.1f}%"

def qtiles(xs: list[float], qs=(0.10, 0.50, 0.90, 0.99)) -> dict[str, float]:
    if not xs:
        return {f"p{int(q*100)}": 0.0 for q in qs}
    xs_sorted = sorted(xs)
    # statistics.quantiles divides into equal-probability intervals; we want
    # arbitrary percentiles, so compute directly via interpolation.
    out: dict[str, float] = {}
    n = len(xs_sorted)
    for q in qs:
        if n == 1:
            out[f"p{int(q*100)}"] = xs_sorted[0]
            continue
        pos = q * (n - 1)
        lo, hi = int(pos), min(int(pos) + 1, n - 1)
        frac = pos - lo
        out[f"p{int(q*100)}"] = xs_sorted[lo] * (1 - frac) + xs_sorted[hi] * frac
    return out

# ---------- analyses ----------

def overall(rows: list[dict]) -> list[str]:
    out = []
    errs = [r for r in rows if r.get("error")]
    ok = [r for r in rows if not r.get("error")]
    ref = [r for r in ok if r.get("refused")]
    ans = [r for r in ok if not r.get("refused")]

    lens = [len(r.get("response", "")) for r in ok]
    toks = [r.get("eval_count", 0) for r in ok if r.get("eval_count")]
    tsec = [r.get("elapsed_seconds", 0.0) for r in ok if r.get("elapsed_seconds", 0) > 0]
    tps  = [r["eval_count"] / r["elapsed_seconds"] for r in ok
            if r.get("eval_count") and r.get("elapsed_seconds", 0) > 0]

    out.append(f"rows            : {len(rows)}")
    out.append(f"errors          : {len(errs)} ({pct(len(errs), len(rows))})")
    out.append(f"refused (heur.) : {len(ref)} ({pct(len(ref), len(ok))})")
    out.append(f"answered        : {len(ans)} ({pct(len(ans), len(ok))})")
    ql = qtiles(lens); out.append(f"response chars  : p10={ql['p10']:.0f} p50={ql['p50']:.0f} p90={ql['p90']:.0f} p99={ql['p99']:.0f}")
    qt = qtiles([float(x) for x in toks]); out.append(f"eval_count toks : p10={qt['p10']:.0f} p50={qt['p50']:.0f} p90={qt['p90']:.0f} p99={qt['p99']:.0f}")
    qs = qtiles(tsec); out.append(f"elapsed seconds : p10={qs['p10']:.2f} p50={qs['p50']:.2f} p90={qs['p90']:.2f} p99={qs['p99']:.2f}")
    if tps:
        out.append(f"throughput tok/s: median={median(tps):.1f}  mean={mean(tps):.1f}")
    return out

def per_category(rows: list[dict]) -> list[str]:
    out = []
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if not r.get("error"):
            by[r.get("category", "?")].append(r)

    # Sort by answered-rate descending (i.e. where the model complies most).
    def answered_rate(rs):  # noqa: ANN001
        return sum(1 for r in rs if not r.get("refused")) / len(rs) if rs else 0.0

    ordered = sorted(by.items(), key=lambda kv: answered_rate(kv[1]), reverse=True)

    out.append(f"{'category':<22}{'n':>6}{'ref%':>8}{'ans%':>8}{'len_med':>10}{'tok_med':>10}")
    for cat, rs in ordered:
        n = len(rs)
        nref = sum(1 for r in rs if r.get("refused"))
        nans = n - nref
        lens = [len(r.get("response", "")) for r in rs]
        toks = [r.get("eval_count", 0) for r in rs if r.get("eval_count")]
        out.append(
            f"{cat:<22}{n:>6}{pct(nref, n):>8}{pct(nans, n):>8}"
            f"{(median(lens) if lens else 0):>10.0f}{(median(toks) if toks else 0):>10.0f}"
        )
    return out

def marker_breakdown(rows: list[dict]) -> list[str]:
    """Which refusal markers actually fire, and how often. Tells us whether
    the heuristic is dominated by 2–3 phrases (fragile) or broadly spread."""
    out = []
    counts: Counter[str] = Counter()
    no_marker = 0
    for r in rows:
        if r.get("error") or not r.get("refused"):
            continue
        head = r.get("response", "").translate(_QUOTE_TRANSLATION).lower().lstrip()[:400]
        hits = [m for m in REFUSAL_MARKERS if m in head]
        if not hits:
            no_marker += 1
            continue
        for m in hits:
            counts[m] += 1
    total = sum(counts.values()) or 1
    out.append(f"unique markers fired : {len(counts)} / {len(REFUSAL_MARKERS)}")
    out.append(f"rows flagged refused but no-marker-in-head : {no_marker}")
    out.append("top markers (may overlap within a response):")
    for m, c in counts.most_common(10):
        out.append(f"  {m!r:<32} {c:>6} ({pct(c, total)})")
    return out

def suspicious_rows(rows: list[dict]) -> tuple[list[str], dict[str, list[dict]]]:
    """Triage buckets worth a judge pass / manual review.

    - `short_answered`: refused=False but very short (likely incomplete or meta).
    - `long_refused` : refused=True but long (likely rebuttal-with-lecture; heuristic firing on "não posso concordar" etc., not on actual refusal).
    - `starts_nao`   : response starts with "Não " (normative verdict) but refused=False.
    - `starts_como_ia`: response starts with "Como um modelo/IA" — often compliance-with-disclaimer.
    """
    out = []
    buckets: dict[str, list[dict]] = {
        "short_answered": [],
        "long_refused": [],
        "starts_nao": [],
        "starts_como_ia": [],
    }
    STARTS_NAO = re.compile(r"^\s*n[ãa]o[\s,.:;]", re.IGNORECASE)
    STARTS_COMO_IA = re.compile(r"^\s*como (um|uma)?\s*(modelo|ia|inteligência)", re.IGNORECASE)

    for r in rows:
        if r.get("error"):
            continue
        text = r.get("response", "")
        n = len(text)
        if not r.get("refused") and n < 40:
            buckets["short_answered"].append(r)
        if r.get("refused") and n > 600:
            buckets["long_refused"].append(r)
        if not r.get("refused") and STARTS_NAO.match(text):
            buckets["starts_nao"].append(r)
        if STARTS_COMO_IA.match(text):
            buckets["starts_como_ia"].append(r)

    ok_n = sum(1 for r in rows if not r.get("error")) or 1
    out.append(f"short_answered  (ans, <40ch)  : {len(buckets['short_answered']):>5}  ({pct(len(buckets['short_answered']), ok_n)})")
    out.append(f"long_refused    (ref, >600ch) : {len(buckets['long_refused']):>5}  ({pct(len(buckets['long_refused']), ok_n)})")
    out.append(f"starts_nao      (ans, 'Não…') : {len(buckets['starts_nao']):>5}  ({pct(len(buckets['starts_nao']), ok_n)})")
    out.append(f"starts_como_ia  (disclaimer)  : {len(buckets['starts_como_ia']):>5}  ({pct(len(buckets['starts_como_ia']), ok_n)})")
    return out, buckets

def top_terms_by_category(rows: list[dict], top_n: int = 8, min_cat_n: int = 100) -> list[str]:
    """Crude lexical signal: most distinctive (TF-IDF-lite) tokens among
    *answered* responses per category. Highlights what topics leak through
    when the model complies — a PT-BR proxy for qualitative review."""
    out = []
    by_cat: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if r.get("error") or r.get("refused"):
            continue
        by_cat[r.get("category", "?")].extend(tokens(r.get("response", "")))

    # Document frequency across categories (how many categories use the term).
    df: Counter[str] = Counter()
    tf: dict[str, Counter[str]] = {}
    for cat, toks in by_cat.items():
        if len(toks) < 50:
            continue
        c = Counter(toks)
        tf[cat] = c
        for term in c:
            df[term] += 1
    n_docs = max(len(tf), 1)

    # Score = tf(term, cat) * log(1 + n_docs/df(term)). Keep it simple.
    import math
    for cat in sorted(tf):
        if len(by_cat[cat]) < min_cat_n:
            continue
        scored = [
            (term, c * math.log(1 + n_docs / df[term]))
            for term, c in tf[cat].items()
            if df[term] < n_docs  # drop terms that appear in every category
        ]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        top = ", ".join(t for t, _ in scored[:top_n])
        out.append(f"  {cat:<22} {top}")
    return out

def examples(buckets: dict[str, list[dict]], k: int = 3) -> list[str]:
    out = []
    for name, rs in buckets.items():
        out.append(f"-- {name} (showing {min(k, len(rs))} / {len(rs)}) --")
        for r in rs[:k]:
            pt = r["prompt_text"].replace("\n", " ")[:120]
            rp = r["response"].replace("\n", " ")[:160]
            out.append(f"  [{r['category']}] id={r['prompt_id']}  prompt: {pt}")
            out.append(f"                              reply : {rp}")
        out.append("")
    return out

def heuristic_vs_length_check(rows: list[dict]) -> list[str]:
    """If the heuristic is healthy, refused responses are short (canned
    sentence) and answered ones are much longer. If the distributions overlap,
    the heuristic is likely misclassifying rebuttal-with-lecture as refusal."""
    out = []
    ref_lens = [len(r.get("response", "")) for r in rows
                if not r.get("error") and r.get("refused")]
    ans_lens = [len(r.get("response", "")) for r in rows
                if not r.get("error") and not r.get("refused")]
    qr = qtiles(ref_lens); qa = qtiles(ans_lens)
    out.append("refused   response chars: "
               f"p10={qr['p10']:.0f} p50={qr['p50']:.0f} p90={qr['p90']:.0f} p99={qr['p99']:.0f}")
    out.append("answered  response chars: "
               f"p10={qa['p10']:.0f} p50={qa['p50']:.0f} p90={qa['p90']:.0f} p99={qa['p99']:.0f}")
    overlap = sum(1 for x in ref_lens if x >= qa["p50"]) if ref_lens else 0
    out.append(f"refused rows longer than answered median : {overlap} "
               f"({pct(overlap, len(ref_lens))}) — high = likely heuristic noise")
    return out

# ---------- render ----------

def section(title: str, lines: Iterable[str]) -> str:
    bar = "=" * len(title)
    return f"\n{title}\n{bar}\n" + "\n".join(lines) + "\n"

def render(path: Path, rows: list[dict]) -> str:
    parts: list[str] = []
    parts.append(section("File & overall", [f"path: {path}", *overall(rows)]))
    parts.append(section("Refusal-heuristic marker breakdown", marker_breakdown(rows)))
    parts.append(section("Heuristic sanity: length by label", heuristic_vs_length_check(rows)))
    susp_lines, buckets = suspicious_rows(rows)
    parts.append(section("Suspicious / ambiguous buckets", susp_lines))
    parts.append(section("Per-category breakdown (sorted by answered rate ↓)", per_category(rows)))
    parts.append(section("Top distinctive terms among answered replies, by category",
                         ["(TF-IDF-lite over PT-BR tokens; stopwords + single 'moral'-refusal words dropped)"]
                         + top_terms_by_category(rows)))
    parts.append(section("Bucket examples", examples(buckets, k=3)))
    return "".join(parts)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, help="JSONL produced by `harmcheck run`")
    ap.add_argument("--out-md", type=Path, default=None, help="Also write the report to this file")
    args = ap.parse_args()

    rows = load(args.path)
    report = render(args.path, rows)
    print(report)
    if args.out_md:
        args.out_md.write_text(report, encoding="utf-8")
        print(f"\nwrote {args.out_md}")

if __name__ == "__main__":
    main()
