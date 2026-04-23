"""Shared JSONL I/O: line-streaming read + bounded-concurrency write.

Used by both the generation runner and the judge so they have the same
streaming / resume / progress behaviour without duplicating the skeleton.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from pathlib import Path
from typing import IO

import orjson


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield parsed objects from a JSONL file, skipping blank/invalid lines."""
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield orjson.loads(line)
            except orjson.JSONDecodeError:
                continue


def load_jsonl(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


async def stream_jsonl_async[T](
    items: Sequence[T],
    work_fn: Callable[[T], Awaitable[dict]],
    out_fh: IO[bytes],
    num_parallel: int,
    label: str,
    *,
    progress_every: int = 20,
    on_result: Callable[[dict], None] | None = None,
) -> None:
    """Run `work_fn` over `items` with bounded concurrency, append each
    returned dict as one JSON line with an immediate flush (crash-resilient),
    and print periodic progress.

    Writes happen in the parent coroutine in finish-order via `as_completed`,
    so no lock is needed. Errors counted via `error` / `judge_error` fields.
    """
    if not items:
        return

    sem = asyncio.Semaphore(num_parallel)

    async def bounded(item: T) -> dict:
        async with sem:
            return await work_fn(item)

    tasks = [asyncio.create_task(bounded(it)) for it in items]
    total = len(tasks)
    t0 = time.monotonic()
    completed = 0
    errors = 0

    try:
        for coro in asyncio.as_completed(tasks):
            payload = await coro
            out_fh.write(orjson.dumps(payload) + b"\n")
            out_fh.flush()
            completed += 1
            if payload.get("error") or payload.get("judge_error"):
                errors += 1
            if on_result is not None:
                on_result(payload)
            if completed % progress_every == 0 or completed == total:
                dt = time.monotonic() - t0
                rate = completed / max(dt, 1e-6)
                eta = (total - completed) / max(rate, 1e-6)
                print(
                    f"    [{label}] {completed}/{total}  "
                    f"{rate:.2f} req/s  errs={errors}  eta={eta:.0f}s"
                )
    except BaseException:
        for t in tasks:
            t.cancel()
        raise
