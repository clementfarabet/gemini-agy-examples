"""Parallel Sandbox Forking: Best-of-N Code & Benchmark Arena.

Demonstrates one of the most powerful primitives of the Antigravity Managed Agent API:
instant container filesystem branching (`client.environments.create(from_environment=...)`)
combined with concurrent multi-agent execution (`ThreadPoolExecutor`).

Pipeline:
1. Provision & Seed Base Sandbox:
   Creates a base Linux sandbox and seeds `/workspace/baseline.py` (a naive reference
   implementation) and `/workspace/benchmark.py` (a strict correctness + latency harness).
   If a custom `--task` is provided, an initial agent turn generates the baseline & harness.
2. Instant Filesystem Forking (`from_environment`):
   Clones the exact filesystem state of the base sandbox into N isolated child sandboxes.
3. Parallel Agent Swarm:
   Dispatches N agents concurrently across the N forked sandboxes, each pursuing a distinct
   optimization strategy, running `benchmark.py` inside its container to verify 100% output
   equivalence and record `/workspace/metrics.json`.
4. Automated Judge & Minimalist Scorecard (`arena_report.html` + `winning_solution.py`):
   Downloads the verified metrics and code from all N sandboxes, crowns the fastest passing
   implementation, exports the winning code and a lean, minimalist HTML comparison report,
   and cleans up all N+1 sandboxes.

Usage:
    export GEMINI_API_KEY="..."

    # Run the default 3-way parallel optimization arena
    uv run python run_parallel_arena.py

    # Run with Gemini 4 Argon and open the resulting scorecard
    uv run python run_parallel_arena.py --argon --open

    # Run a custom optimization challenge across 3 forked sandboxes
    uv run python run_parallel_arena.py --task "Fastest Levenshtein edit distance batch matcher for 20,000 strings"
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import html as html_lib
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any
import warnings
import webbrowser

from google import genai
from google.genai import errors

warnings.filterwarnings("ignore", message=".*usage is experimental.*")

DEFAULT_AGENT = "antigravity-preview-09-2026"
DEFAULT_MODEL_LABEL = "gemini-3.8-flash (agent default)"
ARGON_MODEL = "gemini-4-argon"
DEFAULT_REPORT_PATH = Path("arena_report.html")
DEFAULT_WINNER_PATH = Path("winning_solution.py")

DEFAULT_CHALLENGE_TITLE = "Rolling 60s VWAP & Trade Anomaly Engine (50,000 Trades)"

SEED_BASELINE_PY = '''"""Baseline reference implementation for the Rolling 60s VWAP & Trade Anomaly Engine."""

from __future__ import annotations


def solve(
    timestamps_ms: list[int],
    symbol_ids: list[int],
    prices: list[float],
    volumes: list[float],
    window_ms: int = 60_000,
    top_k: int = 50,
) -> tuple[list[tuple[int, int, float, float]], dict[int, float]]:
    """Compute rolling window_ms VWAP per symbol and return top_k price-deviation anomalies.

    Args:
        timestamps_ms: Monotonically non-decreasing trade timestamps in milliseconds (len N).
        symbol_ids: Integer symbol ID for each trade (len N).
        prices: Trade execution price for each trade (len N).
        volumes: Trade volume for each trade (len N).
        window_ms: Rolling lookback window in milliseconds (inclusive: [t - window_ms, t]).
        top_k: Number of highest relative price-deviation trades to return.

    Returns:
        A tuple (anomalies, symbol_notional_totals):
        - anomalies: list of `top_k` tuples `(trade_idx, symbol_id, round(vwap, 6), round(z_score, 6))`
          sorted by unrounded `z_score = abs(price - vwap) / vwap` descending (tie-broken by `trade_idx` ascending).
        - symbol_notional_totals: dict mapping `symbol_id -> round(sum(price * volume), 4)` across all trades.
    """
    n = len(timestamps_ms)
    history_by_symbol: dict[int, list[tuple[int, float, float]]] = {}
    symbol_notional_totals: dict[int, float] = {}
    scored_trades: list[tuple[float, int, int, float]] = []

    for i in range(n):
        ts = timestamps_ms[i]
        sym = symbol_ids[i]
        px = prices[i]
        vol = volumes[i]
        notional = px * vol

        symbol_notional_totals[sym] = symbol_notional_totals.get(sym, 0.0) + notional
        sym_hist = history_by_symbol.setdefault(sym, [])
        sym_hist.append((ts, notional, vol))

        cutoff = ts - window_ms
        win_notional = 0.0
        win_vol = 0.0
        for h_ts, h_notional, h_vol in sym_hist:
            if h_ts >= cutoff:
                win_notional += h_notional
                win_vol += h_vol

        vwap = win_notional / win_vol if win_vol > 0.0 else px
        z_score = abs(px - vwap) / vwap if vwap > 0.0 else 0.0
        scored_trades.append((-z_score, i, sym, vwap))

    scored_trades.sort()
    anomalies = [
        (idx, sym, round(vwap, 6), round(-neg_z, 6))
        for neg_z, idx, sym, vwap in scored_trades[:top_k]
    ]
    rounded_totals = {sym: round(tot, 4) for sym, tot in symbol_notional_totals.items()}
    return anomalies, rounded_totals
'''

SEED_BENCHMARK_PY = '''"""Deterministic correctness & latency benchmark harness for the Parallel Sandbox Arena."""

from __future__ import annotations

import json
from pathlib import Path
import random
import statistics
import sys
import time

import baseline
import solution


def generate_dataset(n: int = 50_000, num_symbols: int = 200, seed: int = 42):
    rng = random.Random(seed)
    timestamps_ms: list[int] = []
    symbol_ids: list[int] = []
    prices: list[float] = []
    volumes: list[float] = []

    base_prices = {s: rng.uniform(20.0, 500.0) for s in range(num_symbols)}
    cur_ts = 1_700_000_000_000

    for i in range(n):
        cur_ts += rng.randint(0, 45)
        sym = rng.randrange(num_symbols)
        # Occasional sharp price spike to create realistic anomalies
        if rng.random() < 0.004:
            mult = rng.choice([0.72, 0.80, 1.22, 1.35])
        else:
            mult = 1.0 + rng.gauss(0.0, 0.008)
        px = round(max(1.0, base_prices[sym] * mult), 4)
        base_prices[sym] = max(5.0, base_prices[sym] * (1.0 + rng.gauss(0.0, 0.0005)))
        vol = round(rng.uniform(1.0, 250.0), 2)

        timestamps_ms.append(cur_ts)
        symbol_ids.append(sym)
        prices.append(px)
        volumes.append(vol)

    return timestamps_ms, symbol_ids, prices, volumes


def verify_equivalence(ref_out, cand_out) -> tuple[bool, str]:
    ref_anom, ref_tot = ref_out
    cand_anom, cand_tot = cand_out

    if len(ref_anom) != len(cand_anom):
        return False, f"Anomaly count mismatch: expected {len(ref_anom)}, got {len(cand_anom)}"

    for rank, (r_item, c_item) in enumerate(zip(ref_anom, cand_anom)):
        r_idx, r_sym, r_vwap, r_z = r_item
        c_idx, c_sym, c_vwap, c_z = c_item
        if r_idx != c_idx or r_sym != c_sym:
            return False, f"Rank {rank} mismatch: expected trade ({r_idx}, sym={r_sym}), got ({c_idx}, sym={c_sym})"
        if abs(r_vwap - c_vwap) > 1e-4 or abs(r_z - c_z) > 1e-4:
            return False, f"Rank {rank} float drift: expected ({r_vwap}, {r_z}), got ({c_vwap}, {c_z})"

    if set(ref_tot.keys()) != set(cand_tot.keys()):
        return False, "Symbol keys mismatch in notional totals"

    for sym, r_val in ref_tot.items():
        c_val = cand_tot[sym]
        if abs(r_val - c_val) > 1e-2:
            return False, f"Notional total drift for symbol {sym}: expected {r_val}, got {c_val}"

    return True, "100% exact match on top-50 anomalies and 200 symbol totals"


def main() -> int:
    data = generate_dataset()

    # Measure baseline
    t0 = time.perf_counter()
    ref_out = baseline.solve(*data)
    baseline_ms = (time.perf_counter() - t0) * 1000.0

    # Warmup & verify candidate solution
    try:
        cand_out = solution.solve(*data)
    except Exception as exc:
        metrics = {
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "baseline_ms": round(baseline_ms, 3),
            "solution_ms": None,
            "speedup_x": 0.0,
        }
        Path("/workspace/metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2))
        return 1

    ok, detail = verify_equivalence(ref_out, cand_out)
    if not ok:
        metrics = {
            "passed": False,
            "error": detail,
            "baseline_ms": round(baseline_ms, 3),
            "solution_ms": None,
            "speedup_x": 0.0,
        }
        Path("/workspace/metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2))
        return 1

    # Timed benchmark runs (5 iterations, take median)
    timings_ms: list[float] = []
    for _ in range(5):
        t_start = time.perf_counter()
        solution.solve(*data)
        timings_ms.append((time.perf_counter() - t_start) * 1000.0)

    median_ms = statistics.median(timings_ms)
    min_ms = min(timings_ms)
    speedup = baseline_ms / median_ms if median_ms > 0 else 0.0

    metrics = {
        "passed": True,
        "verification": detail,
        "baseline_ms": round(baseline_ms, 3),
        "solution_ms": round(median_ms, 3),
        "min_ms": round(min_ms, 3),
        "speedup_x": round(speedup, 2),
    }
    Path("/workspace/metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


@dataclass(frozen=True)
class StrategySpec:
    """Defines a competing optimization strategy assigned to a forked sandbox."""

    code: str
    name: str
    instructions: str


DEFAULT_STRATEGIES: tuple[StrategySpec, ...] = (
    StrategySpec(
        code="FORK-01",
        name="Two-Pointer Sliding Window",
        instructions=(
            "Optimize `/workspace/solution.py` using a pure-Python monotonic two-pointer "
            "sliding window per symbol (maintaining running window notional and volume accumulators "
            "and advancing the left index as timestamps fall outside `ts - window_ms`), combined "
            "with `heapq.nsmallest` / `heapq.nlargest` for top-K selection."
        ),
    ),
    StrategySpec(
        code="FORK-02",
        name="NumPy Vectorized Prefix-Sums",
        instructions=(
            "Optimize `/workspace/solution.py` using NumPy vectorization: group or sort by symbol, "
            "use cumulative prefix sums (`np.cumsum`) and vectorized binary search (`np.searchsorted`) "
            "or C-level array operations (`np.bincount`, `np.argpartition`) to eliminate Python loop overhead."
        ),
    ),
    StrategySpec(
        code="FORK-03",
        name="Flat-Array & Deque Accumulator",
        instructions=(
            "Optimize `/workspace/solution.py` using pre-allocated flat symbol buckets (indexed `0..max_sym` "
            "lists/arrays instead of dicts), incremental sliding window state pointers, local variable binding, "
            "and partial selection so memory allocations and hash lookups are minimized."
        ),
    ),
)


@dataclass
class ContenderResult:
    """Stores the outcome of a single forked sandbox contender."""

    strategy: StrategySpec
    environment_id: str
    interaction_id: str | None = None
    passed: bool = False
    baseline_ms: float | None = None
    solution_ms: float | None = None
    min_ms: float | None = None
    speedup_x: float = 0.0
    verification: str = ""
    solution_code: str = ""
    total_tokens: int = 0
    elapsed_s: float = 0.0


class Logger:
    """Thread-safe elapsed-time terminal logger."""

    def __init__(self) -> None:
        self._start = time.monotonic()
        self._lock = threading.Lock()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def log(self, icon: str, message: str, prefix: str = "") -> None:
        tag = f"[{prefix}] " if prefix else ""
        with self._lock:
            print(f"[{self.elapsed:5.1f}s] {icon} {tag}{message}", flush=True)


def summarize_one_line(text: str, max_len: int = 78) -> str:
    """Collapse multi-line text into a concise single-line summary."""
    lines = [line.strip().strip("*#").strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    combined = " — ".join(lines) if len(lines) > 1 else lines[0]
    combined = " ".join(combined.split())
    if len(combined) > max_len:
        return combined[: max_len - 3].rstrip() + "..."
    return combined


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Fork a base Antigravity sandbox into N isolated child sandboxes, race competing "
            "optimization strategies in parallel, and export the winning code + HTML scorecard."
        )
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help=(
            "Optional custom optimization challenge description. If omitted, uses the built-in "
            f"'{DEFAULT_CHALLENGE_TITLE}' benchmark."
        ),
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model",
        type=str,
        default=None,
        help="Underlying Gemini model for the agents (e.g., gemini-4-argon, gemini-3.8-flash).",
    )
    model_group.add_argument(
        "--argon",
        action="store_true",
        help=f"Shorthand to run with Gemini 4 Argon (--model {ARGON_MODEL}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"Local path for the HTML arena scorecard (default: {DEFAULT_REPORT_PATH}).",
    )
    parser.add_argument(
        "--winner-out",
        type=Path,
        default=DEFAULT_WINNER_PATH,
        help=f"Local path to save the winning Python implementation (default: {DEFAULT_WINNER_PATH}).",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Automatically open the generated HTML arena scorecard in your browser.",
    )
    parser.add_argument(
        "--keep-env",
        action="store_true",
        help="Preserve the remote sandboxes on exit instead of deleting them.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose agent thought summaries and final summary table.",
    )
    return parser.parse_args(argv)


def stream_agent_turn(
    client: genai.Client,
    logger: Logger,
    *,
    prompt: str,
    environment_id: str,
    model: str | None = None,
    prefix: str = "",
    verbose: bool = False,
) -> tuple[str | None, int]:
    """Run an agent turn inside a specific sandbox and log single-line steps."""
    kwargs: dict[str, Any] = {
        "agent": DEFAULT_AGENT,
        "input": prompt,
        "environment": environment_id,
        "stream": True,
    }
    if model:
        kwargs["agent_config"] = {"type": "antigravity", "model": model}

    events = client.interactions.create(**kwargs)
    interaction_id: str | None = None
    total_tokens = 0

    for event in events:
        event_type = getattr(event, "event_type", "")

        if event_type == "interaction.created":
            interaction = getattr(event, "interaction", None)
            if interaction:
                interaction_id = getattr(interaction, "id", interaction_id)

        elif event_type == "step.start":
            step = getattr(event, "step", None)
            step_type = getattr(step, "type", "")
            args = getattr(step, "arguments", None)
            if step_type == "code_execution_call":
                cmd = getattr(args, "command", getattr(args, "code", "")) if args else ""
                snippet = summarize_one_line(str(cmd), max_len=64)
                logger.log("⚡", f"Exec: {snippet}" if snippet else "Code execution", prefix=prefix)
            elif step_type == "function_call":
                fn_name = getattr(step, "name", "") or "tool"
                logger.log("🔧", f"Tool: {fn_name}", prefix=prefix)
            elif step_type == "model_output":
                logger.log("📝", "Synthesizing result...", prefix=prefix)

        elif event_type == "step.delta":
            delta = getattr(event, "delta", None)
            if getattr(delta, "type", "") == "thought_summary":
                content = getattr(delta, "content", None)
                thought_text = getattr(content, "text", "") if content else ""
                if thought_text:
                    msg = thought_text.strip() if verbose else summarize_one_line(thought_text, max_len=72)
                    logger.log("💡", f"Thought: {msg}", prefix=prefix)

        elif event_type == "step.stop":
            usage = getattr(event, "usage", None)
            if usage and getattr(usage, "total_tokens", None):
                total_tokens = int(usage.total_tokens)

        elif event_type == "interaction.completed":
            interaction = getattr(event, "interaction", None)
            if interaction:
                interaction_id = getattr(interaction, "id", interaction_id)
                usage = getattr(interaction, "usage", None)
                if usage and getattr(usage, "total_tokens", None):
                    total_tokens = int(usage.total_tokens)
            logger.log("✓", f"Completed (tokens: {total_tokens:,})", prefix=prefix)

        elif event_type == "error":
            err = getattr(event, "error", None)
            raise RuntimeError(f"Stream error: {getattr(err, 'message', err)}")

    return interaction_id, total_tokens


def seed_base_sandbox(
    client: genai.Client,
    logger: Logger,
    *,
    custom_task: str | None,
    model: str | None,
    verbose: bool,
) -> tuple[str, list[StrategySpec]]:
    """Create the base sandbox and populate `/workspace/baseline.py` and `/workspace/benchmark.py`."""
    base_env = client.environments.create()
    base_env_id = getattr(base_env, "id", None)
    if not base_env_id:
        raise RuntimeError("Failed to provision base sandbox environment.")
    logger.log("🚀", f"Provisioned base sandbox: {base_env_id}")

    if not custom_task:
        # Fast direct file seed (~1s)
        for fname, content in (
            ("workspace/baseline.py", SEED_BASELINE_PY),
            ("workspace/solution.py", SEED_BASELINE_PY),
            ("workspace/benchmark.py", SEED_BENCHMARK_PY),
        ):
            client.environments.files.upload(
                environment=base_env_id,
                path=fname,
                file=content.encode("utf-8"),
                mime_type="text/x-python",
                overwrite=True,
            )
        logger.log(
            "📦",
            "Seeded /workspace/baseline.py & /workspace/benchmark.py in base sandbox",
        )
        return base_env_id, list(DEFAULT_STRATEGIES)

    # Custom task: use an agent turn in the base sandbox to generate baseline.py and benchmark.py
    logger.log("🧪", f"Generating custom baseline & benchmark harness for: '{custom_task}'")
    setup_prompt = (
        f"Create a deterministic Python optimization benchmark in `/workspace` for: {custom_task}\n\n"
        "Requirements:\n"
        "1. Write `/workspace/baseline.py` containing a clear, working, deliberately unoptimized "
        "`solve(*args)` function.\n"
        "2. Copy `/workspace/baseline.py` to `/workspace/solution.py` initially.\n"
        "3. Write `/workspace/benchmark.py` that generates a deterministic dataset (taking ~0.3s-1.0s "
        "on baseline), runs `baseline.solve` and `solution.solve`, verifies exact equivalence, "
        "benchmarks `solution.solve` across 5 runs (median ms), and writes `/workspace/metrics.json` "
        "with keys: `passed` (bool), `verification` (str), `baseline_ms` (float), `solution_ms` (float), "
        "`min_ms` (float), `speedup_x` (float).\n"
        "4. Run `python3 /workspace/benchmark.py` to confirm `/workspace/metrics.json` is created."
    )
    stream_agent_turn(
        client,
        logger,
        prompt=setup_prompt,
        environment_id=base_env_id,
        model=model,
        prefix="BASE",
        verbose=verbose,
    )
    custom_strategies = [
        StrategySpec(
            code="FORK-01",
            name="Algorithmic & Data-Structure Redesign",
            instructions=(
                "Optimize `/workspace/solution.py` by improving asymptotic complexity, "
                "using optimal data structures, indexing, or pruning redundant work."
            ),
        ),
        StrategySpec(
            code="FORK-02",
            name="Vectorized / Array-Oriented Pipeline",
            instructions=(
                "Optimize `/workspace/solution.py` using NumPy vectorization, contiguous arrays, "
                "or C-backed stdlib primitives (`itertools`, `bisect`, `heapq`, `array`) to remove Python loop overhead."
            ),
        ),
        StrategySpec(
            code="FORK-03",
            name="Precomputation & Memory-Locality Optimization",
            instructions=(
                "Optimize `/workspace/solution.py` via aggressive precomputation, flat lookup tables, "
                "branch elimination, and local variable binding."
            ),
        ),
    ]
    return base_env_id, custom_strategies


def download_sandbox_text(client: genai.Client, env_id: str, filename: str) -> str | None:
    """Download a text file from `/workspace/<filename>` in the given sandbox."""
    for candidate in (f"workspace/{filename}", filename):
        try:
            raw = client.environments.files.download(environment=env_id, path=candidate)
            return raw.decode("utf-8", errors="replace")
        except errors.APIError:
            continue
    return None


def run_contender(
    client: genai.Client,
    logger: Logger,
    *,
    strategy: StrategySpec,
    fork_env_id: str,
    model: str | None,
    verbose: bool,
) -> ContenderResult:
    """Execute a single optimization strategy inside its isolated forked sandbox."""
    prefix = f"{strategy.code} | {strategy.name[:22]}"
    t0 = time.monotonic()

    prompt = (
        "You are competing in a performance optimization arena inside a forked Linux sandbox.\n"
        "`/workspace/baseline.py` contains the reference `solve(...)` implementation, and "
        "`/workspace/benchmark.py` verifies exact output equivalence and writes `/workspace/metrics.json`.\n\n"
        f"Your Assigned Strategy ({strategy.code} — {strategy.name}):\n"
        f"{strategy.instructions}\n\n"
        "Instructions:\n"
        "1. Inspect `/workspace/baseline.py` and `/workspace/benchmark.py`.\n"
        "2. Write your optimized implementation to `/workspace/solution.py` (keep the exact `solve` signature "
        "and return format; do NOT modify `baseline.py` or `benchmark.py`).\n"
        "3. Run `python3 /workspace/benchmark.py` to verify `"
        '"passed": true` and measure `speedup_x` in `/workspace/metrics.json`.\n'
        "4. If verification fails or speedup can be pushed further, iterate until `/workspace/metrics.json` "
        "shows `passed: true` with maximum speedup."
    )

    interaction_id, tokens = stream_agent_turn(
        client,
        logger,
        prompt=prompt,
        environment_id=fork_env_id,
        model=model,
        prefix=prefix,
        verbose=verbose,
    )
    elapsed_s = time.monotonic() - t0

    metrics_raw = download_sandbox_text(client, fork_env_id, "metrics.json")
    solution_code = download_sandbox_text(client, fork_env_id, "solution.py") or ""

    result = ContenderResult(
        strategy=strategy,
        environment_id=fork_env_id,
        interaction_id=interaction_id,
        solution_code=solution_code,
        total_tokens=tokens,
        elapsed_s=elapsed_s,
    )

    if metrics_raw:
        try:
            data = json.loads(metrics_raw)
            result.passed = bool(data.get("passed", False))
            result.baseline_ms = data.get("baseline_ms")
            result.solution_ms = data.get("solution_ms")
            result.min_ms = data.get("min_ms")
            result.speedup_x = float(data.get("speedup_x") or 0.0)
            result.verification = str(data.get("verification") or data.get("error") or "")
        except json.JSONDecodeError:
            result.verification = "Invalid metrics.json format"
    else:
        result.verification = "metrics.json not found in sandbox"

    status_icon = "🏆" if result.passed else "⚠️"
    logger.log(
        status_icon,
        f"Result: passed={result.passed}, median={result.solution_ms}ms "
        f"(baseline={result.baseline_ms}ms, speedup={result.speedup_x:.2f}x)",
        prefix=prefix,
    )
    return result


def build_arena_report_html(
    *,
    title: str,
    base_env_id: str,
    model_display: str,
    results: list[ContenderResult],
    winner: ContenderResult,
    total_elapsed_s: float,
) -> str:
    """Render a lean, minimalist, high-precision HTML scorecard comparing all forked sandboxes."""
    baseline_ms = next((r.baseline_ms for r in results if r.baseline_ms), 1000.0) or 1000.0
    max_speedup = max((r.speedup_x for r in results), default=1.0) or 1.0

    cards_html = []
    rows_html = []
    code_tabs_html = []
    code_panes_html = []

    for idx, res in enumerate(results):
        is_winner = res.environment_id == winner.environment_id and res.passed
        badge = (
            '<span class="pill pill-winner">WINNER</span>'
            if is_winner
            else ('<span class="pill">VERIFIED</span>' if res.passed else '<span class="pill pill-fail">FAILED</span>')
        )
        sol_ms_str = f"{res.solution_ms:.2f} ms" if res.solution_ms is not None else "N/A"
        base_ms_str = f"{res.baseline_ms:.2f} ms" if res.baseline_ms is not None else "N/A"
        speedup_str = f"{res.speedup_x:.2f}x" if res.passed else "0.00x"
        bar_pct = min(100.0, max(6.0, (res.speedup_x / max_speedup) * 100.0)) if res.passed else 4.0

        cards_html.append(f"""
        <div class="card {'card-winner' if is_winner else ''}">
          <div class="card-header">
            <span class="mono-tag">{html_lib.escape(res.strategy.code)}</span>
            {badge}
          </div>
          <div class="card-title">{html_lib.escape(res.strategy.name)}</div>
          <div class="stat-hero">{speedup_str}</div>
          <div class="stat-label">SPEEDUP VS BASELINE ({sol_ms_str} vs {base_ms_str})</div>
          <div class="bar-track" style="margin: 14px 0 12px 0;">
            <div class="bar-fill {'bar-fill-winner' if is_winner else ''}" style="width: {bar_pct:.1f}%;"></div>
          </div>
          <div class="meta-grid">
            <div><span class="meta-k">FORK ENV</span><span class="meta-v">{html_lib.escape(res.environment_id[:16])}…</span></div>
            <div><span class="meta-k">MEDIAN LATENCY</span><span class="meta-v">{sol_ms_str}</span></div>
            <div><span class="meta-k">AGENT TIME</span><span class="meta-v">{res.elapsed_s:.1f}s</span></div>
            <div><span class="meta-k">TOKENS</span><span class="meta-v">{res.total_tokens:,}</span></div>
          </div>
        </div>
        """)

        rows_html.append(f"""
        <tr class="{'row-winner' if is_winner else ''}">
          <td class="mono">{html_lib.escape(res.strategy.code)}</td>
          <td><strong>{html_lib.escape(res.strategy.name)}</strong></td>
          <td class="mono">{html_lib.escape(res.environment_id)}</td>
          <td class="mono">{base_ms_str}</td>
          <td class="mono"><strong>{sol_ms_str}</strong></td>
          <td class="mono"><strong>{speedup_str}</strong></td>
          <td class="mono">{res.total_tokens:,}</td>
          <td>{badge}</td>
        </tr>
        """)

        active_cls = "active" if idx == 0 else ""
        code_tabs_html.append(
            f'<button class="tab-btn {active_cls}" onclick="showCodeTab({idx}, this)">'
            f'{html_lib.escape(res.strategy.code)}: {html_lib.escape(res.strategy.name)}'
            f' ({speedup_str})</button>'
        )
        code_panes_html.append(
            f'<pre class="code-pane {active_cls}" id="code-pane-{idx}">'
            f'<code>{html_lib.escape(res.solution_code or "# No solution.py generated")}</code></pre>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Parallel Sandbox Arena | {html_lib.escape(title)}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-primary: #09090b;
      --bg-secondary: #0f0f12;
      --bg-card: #0c0c0f;
      --bg-subtle: #16161a;
      --border-subtle: #222226;
      --border-bright: #52525b;
      --text-main: #fafafa;
      --text-muted: #a1a1aa;
      --text-dim: #71717a;
      --accent-emerald: #10b981;
      --accent-rose: #ef4444;
      --font-sans: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
      --font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, monospace;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; -webkit-font-smoothing: antialiased; }}
    body {{
      background: var(--bg-primary);
      color: var(--text-main);
      font-family: var(--font-sans);
      line-height: 1.5;
      padding-bottom: 64px;
    }}
    header {{
      height: 48px;
      border-bottom: 1px solid var(--border-subtle);
      padding: 0 32px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-family: var(--font-mono);
      font-size: 11px;
    }}
    .container {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 32px;
    }}
    .eyebrow {{
      font-family: var(--font-mono);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--text-dim);
      margin-bottom: 8px;
    }}
    h1 {{
      font-size: 28px;
      font-weight: 600;
      letter-spacing: -0.035em;
      margin-bottom: 8px;
    }}
    .subtitle {{
      font-size: 13.5px;
      color: var(--text-muted);
      max-width: 820px;
      margin-bottom: 28px;
    }}
    .grid-3 {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
      margin-bottom: 28px;
    }}
    .card {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 4px;
      padding: 20px;
      display: flex;
      flex-direction: column;
    }}
    .card-winner {{
      border-color: var(--text-main);
      background: var(--bg-secondary);
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }}
    .mono-tag {{
      font-family: var(--font-mono);
      font-size: 10.5px;
      color: var(--text-dim);
      letter-spacing: 0.08em;
    }}
    .card-title {{
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 12px;
    }}
    .stat-hero {{
      font-family: var(--font-mono);
      font-variant-numeric: tabular-nums;
      font-size: 36px;
      font-weight: 500;
      letter-spacing: -0.04em;
      line-height: 1;
    }}
    .stat-label {{
      font-family: var(--font-mono);
      font-size: 10px;
      color: var(--text-dim);
      letter-spacing: 0.06em;
      margin-top: 6px;
    }}
    .bar-track {{
      height: 6px;
      background: var(--bg-subtle);
      border-radius: 2px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      background: var(--border-bright);
    }}
    .bar-fill-winner {{
      background: var(--text-main);
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      padding-top: 12px;
      border-top: 1px solid var(--border-subtle);
      font-family: var(--font-mono);
      font-size: 10.5px;
    }}
    .meta-k {{ display: block; color: var(--text-dim); font-size: 9.5px; }}
    .meta-v {{ color: var(--text-muted); font-variant-numeric: tabular-nums; }}
    .pill {{
      font-family: var(--font-mono);
      font-size: 10px;
      padding: 2px 6px;
      border-radius: 2px;
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
    }}
    .pill-winner {{
      color: var(--accent-emerald);
      border-color: rgba(16, 185, 129, 0.35);
      background: rgba(16, 185, 129, 0.08);
    }}
    .pill-fail {{
      color: var(--accent-rose);
      border-color: rgba(239, 68, 68, 0.35);
      background: rgba(239, 68, 68, 0.08);
    }}
    .section-title {{
      font-family: var(--font-mono);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-dim);
      margin-bottom: 12px;
    }}
    .table-wrap {{
      border: 1px solid var(--border-subtle);
      border-radius: 4px;
      overflow-x: auto;
      margin-bottom: 28px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      text-align: left;
    }}
    th {{
      background: var(--bg-secondary);
      padding: 10px 14px;
      font-family: var(--font-mono);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-dim);
      border-bottom: 1px solid var(--border-subtle);
    }}
    td {{
      padding: 10px 14px;
      border-bottom: 1px solid var(--border-subtle);
      color: var(--text-muted);
    }}
    tr:last-child td {{ border-bottom: none; }}
    tr.row-winner td {{ color: var(--text-main); background: rgba(255, 255, 255, 0.02); }}
    .mono {{ font-family: var(--font-mono); font-variant-numeric: tabular-nums; }}
    .tab-bar {{
      display: flex;
      gap: 6px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }}
    .tab-btn {{
      background: transparent;
      border: 1px solid var(--border-subtle);
      color: var(--text-dim);
      padding: 6px 12px;
      border-radius: 2px;
      font-family: var(--font-mono);
      font-size: 11px;
      cursor: pointer;
    }}
    .tab-btn.active {{
      background: var(--text-main);
      color: var(--bg-primary);
      border-color: var(--text-main);
    }}
    .code-pane {{
      display: none;
      background: var(--bg-secondary);
      border: 1px solid var(--border-subtle);
      border-radius: 4px;
      padding: 18px;
      overflow-x: auto;
      font-family: var(--font-mono);
      font-size: 12px;
      line-height: 1.55;
      color: var(--text-main);
    }}
    .code-pane.active {{ display: block; }}
    @media (max-width: 960px) {{ .grid-3 {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <div>ANTIGRAVITY PARALLEL SANDBOX ARENA &bull; FROM_ENVIRONMENT FORK BENCHMARK</div>
    <div>BASE SANDBOX: {html_lib.escape(base_env_id)} &bull; MODEL: {html_lib.escape(model_display)}</div>
  </header>
  <div class="container">
    <div class="eyebrow">BEST-OF-{len(results)} CONCURRENT SANDBOX TOURNAMENT &bull; WALL TIME {total_elapsed_s:.1f}S</div>
    <h1>{html_lib.escape(title)}</h1>
    <p class="subtitle">
      Seeded a single base sandbox (<code>{html_lib.escape(base_env_id)}</code>, baseline latency <code>{baseline_ms:.2f} ms</code>),
      cloned its filesystem state into {len(results)} isolated child sandboxes via <code>environments.create(from_environment=...)</code>,
      and raced {len(results)} autonomous optimization strategies in parallel.
    </p>

    <div class="grid-3">
      {''.join(cards_html)}
    </div>

    <div class="section-title">Contender Telemetry &amp; Verification Matrix</div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Fork</th>
            <th>Strategy</th>
            <th>Forked Sandbox ID</th>
            <th>Baseline</th>
            <th>Median Latency</th>
            <th>Speedup</th>
            <th>Tokens</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>

    <div class="section-title">Generated Implementations (/workspace/solution.py)</div>
    <div class="tab-bar">
      {''.join(code_tabs_html)}
    </div>
    {''.join(code_panes_html)}
  </div>
  <script>
    function showCodeTab(idx, btn) {{
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.code-pane').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('code-pane-' + idx).classList.add('active');
    }}
  </script>
</body>
</html>
"""


def cleanup_sandboxes(
    client: genai.Client,
    logger: Logger,
    env_ids: list[str],
    keep_env: bool,
) -> None:
    """Delete all provisioned sandboxes unless --keep-env was specified."""
    if not env_ids:
        return
    if keep_env:
        logger.log("🔒", f"Retained {len(env_ids)} sandboxes (--keep-env): {', '.join(env_ids)}")
        return

    deleted = 0
    for eid in env_ids:
        try:
            client.environments.delete(id=eid)
            deleted += 1
        except errors.APIError:
            continue
    logger.log("🧹", f"Cleaned up {deleted}/{len(env_ids)} remote sandboxes.")


def main(argv: list[str] | None = None) -> int:
    """Execute the 4-stage Parallel Sandbox Forking Arena."""
    args = parse_args(argv)
    model_name: str | None = ARGON_MODEL if args.argon else args.model
    model_display = model_name or DEFAULT_MODEL_LABEL
    challenge_title = args.task or DEFAULT_CHALLENGE_TITLE

    logger = Logger()
    client = genai.Client()
    created_env_ids: list[str] = []

    try:
        # ---------------- Step 1: Seed Base Sandbox ----------------
        logger.log(
            "▶",
            f"STEP 1/4: Seeding base sandbox for '{challenge_title}' (model: {model_display})",
        )
        base_env_id, strategies = seed_base_sandbox(
            client,
            logger,
            custom_task=args.task,
            model=model_name,
            verbose=args.verbose,
        )
        created_env_ids.append(base_env_id)

        # ---------------- Step 2: Fork Base Sandbox into N Child Sandboxes ----------------
        logger.log(
            "▶",
            f"STEP 2/4: Forking base sandbox ({base_env_id}) into {len(strategies)} isolated child sandboxes...",
        )
        forked_envs: list[tuple[StrategySpec, str]] = []
        for spec in strategies:
            fork_env = client.environments.create(from_environment=base_env_id)
            fork_id = getattr(fork_env, "id", None)
            if not fork_id:
                raise RuntimeError(f"Failed to fork sandbox for {spec.code}")
            created_env_ids.append(fork_id)
            forked_envs.append((spec, fork_id))
            logger.log("🌿", f"Forked {base_env_id} -> {fork_id} ({spec.code}: {spec.name})")

        # ---------------- Step 3: Race All N Agents Concurrently ----------------
        logger.log(
            "▶",
            f"STEP 3/4: Racing {len(forked_envs)} agents in parallel across forked sandboxes...",
        )
        results_by_code: dict[str, ContenderResult] = {}
        with ThreadPoolExecutor(max_workers=len(forked_envs)) as pool:
            future_map = {
                pool.submit(
                    run_contender,
                    client,
                    logger,
                    strategy=spec,
                    fork_env_id=fork_id,
                    model=model_name,
                    verbose=args.verbose,
                ): spec
                for spec, fork_id in forked_envs
            }
            for future in as_completed(future_map):
                spec = future_map[future]
                results_by_code[spec.code] = future.result()

        # Order results by speedup descending (passing solutions first)
        ordered_results = sorted(
            results_by_code.values(),
            key=lambda r: (r.passed, r.speedup_x),
            reverse=True,
        )
        winner = ordered_results[0]

        # ---------------- Step 4: Export Winner & Minimalist Scorecard ----------------
        logger.log("▶", "STEP 4/4: Judging telemetry & exporting winning artifacts...")
        if winner.solution_code:
            args.winner_out.parent.mkdir(parents=True, exist_ok=True)
            args.winner_out.write_text(winner.solution_code, encoding="utf-8")
            logger.log(
                "🥇",
                f"Winner: {winner.strategy.code} ({winner.strategy.name}) — "
                f"{winner.speedup_x:.2f}x speedup ({winner.solution_ms}ms vs {winner.baseline_ms}ms baseline)",
            )
            logger.log("💾", f"Saved winning code to: {args.winner_out.resolve()}")

        report_html = build_arena_report_html(
            title=challenge_title,
            base_env_id=base_env_id,
            model_display=model_display,
            results=ordered_results,
            winner=winner,
            total_elapsed_s=logger.elapsed,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_html, encoding="utf-8")
        logger.log("📊", f"Saved arena scorecard to: {args.output.resolve()}")

        if args.open:
            webbrowser.open(args.output.resolve().as_uri())

        return 0 if winner.passed else 1

    finally:
        cleanup_sandboxes(client, logger, created_env_ids, args.keep_env)


if __name__ == "__main__":
    sys.exit(main())
