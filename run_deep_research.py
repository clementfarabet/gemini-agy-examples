"""Generic Deep Research & Interactive Slide Deck Generator using an Antigravity Managed Agent.

Provisions an isolated remote Linux sandbox, conducts multi-step web research on
any topic, builds a self-contained interactive slide deck website (`report.html`),
iteratively refines it in the same stateful sandbox, and downloads the final
HTML report to your local machine.

Usage:
    export GEMINI_API_KEY="..."

    # Research any topic and export report.html slide deck (with Gemini 4 Argon)
    uv run python run_deep_research.py "Global solid-state battery landscape in 2026" --argon

    # Add custom follow-up refinement pass(es) in the same sandbox and open in browser
    uv run python run_deep_research.py "Open-source Rust web frameworks" --argon \\
        --refine "Add a benchmark comparison slide and an interactive use-case filter" \\
        --open

    # Single-pass mode (skip the follow-up refinement turn)
    uv run python run_deep_research.py "Quantum error correction breakthroughs" --no-refine
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any
import warnings
import webbrowser

from google import genai
from google.genai import errors
try:
    from google.genai._gaos.lib import compat_errors
    API_ERRORS: tuple[type[Exception], ...] = (
        errors.APIError,
        compat_errors.GeminiNextGenAPIClientError,
    )
except ImportError:
    API_ERRORS = (errors.APIError,)

# Suppress experimental API warnings from the SDK for cleaner CLI output
warnings.filterwarnings("ignore", message=".*usage is experimental.*")

DEFAULT_AGENT = "antigravity-preview-09-2026"
DEFAULT_MODEL_LABEL = "gemini-3.8-flash (agent default)"
ARGON_MODEL = "gemini-4-argon"
REMOTE_ARTIFACT_NAME = "report.html"
DEFAULT_TOPIC = "Top 5 sustainable fashion brands and their supply-chain metrics"
DEFAULT_REFINEMENT = (
    f"Refine `/workspace/{REMOTE_ARTIFACT_NAME}` to make both its interactivity and visual precision "
    "exceptional: add interactive in-slide filters/tabs, sortable comparison tables, expandable detail rows, "
    "and a slide-grid overview toggle (`G` key or button). Enforce a lean, minimalist, ultra-precise modern "
    "design (Linear / Vercel / Swiss editorial style): strict monochrome palette with a single restrained "
    "accent, crisp 1px hairline borders, tabular monospace metrics, generous whitespace, and zero neon "
    "gradients, glowing blurs, drop shadows, or decorative emojis."
)
CANDIDATE_ARTIFACT_PATHS = (
    f"workspace/{REMOTE_ARTIFACT_NAME}",
    REMOTE_ARTIFACT_NAME,
)


@dataclass(frozen=True)
class InteractionResult:
    """Holds the key identifiers, final output text, and token usage from an interaction."""

    id: str | None
    environment_id: str | None
    output_text: str
    total_tokens: int | None = None


class Logger:
    """Simple elapsed-time terminal logger."""

    def __init__(self) -> None:
        self._start = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def log(self, icon: str, message: str) -> None:
        print(f"[{self.elapsed:5.1f}s] {icon} {message}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Conduct deep research on any topic inside a remote Antigravity sandbox "
            f"and generate a local {REMOTE_ARTIFACT_NAME} interactive slide deck website."
        )
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default=DEFAULT_TOPIC,
        help=f"Research topic to investigate (default: '{DEFAULT_TOPIC}').",
    )

    refine_group = parser.add_mutually_exclusive_group()
    refine_group.add_argument(
        "-f",
        "--refine",
        action="append",
        default=None,
        help=(
            "Follow-up refinement instruction(s) executed in the same stateful sandbox "
            "(can be passed multiple times). Defaults to an interactive slide deck enhancement pass."
        ),
    )
    refine_group.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip the follow-up refinement turn and export after the initial research pass.",
    )

    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model",
        type=str,
        default=None,
        help="Underlying Gemini model for the agent (e.g., gemini-4-argon, gemini-3.8-flash).",
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
        default=Path(REMOTE_ARTIFACT_NAME),
        help=f"Local file path to save the generated slide deck report (default: {REMOTE_ARTIFACT_NAME}).",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Automatically open the exported HTML slide deck in your default web browser.",
    )
    parser.add_argument(
        "--keep-env",
        action="store_true",
        help="Preserve the remote sandbox on completion instead of deleting it.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Stream full raw agent text responses and multi-line thought summaries to stdout.",
    )
    return parser.parse_args(argv)


def build_initial_prompt(topic: str) -> str:
    """Construct the initial research and interactive slide deck generation prompt."""
    return (
        f"Research the following topic thoroughly using Google Search: {topic}\n\n"
        "Synthesize your findings, quantitative metrics, comparisons, and verified sources, "
        "and build a self-contained, interactive HTML/CSS/JS **slide deck website** "
        f"saved to `/workspace/{REMOTE_ARTIFACT_NAME}`.\n\n"
        "Design Aesthetic (Lean, Minimalist, High-Precision Modern):\n"
        "- Follow a Swiss editorial / Linear / Vercel / Stripe design language.\n"
        "- Use `Inter` for UI/editorial typography (tight tracking `-0.03em` on headings) and `JetBrains Mono` "
        "(`font-variant-numeric: tabular-nums`) for all metrics, indices, tags, and table numbers.\n"
        "- Strict high-contrast monochrome palette (`#09090b` / `#111113` surfaces, `#fafafa` primary text, "
        "`#a1a1aa` secondary text, `#27272a` 1px solid hairline borders, plus clean light-mode support) "
        "with at most ONE restrained technical accent color.\n"
        "- Sharp architectural geometry (`border-radius: 4px` or `0px` hairline grid cells). "
        "NO ambient blur blobs, NO neon gradients, NO heavy drop shadows, and NO decorative emojis.\n"
        "- Structure as a multi-slide presentation (Title & Executive Summary, Architecture/Overview, "
        "Quantitative Benchmarks, Comparative Matrix, Case Studies, Strategic Takeaways & Sources) with "
        "keyboard navigation (`ArrowLeft` / `ArrowRight`, `Space`), minimal header/footer chrome, "
        "1px progress bar, slide counter (`01 / N`), and a clickable slide index."
    )


def summarize_one_line(text: str, max_len: int = 90) -> str:
    """Collapse multi-line markdown text into a clean single-line summary."""
    lines = [line.strip().strip("*#").strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    combined = " — ".join(lines) if len(lines) > 1 else lines[0]
    combined = " ".join(combined.split())
    if len(combined) > max_len:
        return combined[: max_len - 3].rstrip() + "..."
    return combined


def extract_tool_detail(args: Any) -> str:
    """Extract a concise one-line detail from step.arguments if available."""
    if not args:
        return ""
    keys = ("toolSummary", "TargetFile", "AbsolutePath", "path", "CommandLine", "command", "query", "Url")
    if isinstance(args, dict):
        for key in keys:
            if args.get(key):
                return summarize_one_line(str(args[key]), max_len=60)
    else:
        for key in keys:
            val = getattr(args, key, None)
            if val:
                return summarize_one_line(str(val), max_len=60)
    return ""


def format_step_event(step: Any, logger: Logger, *, verbose: bool = False) -> None:
    """Log human-readable progress for an agent step.start event."""
    step_type = getattr(step, "type", "")
    args = getattr(step, "arguments", None)

    if step_type == "thought":
        logger.log("💭", "Planning next action...")
    elif step_type == "google_search_call":
        queries = getattr(args, "queries", "") if args else ""
        logger.log("🔍", f"Google Search: {summarize_one_line(str(queries), max_len=80)}")
    elif step_type == "code_execution_call":
        cmd = getattr(args, "command", getattr(args, "code", "")) if args else ""
        snippet = summarize_one_line(str(cmd), max_len=75)
        logger.log("⚡", f"Code execution: {snippet}" if snippet else "Code execution")
    elif step_type == "function_call":
        fn_name = getattr(step, "name", "") or "tool"
        detail = extract_tool_detail(args)
        suffix = f" ({detail})" if detail else ""
        logger.log("🔧", f"Tool call: {fn_name}{suffix}")
    elif step_type == "model_output":
        if verbose:
            logger.log("📝", "Agent response:\n" + "-" * 50)
        else:
            logger.log("📝", "Synthesizing response...")
    elif step_type and not step_type.endswith("_result"):
        logger.log("🔧", f"Step: {step_type}")


def stream_interaction(
    client: genai.Client,
    logger: Logger,
    *,
    prompt: str,
    environment: str,
    previous_id: str | None = None,
    model: str | None = None,
    verbose: bool = False,
) -> InteractionResult:
    """Execute an Antigravity agent interaction with real-time SSE streaming."""
    kwargs: dict[str, Any] = {
        "agent": DEFAULT_AGENT,
        "input": prompt,
        "environment": environment,
        "stream": True,
    }
    if model:
        kwargs["agent_config"] = {"type": "antigravity", "model": model}
    if previous_id:
        kwargs["previous_interaction_id"] = previous_id

    try:
        events = client.interactions.create(**kwargs)
    except API_ERRORS as exc:
        if exc.code == 429 or "RESOURCE_EXHAUSTED" in str(exc):
            logger.log("❌", f"Quota exceeded ({exc.code}): {exc.message}")
            print(
                "\nNote: Models like Gemini 4 Argon require a Google AI Studio / GCP "
                "project with billing enabled.",
                file=sys.stderr,
            )
        raise

    interaction_id: str | None = None
    env_id: str | None = environment if environment != "remote" else None
    total_tokens: int | None = None
    collected_text: list[str] = []

    for event in events:
        event_type = getattr(event, "event_type", "")

        if event_type == "interaction.created":
            interaction = getattr(event, "interaction", None)
            if interaction:
                interaction_id = getattr(interaction, "id", interaction_id)
                env_id = getattr(interaction, "environment_id", env_id)
            logger.log("🚀", f"Sandbox ready: {env_id or 'provisioning...'}")

        elif event_type == "step.start":
            format_step_event(getattr(event, "step", None), logger, verbose=verbose)

        elif event_type == "step.delta":
            delta = getattr(event, "delta", None)
            delta_type = getattr(delta, "type", "")

            if delta_type == "text":
                text = getattr(delta, "text", "")
                if text:
                    collected_text.append(text)
                    if verbose:
                        sys.stdout.write(text)
                        sys.stdout.flush()
            elif delta_type == "thought_summary":
                content = getattr(delta, "content", None)
                thought_text = getattr(content, "text", "") if content else ""
                if thought_text:
                    if verbose:
                        logger.log("💡", f"Thought summary: {thought_text.strip()}")
                    else:
                        logger.log("💡", f"Thought: {summarize_one_line(thought_text)}")

        elif event_type == "step.stop":
            usage = getattr(event, "usage", None)
            if usage and getattr(usage, "total_tokens", None):
                total_tokens = usage.total_tokens

        elif event_type == "interaction.completed":
            interaction = getattr(event, "interaction", None)
            if interaction:
                env_id = getattr(interaction, "environment_id", env_id)
                interaction_id = getattr(interaction, "id", interaction_id)
                usage = getattr(interaction, "usage", None)
                if usage and getattr(usage, "total_tokens", None):
                    total_tokens = usage.total_tokens
            if verbose:
                print("\n" + "-" * 50)
            token_info = f", tokens: {total_tokens:,}" if total_tokens else ""
            logger.log("✓", f"Step finished (Interaction: {interaction_id}{token_info})")

        elif event_type == "error":
            err = getattr(event, "error", None)
            code = getattr(err, "code", "unknown")
            msg = getattr(err, "message", str(err))
            raise RuntimeError(f"Stream error [{code}]: {msg}")

    return InteractionResult(
        id=interaction_id,
        environment_id=env_id,
        output_text="".join(collected_text),
        total_tokens=total_tokens,
    )


def download_artifact(
    client: genai.Client,
    logger: Logger,
    environment_id: str,
    output_path: Path,
) -> bool:
    """Download the generated slide deck report artifact from the remote sandbox to local disk."""
    logger.log("📥", f"Downloading generated slide deck report to {output_path}...")
    for remote_path in CANDIDATE_ARTIFACT_PATHS:
        try:
            content = client.environments.files.download(
                environment=environment_id,
                path=remote_path,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(content)
            logger.log(
                "✓",
                f"Saved local slide deck report: {output_path} ({len(content):,} bytes from {remote_path})",
            )
            return True
        except API_ERRORS:
            continue

    logger.log("ℹ", f"Could not locate {REMOTE_ARTIFACT_NAME} in sandbox via direct download.")
    return False


def cleanup_environment(
    client: genai.Client,
    logger: Logger,
    environment_id: str | None,
    keep_env: bool,
) -> None:
    """Delete the remote sandbox unless preservation was requested."""
    if not environment_id:
        return
    if keep_env:
        logger.log("🔒", f"Sandbox retained (--keep-env): {environment_id}")
        return

    try:
        client.environments.delete(id=environment_id)
        logger.log("🧹", f"Sandbox {environment_id} deleted cleanly.")
    except API_ERRORS as exc:
        logger.log("ℹ", f"Sandbox retained ({exc.code}); auto-expires after 7 days.")


def main(argv: list[str] | None = None) -> int:
    """Run the deep research, iterative refinement, verification, and export pipeline."""
    args = parse_args(argv)
    model_name: str | None = ARGON_MODEL if args.argon else args.model
    model_display = model_name or DEFAULT_MODEL_LABEL

    refinements: list[str] = []
    if not args.no_refine:
        refinements = args.refine if args.refine else [DEFAULT_REFINEMENT]

    total_steps = 1 + len(refinements) + 1  # Initial + Refinements + Export
    logger = Logger()
    client = genai.Client()
    active_env_id: str | None = None

    try:
        # ---------------- Step 1: Initial Deep Research & Slide Deck Build ----------------
        logger.log(
            "▶",
            f"STEP 1/{total_steps}: Researching '{args.topic}' (model: {model_display}, output: {args.output})",
        )

        run1 = stream_interaction(
            client,
            logger,
            prompt=build_initial_prompt(args.topic),
            environment="remote",
            model=model_name,
            verbose=args.verbose,
        )
        active_env_id = run1.environment_id
        last_interaction_id = run1.id
        if not active_env_id:
            logger.log("❌", "Failed to obtain an environment_id from Step 1.")
            return 1

        # ---------------- Step 2+: Stateful Refinement(s) in Same Sandbox ----------------
        last_run = run1
        for idx, refine_prompt in enumerate(refinements, start=2):
            logger.log(
                "▶",
                f"STEP {idx}/{total_steps}: Refining in sandbox {active_env_id} ({summarize_one_line(refine_prompt, max_len=65)})",
            )

            last_run = stream_interaction(
                client,
                logger,
                prompt=refine_prompt,
                environment=active_env_id,
                previous_id=last_interaction_id,
                model=model_name,
                verbose=args.verbose,
            )
            last_interaction_id = last_run.id

        # ---------------- Final Step: Export Artifact ----------------
        logger.log("▶", f"STEP {total_steps}/{total_steps}: Exporting slide deck report from sandbox...")
        downloaded = download_artifact(client, logger, active_env_id, args.output)

        checks = {
            "sandbox environment_id allocated": bool(run1.environment_id),
            "initial research produced output": bool(run1.output_text),
            "stateful sandbox reused across turns": last_run.environment_id == run1.environment_id,
            f"slide deck report exported to {args.output}": downloaded,
        }

        if args.verbose:
            print("\n" + "=" * 65)
            print("DEEP RESEARCH PIPELINE SUMMARY:")
            for name, ok in checks.items():
                print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
            print(f"Topic:               {args.topic}")
            print(f"Model used:          {model_display}")
            print(f"Local Report:        {args.output.resolve()}")
            print(f"Total pipeline time: {logger.elapsed:.1f}s")
            print(f"Sandbox ID:          {active_env_id}")
            print("=" * 65 + "\n")
        else:
            status_icon = "🎉" if all(checks.values()) else "⚠️"
            logger.log(
                status_icon,
                f"Pipeline complete — Report: {args.output.resolve()} (model: {model_display})",
            )

        if downloaded and args.open:
            webbrowser.open(args.output.resolve().as_uri())

        return 0 if all(checks.values()) else 1

    finally:
        cleanup_environment(client, logger, active_env_id, args.keep_env)


if __name__ == "__main__":
    sys.exit(main())
