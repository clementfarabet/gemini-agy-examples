"""Generic Deep Research & Analytics Dashboard Generator using an Antigravity Managed Agent.

Provisions an isolated remote Linux sandbox, conducts multi-step web research on
any topic, builds a self-contained interactive dashboard (`analytics.html`),
iteratively refines it in the same stateful sandbox, and downloads the final
HTML artifact to your local machine.

Usage:
    export GEMINI_API_KEY="..."

    # Research any topic and export analytics.html (with Gemini 4 Argon)
    uv run python run_deep_research.py "Global solid-state battery landscape in 2026" --argon

    # Add custom follow-up refinement pass(es) in the same sandbox and open in browser
    uv run python run_deep_research.py "Open-source Rust web frameworks" --argon \\
        --refine "Add a benchmark comparison chart and an interactive use-case filter" \\
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

# Suppress experimental API warnings from the SDK for cleaner CLI output
warnings.filterwarnings("ignore", message=".*usage is experimental.*")

DEFAULT_AGENT = "antigravity-preview-09-2026"
DEFAULT_MODEL_LABEL = "gemini-3.8-flash (agent default)"
ARGON_MODEL = "gemini-4-argon"
DEFAULT_TOPIC = "Top 5 sustainable fashion brands and their supply-chain metrics"
DEFAULT_REFINEMENT = (
    "Enhance the dashboard in /workspace/analytics.html by adding interactive "
    "category/tier filters, sortable metric columns, and a key takeaways summary section."
)
REMOTE_ARTIFACT_NAME = "analytics.html"
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
            "and generate a local analytics.html dashboard."
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
            "(can be passed multiple times). Defaults to an interactive filter & summary enhancement pass."
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
        help=f"Local file path to save the generated dashboard (default: {REMOTE_ARTIFACT_NAME}).",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Automatically open the exported HTML dashboard in your default web browser.",
    )
    parser.add_argument(
        "--keep-env",
        action="store_true",
        help="Preserve the remote sandbox on completion instead of deleting it.",
    )
    return parser.parse_args(argv)


def build_initial_prompt(topic: str) -> str:
    """Construct the initial research and dashboard generation prompt."""
    return (
        f"Research the following topic thoroughly using Google Search: {topic}\n\n"
        "Synthesize your findings, quantitative metrics, comparisons, and sources, "
        f"and build a polished, self-contained, interactive HTML/CSS/JS analytics dashboard "
        f"saved to `/workspace/{REMOTE_ARTIFACT_NAME}`."
    )


def format_step_event(step: Any, logger: Logger) -> None:
    """Log human-readable progress for an agent step.start event."""
    step_type = getattr(step, "type", "")
    args = getattr(step, "arguments", None)

    if step_type == "thought":
        logger.log("💭", "Agent planning next action...")
    elif step_type == "google_search_call":
        queries = getattr(args, "queries", "") if args else ""
        logger.log("🔍", f"Google Search: {queries}")
    elif step_type == "code_execution_call":
        cmd = getattr(args, "command", getattr(args, "code", "")) if args else ""
        snippet = str(cmd).strip().replace("\n", " ")[:70]
        if snippet:
            logger.log("⚡", f"Shell / Code execution: {snippet}")
    elif step_type == "function_call":
        fn_name = getattr(step, "name", "") or "tool"
        logger.log("🔧", f"Tool call: {fn_name}")
    elif step_type == "model_output":
        logger.log("📝", "Agent response:\n" + "-" * 50)
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
    except errors.APIError as exc:
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
            logger.log("🚀", f"Sandbox allocated: {env_id or 'provisioning...'}")

        elif event_type == "step.start":
            format_step_event(getattr(event, "step", None), logger)

        elif event_type == "step.delta":
            delta = getattr(event, "delta", None)
            delta_type = getattr(delta, "type", "")

            if delta_type == "text":
                text = getattr(delta, "text", "")
                if text:
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    collected_text.append(text)
            elif delta_type == "thought_summary":
                content = getattr(delta, "content", None)
                thought_text = getattr(content, "text", "") if content else ""
                if thought_text:
                    logger.log("💡", f"Thought summary: {thought_text.strip()}")

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
            print("\n" + "-" * 50)
            token_info = f", tokens: {total_tokens:,}" if total_tokens else ""
            logger.log("✓", f"Step finished! (Interaction: {interaction_id}{token_info})\n")

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
    """Download the generated dashboard artifact from the remote sandbox to local disk."""
    logger.log("📥", f"Downloading generated dashboard to {output_path}...")
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
                f"Saved local dashboard: {output_path} ({len(content):,} bytes from {remote_path})",
            )
            return True
        except errors.APIError:
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
    except errors.APIError as exc:
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
        # ---------------- Step 1: Initial Deep Research & Dashboard Build ----------------
        print("\n" + "=" * 65)
        logger.log("▶", f"STEP 1/{total_steps}: Provisioning sandbox & researching topic...")
        logger.log(" ", f"Topic:  {args.topic}")
        logger.log(" ", f"Model:  {model_display}")
        logger.log(" ", f"Output: {args.output}")
        print("=" * 65)

        run1 = stream_interaction(
            client,
            logger,
            prompt=build_initial_prompt(args.topic),
            environment="remote",
            model=model_name,
        )
        active_env_id = run1.environment_id
        last_interaction_id = run1.id
        if not active_env_id:
            logger.log("❌", "Failed to obtain an environment_id from Step 1.")
            return 1

        # ---------------- Step 2+: Stateful Refinement(s) in Same Sandbox ----------------
        last_run = run1
        for idx, refine_prompt in enumerate(refinements, start=2):
            print("=" * 65)
            logger.log(
                "▶",
                f"STEP {idx}/{total_steps}: Reusing sandbox ({active_env_id}) for refinement...",
            )
            logger.log(" ", f"Prompt: '{refine_prompt}'")
            print("=" * 65)

            last_run = stream_interaction(
                client,
                logger,
                prompt=refine_prompt,
                environment=active_env_id,
                previous_id=last_interaction_id,
                model=model_name,
            )
            last_interaction_id = last_run.id

        # ---------------- Final Step: Export Artifact ----------------
        print("=" * 65)
        logger.log("▶", f"STEP {total_steps}/{total_steps}: Exporting dashboard from sandbox...")
        print("=" * 65)
        downloaded = download_artifact(client, logger, active_env_id, args.output)

        checks = {
            "sandbox environment_id allocated": bool(run1.environment_id),
            "initial research produced output": bool(run1.output_text),
            "stateful sandbox reused across turns": last_run.environment_id == run1.environment_id,
            f"dashboard exported to {args.output}": downloaded,
        }

        print("\n" + "=" * 65)
        print("DEEP RESEARCH PIPELINE SUMMARY:")
        for name, ok in checks.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"Topic:               {args.topic}")
        print(f"Model used:          {model_display}")
        print(f"Local Dashboard:     {args.output.resolve()}")
        print(f"Total pipeline time: {logger.elapsed:.1f}s")
        print(f"Sandbox ID:          {active_env_id}")
        print("=" * 65 + "\n")

        if downloaded and args.open:
            webbrowser.open(args.output.resolve().as_uri())

        return 0 if all(checks.values()) else 1

    finally:
        cleanup_environment(client, logger, active_env_id, args.keep_env)


if __name__ == "__main__":
    sys.exit(main())
