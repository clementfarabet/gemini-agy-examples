"""Example 1: Run an Antigravity agent in a remote sandbox and resume statefully.

Streams live thoughts, Google searches, shell executions, and model output
directly to the terminal, exports the generated HTML artifact to local disk,
and cleans up the remote sandbox.

Usage:
    export GEMINI_API_KEY="..."
    uv run python test_snippet1_run_resume.py [--argon | --model MODEL] [--keep-env] [--output PATH]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any
import warnings

from google import genai
from google.genai import errors

# Suppress experimental API warnings from the SDK for cleaner CLI output
warnings.filterwarnings("ignore", message=".*usage is experimental.*")

DEFAULT_AGENT = "antigravity-preview-09-2026"
DEFAULT_MODEL_LABEL = "gemini-3.8-flash (agent default)"
ARGON_MODEL = "gemini-4-argon"
CANDIDATE_ARTIFACT_PATHS = ("workspace/analysis.html", "analysis.html")


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
        description="Run the Antigravity agent in a remote Linux sandbox and resume statefully."
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
        "--keep-env",
        action="store_true",
        help="Preserve the remote sandbox on completion instead of deleting it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis.html"),
        help="Local file path where the generated dashboard artifact will be saved (default: analysis.html).",
    )
    return parser.parse_args(argv)


def format_step_event(step: Any, logger: Logger) -> None:
    """Log human-readable progress for an agent step.start event."""
    step_type = getattr(step, "type", "")
    args = getattr(step, "arguments", None)

    if step_type == "thought":
        logger.log("💭", "Agent planning next action...")
    elif "search" in step_type:
        queries = getattr(args, "queries", "") if args else ""
        logger.log("🔍", f"Google Search: {queries}")
    elif "code" in step_type:
        cmd = getattr(args, "command", getattr(args, "code", "")) if args else ""
        snippet = str(cmd).strip().replace("\n", " ")[:70]
        logger.log("⚡", f"Shell / Code execution: {snippet}")
    elif step_type == "model_output":
        logger.log("📝", "Agent response:\n" + "-" * 50)
    elif step_type:
        logger.log("🔧", f"Tool step: {step_type}")


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
    """Download the generated artifact from the remote sandbox to local disk."""
    logger.log("📥", f"Step 4/4: Downloading generated artifact to {output_path}...")
    for remote_path in CANDIDATE_ARTIFACT_PATHS:
        try:
            content = client.environments.files.download(
                environment=environment_id,
                path=remote_path,
            )
            output_path.write_bytes(content)
            logger.log(
                "✓",
                f"Saved local artifact: {output_path} ({len(content):,} bytes from {remote_path})",
            )
            return True
        except errors.APIError:
            continue

    logger.log("ℹ", "Could not locate analysis.html in sandbox via direct download.")
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
    """Run the multi-step sandbox creation, resumption, probe, and export pipeline."""
    args = parse_args(argv)
    model_name: str | None = ARGON_MODEL if args.argon else args.model
    model_display = model_name or DEFAULT_MODEL_LABEL

    logger = Logger()
    client = genai.Client()
    active_env_id: str | None = None

    try:
        # ---------------- Step 1: Initial Run ----------------
        print("\n" + "=" * 65)
        logger.log("▶", "STEP 1/4: Provisioning remote sandbox and running Antigravity agent...")
        logger.log(" ", f"Model:  {model_display}")
        logger.log(" ", "Prompt: 'Research top 5 sustainable fashion brands & build analysis.html'")
        print("=" * 65)

        run1 = stream_interaction(
            client,
            logger,
            prompt=(
                "Research the top 5 sustainable fashion brands "
                "and build a dashboard in analysis.html"
            ),
            environment="remote",
            model=model_name,
        )
        active_env_id = run1.environment_id
        if not active_env_id:
            logger.log("❌", "Failed to obtain an environment_id from Step 1.")
            return 1

        # ---------------- Step 2: Follow-up in Same Sandbox ----------------
        # Passes both `environment` (sandbox state) and `previous_interaction_id` (conversation history)
        print("=" * 65)
        logger.log("▶", f"STEP 2/4: Reusing sandbox ({active_env_id}) for follow-up...")
        logger.log(" ", "Prompt: 'Add a pricing-tier filter to the dashboard'")
        print("=" * 65)

        run2 = stream_interaction(
            client,
            logger,
            prompt="Add a pricing-tier filter to the dashboard",
            environment=active_env_id,
            previous_id=run1.id,
            model=model_name,
        )

        # ---------------- Step 3: Verification & Persistence Probe ----------------
        # Intentionally omits `previous_interaction_id` to start a fresh conversation turn
        # while reusing `environment=active_env_id` to verify true filesystem persistence.
        print("=" * 65)
        logger.log("▶", f"STEP 3/4: Probing sandbox ({active_env_id}) to verify persistence...")
        print("=" * 65)

        probe = stream_interaction(
            client,
            logger,
            prompt=(
                "Run `ls -la /workspace` and `grep -ci 'tier' analysis.html` (search for "
                "analysis.html if it is elsewhere). Reply with the raw command output only."
            ),
            environment=active_env_id,
            model=model_name,
        )

        # ---------------- Step 4: Export Artifact ----------------
        downloaded = download_artifact(client, logger, active_env_id, args.output)

        checks = {
            "run1.environment_id is set": bool(run1.environment_id),
            "run2 reused the same sandbox": run2.environment_id == run1.environment_id,
            "run1 produced output_text": bool(run1.output_text),
            "run2 produced output_text": bool(run2.output_text),
            "analysis.html preserved in sandbox": "analysis.html" in probe.output_text,
            "analysis.html exported locally": downloaded,
        }

        print("\n" + "=" * 65)
        print("FINAL VERIFICATION RESULTS:")
        for name, ok in checks.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"Model used:          {model_display}")
        print(f"Total pipeline time: {logger.elapsed:.1f}s")
        print(f"Sandbox ID:          {active_env_id}")
        print("=" * 65 + "\n")

        return 0 if all(checks.values()) else 1

    finally:
        cleanup_environment(client, logger, active_env_id, args.keep_env)


if __name__ == "__main__":
    sys.exit(main())
