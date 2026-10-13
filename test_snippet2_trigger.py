"""Example 2: Create, verify, and manage an always-on scheduled cron trigger.

By default, the trigger is deleted immediately after verification so nothing
runs every morning at 9:00 AM (and nothing gets billed). Pass --keep to leave
the trigger active.

Prerequisite note:
    By default, a stored credential named "github-production" is included in the
    network allowlist. If your project does not have that credential configured,
    pass --no-credential to test trigger creation without credential injection.

Usage:
    export GEMINI_API_KEY="..."
    python test_snippet2_trigger.py [--keep] [--no-credential | --credential NAME] [--argon | --model MODEL]
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any
import warnings

from google import genai

# Suppress experimental API warnings from the SDK for cleaner CLI output
warnings.filterwarnings("ignore", message=".*usage is experimental.*")

DEFAULT_AGENT = "antigravity-preview-09-2026"
DEFAULT_MODEL_LABEL = "gemini-3.8-flash (agent default)"
ARGON_MODEL = "gemini-4-argon"
DEFAULT_CREDENTIAL = "github-production"


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
        description="Create and verify a scheduled cron trigger for an Antigravity agent."
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Leave the scheduled trigger active instead of deleting it after verification.",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="0 9 * * *",
        help="Cron expression for the trigger schedule (default: '0 9 * * *').",
    )
    parser.add_argument(
        "--time-zone",
        type=str,
        default="America/New_York",
        help="IANA time zone for the cron schedule (default: 'America/New_York').",
    )

    cred_group = parser.add_mutually_exclusive_group()
    cred_group.add_argument(
        "--credential",
        type=str,
        default=DEFAULT_CREDENTIAL,
        help=f"Stored credential name to inject for api.github.com (default: '{DEFAULT_CREDENTIAL}').",
    )
    cred_group.add_argument(
        "--no-credential",
        action="store_true",
        help="Omit stored credential requirement so the trigger can be tested without pre-registered secrets.",
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
        help=f"Shorthand to configure the trigger with Gemini 4 Argon (--model {ARGON_MODEL}).",
    )
    return parser.parse_args(argv)


def build_interaction_config(
    *,
    credential: str | None,
    model: str | None,
) -> dict[str, Any]:
    """Construct the interaction payload for the scheduled trigger."""
    allowlist_entry: dict[str, str] = {"domain": "api.github.com"}
    if credential:
        allowlist_entry["credential"] = credential

    interaction: dict[str, Any] = {
        "agent": DEFAULT_AGENT,
        "input": (
            "Triage new issues in my-org/my-app and "
            "save a report to /workspace/reports/"
        ),
        "environment": {
            "type": "remote",
            "network": {"allowlist": [allowlist_entry]},
        },
    }
    if model:
        interaction["agent_config"] = {"type": "antigravity", "model": model}
    return interaction


def cleanup_trigger(
    client: genai.Client,
    logger: Logger,
    trigger_id: str | None,
    keep: bool,
    schedule: str,
    time_zone: str,
) -> None:
    """Delete the trigger unless preservation was requested."""
    if not trigger_id:
        return
    if keep:
        logger.log("🔒", f"Trigger {trigger_id} left ACTIVE ({schedule} {time_zone}).")
        return

    logger.log("🧹", "Step 3/3: Cleaning up trigger...")
    try:
        client.triggers.delete(id=trigger_id)
        logger.log("✓", f"Trigger {trigger_id} deleted cleanly.")
    except Exception as exc:
        logger.log("⚠️", f"Failed to delete trigger {trigger_id}: {exc}")


def main(argv: list[str] | None = None) -> int:
    """Run the trigger creation, verification, and cleanup workflow."""
    args = parse_args(argv)
    model_name: str | None = ARGON_MODEL if args.argon else args.model
    model_display = model_name or DEFAULT_MODEL_LABEL
    credential_name: str | None = None if args.no_credential else args.credential

    logger = Logger()
    client = genai.Client()
    trigger_id: str | None = None

    print("\n" + "=" * 65)
    logger.log("▶", f"Step 1/3: Creating scheduled trigger ({args.schedule} {args.time_zone})...")
    logger.log(" ", f"Agent:      {DEFAULT_AGENT}")
    logger.log(" ", f"Model:      {model_display}")
    logger.log(" ", f"Credential: {credential_name or 'None (--no-credential)'}")
    print("=" * 65)

    try:
        interaction_cfg = build_interaction_config(
            credential=credential_name,
            model=model_name,
        )

        trigger = client.triggers.create(
            schedule=args.schedule,
            time_zone=args.time_zone,
            display_name="daily-issue-triage",
            interaction=interaction_cfg,
        )
        trigger_id = getattr(trigger, "id", None)
        next_run_time = getattr(trigger, "next_run_time", None)

        logger.log("✓", f"Trigger created! ID: {trigger_id}")
        logger.log("🕒", f"Next run time:       {next_run_time}")

        logger.log("🔍", "Step 2/3: Fetching trigger to verify active status...")
        fetched = client.triggers.get(id=trigger_id)
        status_str = str(getattr(fetched, "status", ""))

        checks = {
            "trigger.id is set": bool(trigger_id),
            "next_run_time is set": bool(next_run_time),
            "status is active": status_str.lower().endswith("active"),
        }

        print("\n" + "=" * 65)
        print("TRIGGER VERIFICATION RESULTS:")
        for name, ok in checks.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"Model configured: {model_display}")
        print(f"Trigger status:   {status_str}")
        print("=" * 65 + "\n")

        return 0 if all(checks.values()) else 1

    except Exception as exc:
        err_str = str(exc)
        logger.log("❌", f"Trigger creation failed: {exc}")
        if "github-production" in err_str.lower() or "credential" in err_str.lower():
            print(
                "\nTip: The credential 'github-production' is not registered in your project.\n"
                "  • Register credentials in Google AI Studio / Cloud Console, OR\n"
                "  • Re-run with `--no-credential` to test trigger creation without stored secrets.",
                file=sys.stderr,
            )
        return 1

    finally:
        cleanup_trigger(
            client,
            logger,
            trigger_id=trigger_id,
            keep=args.keep,
            schedule=args.schedule,
            time_zone=args.time_zone,
        )


if __name__ == "__main__":
    sys.exit(main())
