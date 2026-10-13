"""Cloud Cron Scheduler for Unattended Antigravity Agents (`client.triggers`).

Registers a recurring cron job on Google's infrastructure that automatically
wakes up an Antigravity agent on a schedule (e.g., every morning at 9:00 AM).
Each scheduled execution reuses the SAME persistent Linux sandbox, so files and
reports accumulate across runs without needing your local machine online.

By default, running this script creates a trigger, verifies that it is active,
and immediately deletes it so no recurring job is left running (and billing).
Pass `--keep` to leave the scheduled trigger active in the cloud.

Usage:
    export GEMINI_API_KEY="..."

    # Verify trigger creation & cleanup safely (no stored secret required)
    uv run python schedule_trigger.py --no-credential --argon

    # Create a custom recurring trigger and keep it active
    uv run python schedule_trigger.py --no-credential --argon --keep \\
        --name "daily-hn-digest" \\
        --schedule "0 9 * * *" \\
        --prompt "Summarize top 5 Hacker News stories into /workspace/digests/"

    # Fire a trigger immediately on demand (without waiting for 9:00 AM)
    uv run python schedule_trigger.py --no-credential --argon --run-now

    # List active triggers, view execution history, or delete a trigger
    uv run python schedule_trigger.py --list
    uv run python schedule_trigger.py --executions <TRIGGER_ID>
    uv run python schedule_trigger.py --delete <TRIGGER_ID>
"""

from __future__ import annotations

import argparse
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
DEFAULT_CREDENTIAL = "github-production"
DEFAULT_PROMPT = (
    "Triage new issues in my-org/my-app and "
    "save a report to /workspace/reports/"
)


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
        description="Create, inspect, and manage cloud cron triggers for Antigravity agents."
    )

    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--list",
        action="store_true",
        help="List all active triggers in your project and exit.",
    )
    action_group.add_argument(
        "--executions",
        type=str,
        metavar="TRIGGER_ID",
        help="List past executions (interaction_id and environment_id) for a trigger and exit.",
    )
    action_group.add_argument(
        "--delete",
        type=str,
        metavar="TRIGGER_ID",
        help="Delete a specific trigger by ID and exit.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="daily-issue-triage",
        help="Display name for the trigger (default: 'daily-issue-triage').",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt the agent will execute every time the schedule fires.",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="0 9 * * *",
        help="5-field cron expression (default: '0 9 * * *' = daily at 9:00 AM).",
    )
    parser.add_argument(
        "--time-zone",
        type=str,
        default="America/New_York",
        help="IANA time zone for the cron schedule (default: 'America/New_York').",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="api.github.com",
        help="Allowed outbound domain for the sandbox network (default: 'api.github.com').",
    )

    cred_group = parser.add_mutually_exclusive_group()
    cred_group.add_argument(
        "--credential",
        type=str,
        default=DEFAULT_CREDENTIAL,
        help=f"Stored credential name injected by the egress proxy (default: '{DEFAULT_CREDENTIAL}').",
    )
    cred_group.add_argument(
        "--no-credential",
        action="store_true",
        help="Omit stored credential requirement so the trigger can be created without pre-registered secrets.",
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

    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Fire the trigger immediately on demand after creating it (client.triggers.run).",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Leave the scheduled trigger active in the cloud instead of deleting it after verification.",
    )
    return parser.parse_args(argv)


def list_triggers(client: genai.Client) -> int:
    """List all scheduled triggers in the project."""
    try:
        resp = client.triggers.list()
        triggers = getattr(resp, "triggers", None) or []
        if not triggers:
            print("No scheduled triggers found in this project.")
            return 0
        print(f"\nScheduled Triggers ({len(triggers)}):")
        for t in triggers:
            tid = getattr(t, "id", "")
            name = getattr(t, "display_name", "unnamed")
            sched = getattr(t, "schedule", "")
            tz = getattr(t, "time_zone", "")
            status = getattr(t, "status", "")
            next_run = getattr(t, "next_run_time", "")
            print(f"  • {tid}  [{status}]  name='{name}'  schedule='{sched}' ({tz})  next={next_run}")
        print()
        return 0
    except errors.APIError as exc:
        print(f"❌ Failed to list triggers ({exc.code}): {exc.message}", file=sys.stderr)
        return 1


def list_trigger_executions(client: genai.Client, trigger_id: str) -> int:
    """List past executions for a given trigger ID."""
    try:
        resp = client.triggers.list_executions(trigger_id=trigger_id)
        executions = getattr(resp, "trigger_executions", None) or []
        if not executions:
            print(f"No executions recorded yet for trigger {trigger_id}.")
            return 0
        print(f"\nExecutions for Trigger {trigger_id} ({len(executions)}):")
        for ex in executions:
            ex_id = getattr(ex, "id", "")
            status = getattr(ex, "status", "")
            inter_id = getattr(ex, "interaction_id", "")
            env_id = getattr(ex, "environment_id", "")
            start = getattr(ex, "start_time", "")
            print(
                f"  • {ex_id}  [{status}]  started={start}\n"
                f"    interaction_id: {inter_id}\n"
                f"    environment_id: {env_id}  (inspect with: uv run python interactive_cli.py --env-id {env_id})"
            )
        print()
        return 0
    except errors.APIError as exc:
        print(f"❌ Failed to list executions ({exc.code}): {exc.message}", file=sys.stderr)
        return 1


def build_interaction_config(
    *,
    prompt: str,
    domain: str,
    credential: str | None,
    model: str | None,
) -> dict[str, Any]:
    """Construct the interaction payload for the scheduled trigger."""
    allowlist_entry: dict[str, str] = {"domain": domain}
    if credential:
        allowlist_entry["credential"] = credential

    interaction: dict[str, Any] = {
        "agent": DEFAULT_AGENT,
        "input": prompt,
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
        logger.log(
            "🔒",
            f"Trigger {trigger_id} left ACTIVE ({schedule} {time_zone}).\n"
            f"          View runs anytime with: uv run python schedule_trigger.py --executions {trigger_id}\n"
            f"          Delete anytime with:    uv run python schedule_trigger.py --delete {trigger_id}",
        )
        return

    logger.log("🧹", "Step 3/3: Deleting test trigger (pass --keep to leave it active)...")
    try:
        client.triggers.delete(id=trigger_id)
        logger.log("✓", f"Trigger {trigger_id} deleted cleanly.")
    except errors.APIError as exc:
        logger.log("⚠️", f"Failed to delete trigger {trigger_id} ({exc.code}): {exc.message}")


def main(argv: list[str] | None = None) -> int:
    """Run the trigger management or creation/verification workflow."""
    args = parse_args(argv)
    client = genai.Client()

    if args.list:
        return list_triggers(client)

    if args.executions:
        return list_trigger_executions(client, args.executions)

    if args.delete:
        try:
            client.triggers.delete(id=args.delete)
            print(f"🧹 Deleted trigger: {args.delete}")
            return 0
        except errors.APIError as exc:
            print(f"❌ Failed to delete trigger ({exc.code}): {exc.message}", file=sys.stderr)
            return 1

    model_name: str | None = ARGON_MODEL if args.argon else args.model
    model_display = model_name or DEFAULT_MODEL_LABEL
    credential_name: str | None = None if args.no_credential else args.credential

    logger = Logger()
    trigger_id: str | None = None

    print("\n" + "=" * 65)
    logger.log("▶", f"Step 1/3: Registering cloud cron trigger ({args.schedule} {args.time_zone})...")
    logger.log(" ", f"Name:       {args.name}")
    logger.log(" ", f"Agent:      {DEFAULT_AGENT}")
    logger.log(" ", f"Model:      {model_display}")
    logger.log(" ", f"Domain:     {args.domain} (Credential: {credential_name or 'None'})")
    logger.log(" ", f"Prompt:     '{args.prompt}'")
    print("=" * 65)

    try:
        interaction_cfg = build_interaction_config(
            prompt=args.prompt,
            domain=args.domain,
            credential=credential_name,
            model=model_name,
        )

        trigger = client.triggers.create(
            schedule=args.schedule,
            time_zone=args.time_zone,
            display_name=args.name,
            interaction=interaction_cfg,
        )
        trigger_id = getattr(trigger, "id", None)
        next_run_time = getattr(trigger, "next_run_time", None)

        logger.log("✓", f"Trigger registered! ID: {trigger_id}")
        logger.log("🕒", f"Next scheduled run:     {next_run_time}")

        logger.log("🔍", "Step 2/3: Fetching trigger from API to verify active status...")
        fetched = client.triggers.get(id=trigger_id)
        status_str = str(getattr(fetched, "status", ""))

        if args.run_now and trigger_id:
            logger.log("🚀", f"Firing trigger {trigger_id} immediately (--run-now)...")
            client.triggers.run(trigger_id=trigger_id)
            logger.log("✓", "Immediate execution started in the background!")

        checks = {
            "trigger.id is set": bool(trigger_id),
            "next_run_time is computed": bool(next_run_time),
            "status is active": status_str.lower().endswith("active"),
        }

        print("\n" + "=" * 65)
        print("TRIGGER VERIFICATION RESULTS:")
        for name, ok in checks.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"Trigger ID:       {trigger_id}")
        print(f"Model configured: {model_display}")
        print(f"Trigger status:   {status_str}")
        print("=" * 65 + "\n")

        return 0 if all(checks.values()) else 1

    except errors.APIError as exc:
        err_str = str(exc)
        logger.log("❌", f"Trigger creation failed ({exc.code}): {exc.message}")
        if "github-production" in err_str.lower() or "credential" in err_str.lower():
            print(
                "\nTip: The credential 'github-production' is not registered in your project.\n"
                "  • Register credentials in Google AI Studio / Cloud Console, OR\n"
                "  • Re-run with `--no-credential` to create a trigger without stored secrets.",
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
