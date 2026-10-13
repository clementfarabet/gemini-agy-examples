"""Test for code image 2 in the tweet: create an always-on cron trigger.

The client.triggers.create(...) call is copied verbatim from the tweet.
By default the trigger is DELETED right after it's verified, so nothing runs every
morning at 9am (and nothing gets billed). Pass --keep to leave it active.

Prerequisite: a stored credential named "github-production" in your project
(see https://ai.google.dev/gemini-api/docs/agent-credentials). Without it,
create() is expected to fail. That's still a useful result: it tells you
the tweet needs a note about credentials.

Usage:
    export GEMINI_API_KEY=...
    .venv/bin/python test_snippet2_trigger.py [--keep]
"""
import sys

from google import genai

keep = "--keep" in sys.argv
client = genai.Client()

# ---------------- snippet (verbatim from the tweet) ----------------
trigger = client.triggers.create(
    schedule="0 9 * * *",
    time_zone="America/New_York",
    display_name="daily-issue-triage",
    interaction={
        "agent": "antigravity-preview-09-2026",
        "input": "Triage new issues in my-org/my-app and "
                 "save a report to /workspace/reports/",
        "environment": {
            "type": "remote",
            "network": {"allowlist": [
                {"domain": "api.github.com",
                 "credential": "github-production"},
            ]},
        },
    },
)
print(trigger.next_run_time)
# -------------------------------------------------------------------

checks = {"trigger.id is set": bool(getattr(trigger, "id", None)),
          "next_run_time is set": bool(getattr(trigger, "next_run_time", None))}

fetched = client.triggers.get(id=trigger.id)
print("Fetched trigger:", fetched)
checks["status is active"] = str(getattr(fetched, "status", "")).lower().endswith("active")

print("\nRESULTS")
for name, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

if keep:
    print(f"\nTrigger {trigger.id} left ACTIVE: it will fire daily at 09:00 New York time.")
else:
    client.triggers.delete(id=trigger.id)
    print(f"\nTrigger {trigger.id} deleted (pass --keep to leave it running).")

sys.exit(0 if all(checks.values()) else 1)
