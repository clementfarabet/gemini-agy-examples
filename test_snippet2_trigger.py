"""Test for code image 2 in the tweet: create an always-on cron trigger.

The client.triggers.create(...) call is copied verbatim from the tweet.
By default the trigger is DELETED right after it's verified, so nothing runs every
morning at 9am (and nothing gets billed). Pass --keep to leave it active.

Prerequisite note: A stored credential named "github-production" is specified in the network
allowlist. If your project does not have that credential configured, the API will return a
clear validation error explaining that the credential resource was not found.

Usage:
    export GEMINI_API_KEY=...
    python test_snippet2_trigger.py [--keep]
"""
import sys
import time

from google import genai

keep = "--keep" in sys.argv
t_start = time.time()

def log(msg):
    elapsed = time.time() - t_start
    print(f"[{elapsed:5.1f}s] {msg}", flush=True)

client = genai.Client()

log("Step 1/3: Creating scheduled trigger (daily at 09:00 NY time)...")
log("  Agent: antigravity-preview-09-2026")

try:
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
    print(f"Next run time: {trigger.next_run_time}")
    # -------------------------------------------------------------------

    log(f"Trigger created successfully! ID: {trigger.id}")
    checks = {
        "trigger.id is set": bool(getattr(trigger, "id", None)),
        "next_run_time is set": bool(getattr(trigger, "next_run_time", None)),
    }

    log("Step 2/3: Fetching trigger to verify active status...")
    fetched = client.triggers.get(id=trigger.id)
    checks["status is active"] = str(getattr(fetched, "status", "")).lower().endswith("active")

    print("\n" + "=" * 60)
    print("TRIGGER VERIFICATION RESULTS:")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("=" * 60 + "\n")

    if keep:
        log(f"Trigger {trigger.id} left ACTIVE (daily at 09:00 America/New_York).")
    else:
        log("Step 3/3: Cleaning up trigger...")
        client.triggers.delete(id=trigger.id)
        log(f"Trigger {trigger.id} deleted successfully.")

    sys.exit(0 if all(checks.values()) else 1)

except Exception as e:
    log(f"Trigger creation failed: {e}")
    if "github-production" in str(e).lower() or "credential" in str(e).lower():
        print("\nNote: This failed because the credential 'github-production' has not been registered")
        print("in your Google project. You can register credentials in AI Studio / Cloud Console,")
        print("or run triggers without stored credentials if no authenticated egress is needed.")
    sys.exit(1)
