"""Test for code image 1 in the tweet: run the agent, then resume in the same sandbox.

The two client.interactions.create(...) calls match the tweet snippet.
Includes an active animated timer/spinner so you see live progress every second while the agent works.

Usage:
    export GEMINI_API_KEY=...
    python test_snippet1_run_resume.py [--keep-env]
"""
import sys
import time
import threading
from contextlib import contextmanager

from google import genai

keep_env = "--keep-env" in sys.argv
t_start = time.time()

def log(msg):
    elapsed = time.time() - t_start
    print(f"[{elapsed:5.1f}s] {msg}", flush=True)

@contextmanager
def busy_indicator(description):
    """Shows an active spinner and live elapsed seconds while waiting on a blocking API call."""
    stop_event = threading.Event()
    chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    t0 = time.time()

    def spin():
        i = 0
        while not stop_event.is_set():
            elapsed = time.time() - t0
            sys.stdout.write(f"\r  {chars[i % len(chars)]} {description} ({elapsed:.1f}s)... ")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=0.5)
        sys.stdout.write(f"\r  ✓ {description} (done in {time.time()-t0:.1f}s)                \n")
        sys.stdout.flush()

client = genai.Client()

# ---------------- Step 1: Initial Run (verbatim from tweet) ----------------
log("Step 1/3: Provisioning remote sandbox and running Antigravity agent...")
log("  Prompt: 'Research top 5 sustainable fashion brands & build analysis.html'")

with busy_indicator("Agent researching & generating analysis.html in sandbox"):
    run = client.interactions.create(
        agent="antigravity-preview-09-2026",
        input="Research the top 5 sustainable fashion brands "
              "and build a dashboard in analysis.html",
        environment="remote",  # fresh, isolated Linux sandbox
    )

log(f"Step 1 complete!")
log(f"  Environment ID : {run.environment_id}")
log(f"  Interaction ID : {run.id}")
print("\n--- Agent Output (Run 1) ---")
print(run.output_text)
print("----------------------------\n")

# ---------------- Step 2: Follow-up in same sandbox (from tweet) -----------
log("Step 2/3: Reusing sandbox for follow-up modification...")
log("  Prompt: 'Add a pricing-tier filter to the dashboard'...")

with busy_indicator("Agent editing dashboard in existing sandbox"):
    run2 = client.interactions.create(
        agent="antigravity-preview-09-2026",
        input="Add a pricing-tier filter to the dashboard",
        environment=run.environment_id,
        previous_interaction_id=run.id,
    )

log("Step 2 complete!")
print("\n--- Agent Output (Run 2) ---")
print(run2.output_text)
print("----------------------------\n")

# ---------------- Step 3: Verification & Persistence Probe -----------------
log("Step 3/3: Probing sandbox to verify file persistence...")
with busy_indicator("Probing sandbox filesystem"):
    probe = client.interactions.create(
        agent="antigravity-preview-09-2026",
        input="Run `ls -la /workspace` and `grep -ci 'tier' analysis.html` (search for "
              "analysis.html if it is elsewhere). Reply with the raw command output only.",
        environment=run.environment_id,
    )

log("Step 3 complete!")
print("\n--- Persistence Probe Output ---")
print(probe.output_text)
print("--------------------------------\n")

checks = {
    "run.environment_id is set": bool(getattr(run, "environment_id", None)),
    "run2 reused the same sandbox": getattr(run2, "environment_id", None) == run.environment_id,
    "run produced output_text": bool(run.output_text),
    "run2 produced output_text": bool(run2.output_text),
    "analysis.html preserved in sandbox": "analysis.html" in (probe.output_text or ""),
}

print("=" * 60)
print("VERIFICATION RESULTS:")
for name, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
print(f"Total time: {time.time()-t_start:.1f}s")
print(f"Sandbox ID: {run.environment_id}")
print("=" * 60)

if not keep_env:
    try:
        client.environments.delete(id=run.environment_id)
        log("Sandbox deleted cleanly.")
    except Exception as e:
        log(f"Sandbox retained ({e!r}); auto-expires after 7 days.")

sys.exit(0 if all(checks.values()) else 1)
