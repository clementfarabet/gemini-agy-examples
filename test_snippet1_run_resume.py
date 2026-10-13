"""Test for code image 1 in the tweet: run the agent, then resume in the same sandbox.

The two client.interactions.create(...) calls are copied verbatim from the tweet.
Everything else here checks the results and cleans up.

Usage:
    export GEMINI_API_KEY=...
    .venv/bin/python test_snippet1_run_resume.py [--keep-env]
"""
import sys

from google import genai

keep_env = "--keep-env" in sys.argv

# ---------------- snippet (verbatim from the tweet) ----------------
client = genai.Client()

run = client.interactions.create(
    agent="antigravity-preview-09-2026",
    input="Research the top 5 sustainable fashion brands "
          "and build a dashboard in analysis.html",
    environment="remote",  # fresh, isolated Linux sandbox
)
print(run.output_text)

# Follow-up: same sandbox, same files
run2 = client.interactions.create(
    agent="antigravity-preview-09-2026",
    input="Add a pricing-tier filter to the dashboard",
    environment=run.environment_id,
    previous_interaction_id=run.id,
)
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("run2 output:\n", run2.output_text)
print("=" * 70)

checks = {
    "run.environment_id is set": bool(getattr(run, "environment_id", None)),
    "run2 reused the same sandbox": getattr(run2, "environment_id", None) == run.environment_id,
    "run produced output_text": bool(run.output_text),
    "run2 produced output_text": bool(run2.output_text),
}

# Ask the agent to prove the file from run 1 is still there (persistence check).
probe = client.interactions.create(
    agent="antigravity-preview-09-2026",
    input="Run `ls -la /workspace` and `grep -ci 'tier' analysis.html` (search for "
          "analysis.html if it is elsewhere). Reply with the raw command output only.",
    environment=run.environment_id,
)
print("\nPersistence probe:\n", probe.output_text)
checks["analysis.html mentioned in probe"] = "analysis.html" in (probe.output_text or "")

print("\nRESULTS")
for name, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
print(f"\nenvironment_id: {run.environment_id}")

if not keep_env:
    try:
        client.environments.delete(id=run.environment_id)
        print("Sandbox deleted (pass --keep-env to keep it).")
    except Exception as e:  # SDK method name not verified; sandbox auto-expires after 7 days anyway
        print(f"Could not delete sandbox ({e!r}); it will auto-expire 7 days after last use.")

sys.exit(0 if all(checks.values()) else 1)
