"""Test for code image 1 in the tweet: run the agent, then resume in the same sandbox.

Uses stream=True to stream live thoughts, searches, and tool executions
directly to your terminal so you can watch the agent work in real time.

Usage:
    export GEMINI_API_KEY=...
    python test_snippet1_run_resume.py [--keep-env]
"""
import sys
import time
from types import SimpleNamespace

from google import genai

keep_env = "--keep-env" in sys.argv
t_start = time.time()

def log(tag, msg):
    elapsed = time.time() - t_start
    print(f"[{elapsed:5.1f}s] {tag} {msg}", flush=True)

client = genai.Client()

def stream_interaction(prompt, environment, previous_id=None):
    """Executes an Antigravity agent interaction with real-time live event streaming."""
    kwargs = {
        "agent": "antigravity-preview-09-2026",
        "input": prompt,
        "environment": environment,
        "stream": True,
    }
    if previous_id:
        kwargs["previous_interaction_id"] = previous_id

    events = client.interactions.create(**kwargs)
    
    interaction_id = None
    env_id = environment if isinstance(environment, str) and environment != "remote" else None
    collected_text = []

    for e in events:
        etype = getattr(e, "event_type", "")

        if etype == "interaction.created":
            if getattr(e, "interaction", None):
                interaction_id = getattr(e.interaction, "id", None)
                env_id = getattr(e.interaction, "environment_id", env_id)
            log("🚀", f"Sandbox allocated: {env_id or 'provisioning...'}")

        elif etype == "step.start":
            step = getattr(e, "step", None)
            stype = getattr(step, "type", "")
            
            if stype == "thought":
                log("💭", "Agent planning next action...")
            elif "search" in stype:
                args = getattr(step, "arguments", None)
                q = getattr(args, "queries", "") if args else ""
                log("🔍", f"Google Search: {q}")
            elif "code" in stype:
                args = getattr(step, "arguments", None)
                cmd = getattr(args, "command", getattr(args, "code", "")) if args else ""
                log("⚡", f"Shell / Code execution: {cmd[:70]}")
            elif stype == "model_output":
                log("📝", "Agent response:\n" + "-" * 50)
            elif stype:
                log("🔧", f"Tool step: {stype}")

        elif etype == "step.delta":
            delta = getattr(e, "delta", None)
            # Delta can be text delta or thought delta
            text = getattr(delta, "text", None)
            if not text and hasattr(delta, "content"):
                text = getattr(delta.content, "text", "")
            if text:
                sys.stdout.write(text)
                sys.stdout.flush()
                collected_text.append(text)

        elif etype == "interaction.completed":
            if getattr(e, "interaction", None):
                env_id = getattr(e.interaction, "environment_id", env_id)
                interaction_id = getattr(e.interaction, "id", interaction_id)
            print("\n" + "-" * 50)
            log("✓", f"Step finished! (Interaction: {interaction_id})\n")

    return SimpleNamespace(
        id=interaction_id,
        environment_id=env_id,
        output_text="".join(collected_text),
    )

# ---------------- Step 1: Initial Run (verbatim from tweet) ----------------
print("\n" + "=" * 65)
log("▶", "STEP 1/3: Provisioning remote sandbox and running Antigravity agent...")
log(" ", "Prompt: 'Research top 5 sustainable fashion brands & build analysis.html'")
print("=" * 65)

run = stream_interaction(
    prompt="Research the top 5 sustainable fashion brands "
           "and build a dashboard in analysis.html",
    environment="remote",  # fresh, isolated Linux sandbox
)

# ---------------- Step 2: Follow-up in same sandbox (from tweet) -----------
print("=" * 65)
log("▶", f"STEP 2/3: Reusing sandbox ({run.environment_id}) for follow-up...")
log(" ", "Prompt: 'Add a pricing-tier filter to the dashboard'")
print("=" * 65)

run2 = stream_interaction(
    prompt="Add a pricing-tier filter to the dashboard",
    environment=run.environment_id,
    previous_id=run.id,
)

# ---------------- Step 3: Verification & Persistence Probe -----------------
print("=" * 65)
log("▶", f"STEP 3/3: Probing sandbox ({run.environment_id}) to verify persistence...")
print("=" * 65)

probe = stream_interaction(
    prompt="Run `ls -la /workspace` and `grep -ci 'tier' analysis.html` (search for "
           "analysis.html if it is elsewhere). Reply with the raw command output only.",
    environment=run.environment_id,
)

checks = {
    "run.environment_id is set": bool(run.environment_id),
    "run2 reused the same sandbox": run2.environment_id == run.environment_id,
    "run produced output_text": bool(run.output_text),
    "run2 produced output_text": bool(run2.output_text),
    "analysis.html preserved in sandbox": "analysis.html" in (probe.output_text or ""),
}

print("\n" + "=" * 65)
print("FINAL VERIFICATION RESULTS:")
for name, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
print(f"Total pipeline time: {time.time()-t_start:.1f}s")
print(f"Preserved Sandbox ID: {run.environment_id}")
print("=" * 65 + "\n")

# ---------------- Step 4: Export Artifact (before cleanup) -----------------
log("📥", "Step 4: Downloading generated artifact (analysis.html) to local disk...")
downloaded = False
for candidate_path in ["workspace/analysis.html", "analysis.html"]:
    try:
        content = client.environments.files.download(
            environment=run.environment_id,
            path=candidate_path,
        )
        with open("analysis.html", "wb") as f:
            f.write(content)
        log("✓", f"Successfully saved local file: analysis.html ({len(content)} bytes)")
        downloaded = True
        break
    except Exception:
        continue

if not downloaded:
    log("ℹ", "Could not locate analysis.html via direct download.")

checks["analysis.html exported locally"] = downloaded

if not keep_env and run.environment_id:
    try:
        client.environments.delete(id=run.environment_id)
        log("🧹", "Sandbox deleted cleanly.")
    except Exception as e:
        log("ℹ", f"Sandbox retained ({e!r}); auto-expires after 7 days.")

sys.exit(0 if all(checks.values()) else 1)
