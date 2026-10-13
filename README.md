# gemini-agy-examples

Examples and CLI tools for **Google Gemini API Managed Agents** running the Google Antigravity agent harness (`antigravity-preview-09-2026`).

![Managed Agent Diagram](managed_agent_diagram.svg)

## Overview: Three Ways to Run an Antigravity Agent

| Script | Mode | What It Does |
| :--- | :--- | :--- |
| [`run_deep_research.py`](run_deep_research.py) | **Automated Pipeline** | Provisions a fresh remote Linux sandbox, researches any topic via Google Search, builds an interactive `analytics.html` dashboard, refines it statefully in a second pass, downloads the HTML locally, and deletes the sandbox. |
| [`interactive_cli.py`](interactive_cli.py) | **Interactive REPL** | Opens a live terminal chat session attached to a remote Linux sandbox. Supports multi-turn coding/research, live SSE streaming, and direct file transfers (`/ls`, `/upload`, `/download`, `/fork`, `/attach`, `/detach`). |
| [`schedule_trigger.py`](schedule_trigger.py) | **Cloud Cron Job** | Registers an unattended **cloud cron schedule** (`client.triggers`) on Google's servers so an agent wakes up automatically (e.g., every day at 9:00 AM) without your machine running. Every scheduled run reuses the **same persistent sandbox** so files accumulate over time. |

---

## Prerequisites

- **Python ≥ 3.10** and [**uv**](https://docs.astral.sh/uv/) (`brew install uv`)
- `GEMINI_API_KEY` set in your environment (recommended via macOS Keychain):

```bash
# Store key in macOS Keychain
security add-generic-password -U -a "$USER" -s GEMINI_API_KEY -w

# Load in ~/.zshrc
export GEMINI_API_KEY="$(security find-generic-password -a "$USER" -s GEMINI_API_KEY -w 2>/dev/null)"
```

## Setup

```bash
uv sync --frozen
```

---

## 1. Deep Research & Analytics Dashboard (`run_deep_research.py`)

Runs a multi-step research workflow in a remote sandbox and exports a self-contained `analytics.html` dashboard to your local machine.

```bash
# Research any topic and generate analytics.html (powered by Gemini 4 Argon)
uv run python run_deep_research.py "Global solid-state battery landscape in 2026" --argon

# Pass custom follow-up refinement instructions in the same sandbox and open in browser
uv run python run_deep_research.py "Open-source Rust web frameworks" --argon \
  --refine "Add a benchmark comparison chart and an interactive use-case filter" \
  --open

# Single-pass mode (skip the follow-up refinement turn)
uv run python run_deep_research.py "Quantum error correction breakthroughs" --argon --no-refine
```

---

## 2. Interactive Remote Agent CLI (`interactive_cli.py`)

A local REPL for working interactively inside a remote Antigravity Linux sandbox.

```bash
# Launch interactive session powered by Gemini 4 Argon
uv run python interactive_cli.py --argon

# Attach to an existing remote sandbox ID (e.g. from a previous session or a cron trigger)
uv run python interactive_cli.py --argon --env-id <ENVIRONMENT_ID>
```

**Built-in REPL Slash Commands:**
- `/new` | `/fork` — Provision a fresh sandbox immediately or clone the current sandbox's filesystem
- `/ls [path]` — List files in the remote sandbox (auto-provisions a sandbox if needed)
- `/upload <local> [remote]` — Upload a local file into the sandbox (auto-provisions if needed)
- `/download <remote> [local]` — Download a file from the sandbox to your local machine
- `/model <name>` | `/argon` — Switch the underlying Gemini model on the fly
- `/clear` — Clear conversation history while keeping all files in the sandbox
- `/reset` — Delete the current sandbox and provision a fresh one
- `/envs` | `/attach <id>` | `/detach` — List active sandboxes, attach to one, or detach while leaving it alive
- `/status` | `/help` | `/quit` — View session token usage, help, or exit cleanly

---

## 3. Cloud Cron Scheduler (`schedule_trigger.py`)

### How Triggers Work
Unlike `run_deep_research.py` and `interactive_cli.py` (which run interactively while your terminal is open), **`client.triggers`** registers a **persistent cloud cron job** on Google's infrastructure:

1. **Unattended Execution**: Google's scheduler wakes the agent up on your cron expression (e.g., `"0 9 * * *"` in `America/New_York`).
2. **Persistent Shared Sandbox**: Every time the trigger fires, it reuses the **same remote Linux sandbox** (`environment_id`), so reports, git clones, or state files saved on Day 1 are still on disk on Day 2.
3. **Egress Proxy Credential Injection**: Because a trigger runs unattended, you can lock its network access to specific domains (e.g., `api.github.com`) and reference a stored credential name (`credential="github-production"`). Google's egress proxy injects the secret on the fly so raw tokens never enter the sandbox or prompt.
4. **Safe Default Behavior**: Running `schedule_trigger.py` without `--keep` registers the cron trigger, verifies that Google's scheduler accepted it and computed `next_run_time`, and **immediately deletes it** so you don't accidentally leave a recurring job billing tokens every morning.

### Commands

```bash
# 1. Safe test run: create a trigger, verify it is active, and immediately delete it
uv run python schedule_trigger.py --no-credential --argon

# 2. Create a custom daily trigger and KEEP it active in the cloud (--keep)
uv run python schedule_trigger.py --no-credential --argon --keep \
  --name "daily-hn-digest" \
  --schedule "0 9 * * *" \
  --time-zone "America/New_York" \
  --prompt "Check top stories on news.ycombinator.com and save a markdown summary to /workspace/digests/"

# 3. Fire a trigger immediately on demand without waiting for the cron time (--run-now)
uv run python schedule_trigger.py --no-credential --argon --keep --run-now

# 4. Inspect active triggers, view past runs (and their sandbox IDs), or delete a trigger
uv run python schedule_trigger.py --list
uv run python schedule_trigger.py --executions <TRIGGER_ID>
uv run python schedule_trigger.py --delete <TRIGGER_ID>
```
