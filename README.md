# gemini-agy-examples

Examples and tools for **Google Gemini API Managed Agents** running the Google Antigravity agent harness (`antigravity-preview-09-2026`).

![Managed Agent Diagram](managed_agent_diagram.svg)

## Overview

| Script | Description |
| :--- | :--- |
| [`run_deep_research.py`](run_deep_research.py) | Generic Deep Research & Dashboard generator: provisions an isolated remote Linux sandbox, researches any topic via Google Search, builds an interactive `analytics.html` dashboard, refines it statefully in the same sandbox, and exports the final HTML locally. |
| [`schedule_trigger.py`](schedule_trigger.py) | Creates an always-on cron trigger (`triggers.create`) with domain allowlisting and optional credential injection, verifies its active status via `triggers.get`, and cleans up automatically. |
| [`interactive_cli.py`](interactive_cli.py) | Full interactive local REPL for chatting with a remote Antigravity agent in real time, with live SSE streaming, tab-completed slash commands (`/new`, `/fork`, `/ls`, `/upload`, `/download`, `/model`, `/argon`, `/clear`, `/reset`, `/envs`, `/attach`, `/detach`), and automatic sandbox lifecycle management. |

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

## Usage

### 1. Deep Research & Analytics Dashboard Generator (`run_deep_research.py`)

```bash
# Research any topic and generate analytics.html (powered by Gemini 4 Argon)
uv run python run_deep_research.py "Global solid-state battery landscape in 2026" --argon

# Pass custom follow-up refinement instructions in the same sandbox and open in browser
uv run python run_deep_research.py "Open-source Rust web frameworks" --argon \
  --refine "Add a benchmark comparison chart and an interactive use-case filter" \
  --open

# Single-pass mode (skip the follow-up refinement turn)
uv run python run_deep_research.py "Quantum error correction breakthroughs" --no-refine

# Default run (researches sustainable fashion brands if no topic is provided)
uv run python run_deep_research.py
```

### 2. Scheduled Cron Trigger (`schedule_trigger.py`)

```bash
# Test full trigger creation, verification, and cleanup without requiring a pre-registered secret
uv run python schedule_trigger.py --no-credential

# Configure the scheduled trigger to use Gemini 4 Argon
uv run python schedule_trigger.py --no-credential --argon

# Run with the default 'github-production' stored credential
uv run python schedule_trigger.py

# Leave the trigger active on the schedule (default: 09:00 America/New_York)
uv run python schedule_trigger.py --no-credential --keep
```

### 3. Interactive Remote Agent CLI (`interactive_cli.py`)

```bash
# Launch interactive session (provisions a remote Linux sandbox on your first prompt)
uv run python interactive_cli.py

# Launch interactive session powered by Gemini 4 Argon
uv run python interactive_cli.py --argon

# Attach to an existing remote sandbox ID
uv run python interactive_cli.py --env-id <ENVIRONMENT_ID>

# Built-in REPL Slash Commands:
#   /new | /fork                Provision a fresh sandbox or clone the current one
#   /ls [path]                  List files in the remote sandbox (auto-provisions if needed)
#   /upload <local> [remote]    Upload a local file into the sandbox (auto-provisions if needed)
#   /download <remote> [local]  Download a file from the sandbox to local disk
#   /model <name> | /argon      Switch the underlying model on the fly
#   /clear                      Clear conversation history while keeping sandbox files
#   /reset                      Delete current sandbox and provision a fresh one
#   /envs | /attach | /detach   List, attach to, or detach from remote sandboxes
#   /status | /help | /quit     Inspect session status, help, or exit cleanly
```
