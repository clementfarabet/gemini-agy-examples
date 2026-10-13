# gemini-agy-examples

Examples and test suites for Google Gemini API Managed Agents running the Google Antigravity agent harness (`antigravity-preview-09-2026`).

![Managed Agent Diagram](managed_agent_diagram.svg)

## Overview

| Script | Description |
| :--- | :--- |
| `test_snippet1_run_resume.py` | `interactions.create` with `environment="remote"` (isolated Linux sandbox), followed by resuming via `environment_id` and verifying sandbox file persistence. |
| `test_snippet2_trigger.py` | `triggers.create` with cron schedule for an always-on agent. Verifies trigger configuration via `triggers.get` and cleans up. |

## Prerequisites

- **Python ≥ 3.10** (e.g., Homebrew Python 3.15+)
- `google-genai` SDK (≥ 2.30.1)
- `GEMINI_API_KEY` set in your environment (recommended via macOS Keychain):

```bash
# Add to Keychain securely
security add-generic-password -U -a "$USER" -s GEMINI_API_KEY -w

# Load in ~/.zshrc
export GEMINI_API_KEY="$(security find-generic-password -a "$USER" -s GEMINI_API_KEY -w 2>/dev/null)"
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Tests

```bash
# Test 1: Run and resume agent interaction with sandbox persistence check
python test_snippet1_run_resume.py

# Test 2: Test always-on cron trigger creation (deletes trigger immediately after verification)
python test_snippet2_trigger.py
```
