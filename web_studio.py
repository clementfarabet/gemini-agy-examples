"""Antigravity Web Studio: Split-Screen Agent Chat, Sandbox Explorer & Live App Preview.

A zero-dependency local web UI (`ThreadingHTTPServer` + Server-Sent Events) for
interacting with a remote Antigravity Managed Agent and its Linux cloud sandbox.

Features:
- Real-Time SSE Chat Stream: Watch live agent steps (thoughts, Google Search queries,
  shell commands, tool calls) and streaming responses.
- Remote `/workspace` File Explorer: Automatically lists files created inside the
  remote Linux container after every turn; inspect source code or download files.
- Live Preview Canvas (Iframe Proxy): Any `.html` or `.svg` file created in `/workspace`
  is proxied directly from the remote sandbox (`/sandbox/<env_id>/workspace/...`) and
  rendered live side-by-side with your chat.
- 1-Click Sandbox Forking (`from_environment`): Clone the current container filesystem
  into a new independent sandbox branch at any point and switch between branches.

Usage:
    export GEMINI_API_KEY="..."

    # Start Web Studio on http://127.0.0.1:8080 and open in browser
    uv run python web_studio.py --open

    # Start with Gemini 4 Argon selected by default or attach to an existing sandbox
    uv run python web_studio.py --argon --env-id <ENVIRONMENT_ID> --port 8080
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
import threading
import time
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
import warnings
import webbrowser

from google import genai
from google.genai import errors

warnings.filterwarnings("ignore", message=".*usage is experimental.*")

DEFAULT_AGENT = "antigravity-preview-09-2026"
DEFAULT_MODEL = "gemini-3.8-flash"
ARGON_MODEL = "gemini-4-argon"


class StudioState:
    """Thread-safe state manager for sandboxes and conversation history per environment."""

    def __init__(
        self,
        client: genai.Client,
        *,
        default_model: str | None = None,
        initial_env_id: str | None = None,
        keep_env: bool = False,
    ) -> None:
        self.client = client
        self.model: str | None = default_model
        self.current_env_id: str | None = initial_env_id
        self.keep_env = keep_env
        self.lock = threading.Lock()
        # Maps environment_id -> last interaction_id for stateful multi-turn continuity
        self.last_interaction_by_env: dict[str, str] = {}
        # Maps environment_id -> human label (e.g. "Sandbox #1", "Fork #2 (from a1b2c3)")
        self.env_labels: dict[str, str] = {}
        self.created_env_ids: set[str] = set()
        if initial_env_id:
            self.env_labels[initial_env_id] = f"Attached ({initial_env_id[:8]})"

    def register_env(self, env_id: str, label: str | None = None, created: bool = True) -> None:
        with self.lock:
            self.current_env_id = env_id
            if created:
                self.created_env_ids.add(env_id)
            if env_id not in self.env_labels:
                idx = len(self.env_labels) + 1
                self.env_labels[env_id] = label or f"Sandbox #{idx} ({env_id[:8]})"

    def to_status_dict(self) -> dict[str, Any]:
        with self.lock:
            return {
                "agent": DEFAULT_AGENT,
                "model": self.model or DEFAULT_MODEL,
                "current_env_id": self.current_env_id,
                "previous_interaction_id": (
                    self.last_interaction_by_env.get(self.current_env_id)
                    if self.current_env_id
                    else None
                ),
                "environments": [
                    {"id": eid, "label": lbl} for eid, lbl in self.env_labels.items()
                ],
            }

    def cleanup(self) -> None:
        if self.keep_env or not self.created_env_ids:
            return
        for eid in list(self.created_env_ids):
            try:
                self.client.environments.delete(id=eid)
                print(f"🧹 Deleted sandbox: {eid}")
            except errors.APIError:
                pass


def summarize_one_line(text: str, max_len: int = 100) -> str:
    lines = [line.strip().strip("*#").strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    combined = " — ".join(lines) if len(lines) > 1 else lines[0]
    combined = " ".join(combined.split())
    if len(combined) > max_len:
        return combined[: max_len - 3].rstrip() + "..."
    return combined


def extract_step_summary(step: Any) -> dict[str, str] | None:
    """Convert an SSE step.start object into a structured step badge event."""
    if not step:
        return None
    step_type = getattr(step, "type", "")
    args = getattr(step, "arguments", None)

    if step_type == "thought":
        return {"kind": "thought", "tag": "PLAN", "detail": "Planning next action..."}
    if step_type == "google_search_call":
        queries = getattr(args, "queries", "") if args else ""
        return {
            "kind": "search",
            "tag": "SEARCH",
            "detail": summarize_one_line(str(queries), max_len=90) or "Google Search",
        }
    if step_type == "code_execution_call":
        cmd = getattr(args, "command", getattr(args, "code", "")) if args else ""
        return {
            "kind": "exec",
            "tag": "EXEC",
            "detail": summarize_one_line(str(cmd), max_len=90) or "Shell execution",
        }
    if step_type == "function_call":
        fn_name = getattr(step, "name", "") or "tool"
        detail = fn_name
        if args:
            for key in ("TargetFile", "AbsolutePath", "path", "CommandLine", "command", "query", "Url"):
                val = args.get(key) if isinstance(args, dict) else getattr(args, key, None)
                if val:
                    detail = f"{fn_name} ({summarize_one_line(str(val), max_len=60)})"
                    break
        return {"kind": "tool", "tag": "TOOL", "detail": detail}
    return None


def list_sandbox_files(client: genai.Client, env_id: str) -> list[dict[str, Any]]:
    """Return all files under `/workspace` in the given remote sandbox."""
    if not env_id:
        return []
    try:
        resp = client.environments.files.list(
            environment=env_id,
            path="workspace",
            recursive=True,
        )
        raw_files = getattr(resp, "files", None) or []
        items: list[dict[str, Any]] = []
        for f in raw_files:
            ftype = str(getattr(f, "type", "FILE")).upper()
            fpath = getattr(f, "path", getattr(f, "name", "")) or ""
            if ftype == "DIRECTORY" or not fpath:
                continue
            rel_path = fpath.removeprefix("workspace/").lstrip("/")
            if not rel_path or rel_path.startswith("."):
                continue
            size_bytes = getattr(f, "size_bytes", None)
            ext = Path(rel_path).suffix.lower()
            items.append(
                {
                    "path": fpath if fpath.startswith("workspace/") else f"workspace/{rel_path}",
                    "name": rel_path,
                    "size_bytes": int(size_bytes) if size_bytes is not None else None,
                    "is_previewable": ext in (".html", ".htm", ".svg"),
                }
            )
        items.sort(key=lambda x: (not x["is_previewable"], x["name"]))
        return items
    except errors.APIError:
        return []


def download_sandbox_bytes(client: genai.Client, env_id: str, remote_path: str) -> tuple[bytes, str] | None:
    """Download raw bytes from a remote sandbox path and guess its MIME type."""
    clean = remote_path.lstrip("/")
    candidates = [clean]
    if not clean.startswith("workspace/"):
        candidates.insert(0, f"workspace/{clean}")

    for candidate in candidates:
        try:
            data = client.environments.files.download(environment=env_id, path=candidate)
            mime, _ = mimetypes.guess_type(candidate)
            if not mime:
                if candidate.endswith((".py", ".md", ".txt", ".json", ".sh", ".toml", ".yaml", ".yml")):
                    mime = "text/plain; charset=utf-8"
                else:
                    mime = "application/octet-stream"
            elif mime.startswith("text/") or mime in ("application/javascript", "application/json"):
                mime = f"{mime}; charset=utf-8"
            return data, mime
        except errors.APIError:
            continue
    return None


STUDIO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Antigravity Web Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-primary: #09090b;
      --bg-secondary: #0f0f12;
      --bg-card: #0c0c0f;
      --bg-subtle: #16161a;
      --bg-hover: #1c1c21;
      --border-subtle: #222226;
      --border-bright: #52525b;
      --text-main: #fafafa;
      --text-muted: #a1a1aa;
      --text-dim: #71717a;
      --accent-emerald: #10b981;
      --accent-rose: #ef4444;
      --font-sans: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
      --font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, monospace;
    }

    [data-theme="light"] {
      --bg-primary: #ffffff;
      --bg-secondary: #fafafa;
      --bg-card: #ffffff;
      --bg-subtle: #f4f4f5;
      --bg-hover: #e4e4e7;
      --border-subtle: #e4e4e7;
      --border-bright: #a1a1aa;
      --text-main: #09090b;
      --text-muted: #52525b;
      --text-dim: #71717a;
      --accent-emerald: #059669;
      --accent-rose: #dc2626;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; -webkit-font-smoothing: antialiased; }
    body {
      background: var(--bg-primary);
      color: var(--text-main);
      font-family: var(--font-sans);
      height: 100vh;
      width: 100vw;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    /* Top Header */
    header {
      height: 46px;
      border-bottom: 1px solid var(--border-subtle);
      padding: 0 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: var(--bg-primary);
      flex-shrink: 0;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      font-family: var(--font-mono);
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-weight: 500;
    }
    .status-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--text-dim);
    }
    .status-dot.ready { background: var(--accent-emerald); }
    .status-dot.busy { background: var(--text-main); animation: blink 1s infinite; }

    @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.25; } }

    .toolbar {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    select.ctrl-select, button.ctrl-btn, input.ctrl-input {
      background: var(--bg-secondary);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      font-family: var(--font-mono);
      font-size: 11px;
      padding: 5px 10px;
      border-radius: 3px;
      cursor: pointer;
      transition: border-color 0.15s ease, background 0.15s ease;
    }
    select.ctrl-select:hover, button.ctrl-btn:hover:not(:disabled) {
      border-color: var(--border-bright);
      background: var(--bg-subtle);
    }
    button.ctrl-btn:disabled {
      opacity: 0.35;
      cursor: not-allowed;
    }
    button.ctrl-btn.primary {
      background: var(--text-main);
      color: var(--bg-primary);
      border-color: var(--text-main);
      font-weight: 500;
    }

    /* Main 3-Column Split Layout */
    main.workspace-grid {
      flex: 1;
      display: grid;
      grid-template-columns: 440px 230px 1fr;
      overflow: hidden;
    }

    /* Column 1: Chat & Live Agent Stream */
    section.chat-pane {
      border-right: 1px solid var(--border-subtle);
      display: flex;
      flex-direction: column;
      background: var(--bg-primary);
      overflow: hidden;
    }
    .pane-header {
      height: 36px;
      padding: 0 14px;
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-family: var(--font-mono);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-dim);
      background: var(--bg-secondary);
      flex-shrink: 0;
    }

    #chat-messages {
      flex: 1;
      overflow-y: auto;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .empty-state {
      margin: auto 0;
      padding: 12px 4px;
    }
    .empty-title {
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 6px;
      letter-spacing: -0.02em;
    }
    .empty-desc {
      font-size: 12px;
      color: var(--text-muted);
      line-height: 1.5;
      margin-bottom: 14px;
    }
    .preset-list {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .preset-btn {
      text-align: left;
      background: var(--bg-secondary);
      border: 1px solid var(--border-subtle);
      border-radius: 3px;
      padding: 8px 10px;
      color: var(--text-muted);
      font-size: 11.5px;
      line-height: 1.4;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .preset-btn:hover {
      border-color: var(--border-bright);
      color: var(--text-main);
      background: var(--bg-subtle);
    }
    .preset-btn strong {
      display: block;
      font-family: var(--font-mono);
      font-size: 10px;
      color: var(--text-main);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 2px;
    }

    .msg {
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-size: 12.5px;
      line-height: 1.55;
    }
    .msg-role {
      font-family: var(--font-mono);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-dim);
      display: flex;
      justify-content: space-between;
    }
    .msg-user .msg-body {
      background: var(--bg-subtle);
      border: 1px solid var(--border-subtle);
      padding: 10px 12px;
      border-radius: 3px;
      color: var(--text-main);
      white-space: pre-wrap;
    }
    .msg-agent .msg-steps {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .step-pill {
      display: flex;
      align-items: baseline;
      gap: 8px;
      font-family: var(--font-mono);
      font-size: 10.5px;
      color: var(--text-muted);
      background: var(--bg-secondary);
      border-left: 2px solid var(--border-bright);
      padding: 4px 8px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .step-tag {
      font-size: 9.5px;
      color: var(--text-dim);
      min-width: 44px;
      flex-shrink: 0;
    }
    .msg-agent .msg-body {
      color: var(--text-muted);
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12.5px;
      padding-top: 4px;
    }

    .chat-input-bar {
      padding: 12px;
      border-top: 1px solid var(--border-subtle);
      background: var(--bg-secondary);
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    textarea#prompt-input {
      width: 100%;
      min-height: 64px;
      max-height: 160px;
      resize: vertical;
      background: var(--bg-primary);
      border: 1px solid var(--border-subtle);
      border-radius: 3px;
      padding: 8px 10px;
      color: var(--text-main);
      font-family: var(--font-sans);
      font-size: 12.5px;
      line-height: 1.45;
      outline: none;
    }
    textarea#prompt-input:focus {
      border-color: var(--border-bright);
    }
    .chat-input-actions {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: var(--font-mono);
      font-size: 10px;
      color: var(--text-dim);
    }

    /* Column 2: Sandbox File Explorer */
    section.files-pane {
      border-right: 1px solid var(--border-subtle);
      display: flex;
      flex-direction: column;
      background: var(--bg-secondary);
      overflow: hidden;
    }
    #file-list {
      flex: 1;
      overflow-y: auto;
      padding: 8px;
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .file-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 6px 8px;
      border-radius: 3px;
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-muted);
      cursor: pointer;
      border: 1px solid transparent;
    }
    .file-item:hover {
      background: var(--bg-subtle);
      color: var(--text-main);
    }
    .file-item.active {
      background: var(--bg-subtle);
      border-color: var(--border-bright);
      color: var(--text-main);
    }
    .file-name {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .file-badge {
      font-size: 9px;
      color: var(--text-dim);
      flex-shrink: 0;
      margin-left: 6px;
    }

    /* Column 3: Live Preview & Code Inspector Canvas */
    section.canvas-pane {
      display: flex;
      flex-direction: column;
      background: var(--bg-primary);
      overflow: hidden;
    }
    .canvas-tabs {
      display: flex;
      gap: 4px;
    }
    .canvas-tab {
      background: transparent;
      border: 1px solid transparent;
      color: var(--text-dim);
      font-family: var(--font-mono);
      font-size: 10px;
      padding: 2px 8px;
      border-radius: 2px;
      cursor: pointer;
      text-transform: uppercase;
    }
    .canvas-tab.active {
      border-color: var(--border-subtle);
      background: var(--bg-primary);
      color: var(--text-main);
    }

    #canvas-body {
      flex: 1;
      position: relative;
      overflow: hidden;
      display: flex;
    }
    iframe#preview-frame {
      width: 100%;
      height: 100%;
      border: none;
      background: #ffffff;
      display: none;
    }
    pre#code-viewer {
      width: 100%;
      height: 100%;
      overflow: auto;
      padding: 18px;
      font-family: var(--font-mono);
      font-size: 12px;
      line-height: 1.55;
      color: var(--text-main);
      background: var(--bg-primary);
      display: none;
    }
    .canvas-placeholder {
      margin: auto;
      text-align: center;
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-dim);
      max-width: 360px;
      line-height: 1.6;
    }

    @media (max-width: 1024px) {
      main.workspace-grid { grid-template-columns: 360px 180px 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span class="status-dot" id="status-dot"></span>
      <span>ANTIGRAVITY WEB STUDIO</span>
      <span style="color: var(--text-dim);" id="env-badge">NO SANDBOX</span>
    </div>

    <div class="toolbar">
      <select class="ctrl-select" id="env-select" onchange="switchEnvironment(this.value)" title="Active Remote Sandbox">
        <option value="">Sandbox: Auto-provision on prompt</option>
      </select>
      <button class="ctrl-btn" id="fork-btn" onclick="forkSandbox()" disabled title="Clone current sandbox filesystem via from_environment">
        Fork Sandbox
      </button>
      <button class="ctrl-btn" onclick="newSandbox()" title="Start in a fresh sandbox">
        New Sandbox
      </button>
      <select class="ctrl-select" id="model-select" onchange="changeModel(this.value)" title="Underlying Gemini Model">
        <option value="gemini-3.8-flash">Model: gemini-3.8-flash</option>
        <option value="gemini-4-argon">Model: gemini-4-argon</option>
      </select>
      <button class="ctrl-btn" onclick="toggleTheme()" id="theme-btn">LIGHT</button>
    </div>
  </header>

  <main class="workspace-grid">
    <!-- Left: Chat & Live Agent Stream -->
    <section class="chat-pane">
      <div class="pane-header">
        <span>AGENT INTERACTION STREAM</span>
        <span id="token-counter">0 TOKENS</span>
      </div>

      <div id="chat-messages">
        <div class="empty-state" id="empty-state">
          <div class="empty-title">Remote Linux Sandbox + Live App Canvas</div>
          <div class="empty-desc">
            Every prompt executes inside an isolated Google Cloud Linux container. Ask the agent to build web apps, run Python benchmarks, or synthesize research into <code>/workspace/app.html</code>—it will render live in the right-hand canvas.
          </div>
          <div class="preset-list">
            <button class="preset-btn" onclick="usePreset('Build a sleek, minimalist interactive Conway\\'s Game of Life simulator in `/workspace/app.html` with play/pause, speed slider, step counter, pattern presets (Glider Gun, Pulsar, Spaceship), and click-to-toggle cells.')">
              <strong>01 / INTERACTIVE SIMULATION</strong>
              Build a minimalist Conway's Game of Life simulator in /workspace/app.html
            </button>
            <button class="preset-btn" onclick="usePreset('Write a Python script in `/workspace/bench.py` that benchmarks 4 sorting/searching techniques on 100,000 numbers, runs it to collect real timings, and generates a minimalist interactive benchmark report in `/workspace/app.html`.')">
              <strong>02 / PYTHON BENCHMARK + LIVE CHART</strong>
              Benchmark 4 algorithms in Python and render results in /workspace/app.html
            </button>
            <button class="preset-btn" onclick="usePreset('Search the web for the latest breakthroughs in fusion energy in 2026 and build a clean, minimalist interactive 5-slide briefing deck in `/workspace/app.html`.')">
              <strong>03 / LIVE WEB RESEARCH DECK</strong>
              Research fusion energy breakthroughs and build a 5-slide deck in /workspace/app.html
            </button>
          </div>
        </div>
      </div>

      <div class="chat-input-bar">
        <textarea id="prompt-input" placeholder="Ask the remote agent to write code, run commands, or build /workspace/app.html... (Enter to send, Shift+Enter for newline)"></textarea>
        <div class="chat-input-actions">
          <span id="stream-status">IDLE</span>
          <button class="ctrl-btn primary" id="send-btn" onclick="sendPrompt()">Send Prompt &rarr;</button>
        </div>
      </div>
    </section>

    <!-- Middle: Remote Sandbox File Explorer -->
    <section class="files-pane">
      <div class="pane-header">
        <span>/WORKSPACE FILES</span>
        <button class="canvas-tab" onclick="refreshFiles()" title="Refresh sandbox files">↻</button>
      </div>
      <div id="file-list">
        <div style="padding: 12px 8px; font-family: var(--font-mono); font-size: 10.5px; color: var(--text-dim);">
          No files in /workspace yet.
        </div>
      </div>
    </section>

    <!-- Right: Live Preview & Code Viewer Canvas -->
    <section class="canvas-pane">
      <div class="pane-header">
        <span id="active-file-label">LIVE ARTIFACT CANVAS</span>
        <div class="canvas-tabs">
          <button class="canvas-tab active" id="tab-preview" onclick="setCanvasMode('preview')">Preview</button>
          <button class="canvas-tab" id="tab-code" onclick="setCanvasMode('code')">Source</button>
          <button class="canvas-tab" id="tab-open" onclick="openInNewTab()" style="display:none;">↗</button>
        </div>
      </div>
      <div id="canvas-body">
        <div class="canvas-placeholder" id="canvas-placeholder">
          When the agent creates files in <code>/workspace</code> (such as <code>app.html</code> or <code>report.html</code>), they will appear in the file explorer and render live here.
        </div>
        <iframe id="preview-frame" sandbox="allow-scripts allow-same-origin allow-forms"></iframe>
        <pre id="code-viewer"><code id="code-content"></code></pre>
      </div>
    </section>
  </main>

  <script>
    let currentEnvId = null;
    let selectedFile = null;
    let canvasMode = 'preview';
    let cumulativeTokens = 0;
    let isStreaming = false;

    async function loadStatus() {
      const res = await fetch('/api/status');
      const data = await res.json();
      currentEnvId = data.current_env_id;
      document.getElementById('model-select').value = data.model || 'gemini-3.8-flash';
      updateEnvUI(data);
      if (currentEnvId) {
        await refreshFiles();
      }
    }

    function updateEnvUI(statusData) {
      const dot = document.getElementById('status-dot');
      const badge = document.getElementById('env-badge');
      const forkBtn = document.getElementById('fork-btn');
      const select = document.getElementById('env-select');

      if (currentEnvId) {
        dot.className = isStreaming ? 'status-dot busy' : 'status-dot ready';
        badge.textContent = currentEnvId.slice(0, 12) + '…';
        forkBtn.disabled = isStreaming;
      } else {
        dot.className = isStreaming ? 'status-dot busy' : 'status-dot';
        badge.textContent = 'NO SANDBOX';
        forkBtn.disabled = true;
      }

      if (statusData && statusData.environments) {
        select.innerHTML = '';
        if (statusData.environments.length === 0) {
          const opt = document.createElement('option');
          opt.value = '';
          opt.textContent = 'Sandbox: Auto-provision on prompt';
          select.appendChild(opt);
        } else {
          statusData.environments.forEach(env => {
            const opt = document.createElement('option');
            opt.value = env.id;
            opt.textContent = env.label;
            if (env.id === currentEnvId) opt.selected = true;
            select.appendChild(opt);
          });
        }
      }
    }

    function usePreset(text) {
      const input = document.getElementById('prompt-input');
      input.value = text;
      input.focus();
      sendPrompt();
    }

    async function sendPrompt() {
      if (isStreaming) return;
      const input = document.getElementById('prompt-input');
      const prompt = input.value.trim();
      if (!prompt) return;

      const empty = document.getElementById('empty-state');
      if (empty) empty.remove();

      input.value = '';
      setStreamingState(true, 'STREAMING FROM SANDBOX...');

      const container = document.getElementById('chat-messages');

      // Append user message
      const userDiv = document.createElement('div');
      userDiv.className = 'msg msg-user';
      userDiv.innerHTML = `<div class="msg-role"><span>YOU</span></div><div class="msg-body"></div>`;
      userDiv.querySelector('.msg-body').textContent = prompt;
      container.appendChild(userDiv);

      // Append agent response container
      const agentDiv = document.createElement('div');
      agentDiv.className = 'msg msg-agent';
      agentDiv.innerHTML = `
        <div class="msg-role"><span>ANTIGRAVITY AGENT</span><span class="turn-meta"></span></div>
        <div class="msg-steps"></div>
        <div class="msg-body"></div>
      `;
      container.appendChild(agentDiv);
      const stepsBox = agentDiv.querySelector('.msg-steps');
      const bodyBox = agentDiv.querySelector('.msg-body');
      const metaBox = agentDiv.querySelector('.turn-meta');
      container.scrollTop = container.scrollHeight;

      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt, env_id: currentEnvId }),
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split('\\n\\n');
          buffer = parts.pop();

          for (const block of parts) {
            const line = block.trim();
            if (!line.startsWith('data:')) continue;
            const payload = JSON.parse(line.slice(5).trim());

            if (payload.type === 'env_ready') {
              currentEnvId = payload.env_id;
              updateEnvUI(payload.status);
            } else if (payload.type === 'step') {
              const pill = document.createElement('div');
              pill.className = 'step-pill';
              pill.innerHTML = `<span class="step-tag">${payload.tag}</span><span class="step-detail"></span>`;
              pill.querySelector('.step-detail').textContent = payload.detail;
              stepsBox.appendChild(pill);
              container.scrollTop = container.scrollHeight;
            } else if (payload.type === 'thought') {
              const pill = document.createElement('div');
              pill.className = 'step-pill';
              pill.innerHTML = `<span class="step-tag">THOUGHT</span><span class="step-detail"></span>`;
              pill.querySelector('.step-detail').textContent = payload.text;
              stepsBox.appendChild(pill);
              container.scrollTop = container.scrollHeight;
            } else if (payload.type === 'text') {
              bodyBox.textContent += payload.text;
              container.scrollTop = container.scrollHeight;
            } else if (payload.type === 'done') {
              if (payload.tokens) {
                cumulativeTokens += payload.tokens;
                document.getElementById('token-counter').textContent = `${cumulativeTokens.toLocaleString()} TOKENS`;
                metaBox.textContent = `${payload.elapsed_s}s · ${payload.tokens.toLocaleString()} tok`;
              }
              updateEnvUI(payload.status);
              await refreshFiles(true);
            } else if (payload.type === 'error') {
              bodyBox.textContent += `\\n[Error: ${payload.message}]`;
            }
          }
        }
      } catch (err) {
        bodyBox.textContent += `\\n[Connection error: ${err.message}]`;
      } finally {
        setStreamingState(false, 'READY');
      }
    }

    function setStreamingState(active, label) {
      isStreaming = active;
      document.getElementById('send-btn').disabled = active;
      document.getElementById('stream-status').textContent = label;
      const dot = document.getElementById('status-dot');
      if (active) {
        dot.className = 'status-dot busy';
      } else {
        dot.className = currentEnvId ? 'status-dot ready' : 'status-dot';
      }
    }

    async function refreshFiles(autoSelectPreview = false) {
      if (!currentEnvId) return;
      const res = await fetch(`/api/files?env_id=${encodeURIComponent(currentEnvId)}`);
      const data = await res.json();
      const listEl = document.getElementById('file-list');

      if (!data.files || data.files.length === 0) {
        listEl.innerHTML = `<div style="padding: 12px 8px; font-family: var(--font-mono); font-size: 10.5px; color: var(--text-dim);">No files in /workspace yet.</div>`;
        return;
      }

      listEl.innerHTML = '';
      data.files.forEach(f => {
        const item = document.createElement('div');
        item.className = 'file-item' + (selectedFile && selectedFile.path === f.path ? ' active' : '');
        const sizeLabel = f.size_bytes != null ? `${(f.size_bytes / 1024).toFixed(1)}K` : '';
        item.innerHTML = `<span class="file-name"></span><span class="file-badge">${f.is_previewable ? 'HTML' : sizeLabel}</span>`;
        item.querySelector('.file-name').textContent = f.name;
        item.onclick = () => selectFile(f);
        listEl.appendChild(item);
      });

      if (autoSelectPreview) {
        const htmlFile = data.files.find(f => f.is_previewable);
        if (htmlFile) {
          await selectFile(htmlFile, 'preview');
        } else if (!selectedFile && data.files.length > 0) {
          await selectFile(data.files[0], 'code');
        } else if (selectedFile) {
          await selectFile(selectedFile, canvasMode);
        }
      }
    }

    async function selectFile(fileObj, preferredMode = null) {
      selectedFile = fileObj;
      if (preferredMode) {
        canvasMode = preferredMode;
      } else if (!fileObj.is_previewable && canvasMode === 'preview') {
        canvasMode = 'code';
      } else if (fileObj.is_previewable && !preferredMode) {
        canvasMode = 'preview';
      }
      document.getElementById('active-file-label').textContent = `/workspace/${fileObj.name}`;
      document.getElementById('tab-open').style.display = 'inline-block';
      await renderCanvas();
      await refreshFiles(false);
    }

    async function setCanvasMode(mode) {
      canvasMode = mode;
      await renderCanvas();
    }

    async function renderCanvas() {
      document.getElementById('tab-preview').classList.toggle('active', canvasMode === 'preview');
      document.getElementById('tab-code').classList.toggle('active', canvasMode === 'code');

      const placeholder = document.getElementById('canvas-placeholder');
      const iframe = document.getElementById('preview-frame');
      const codePre = document.getElementById('code-viewer');

      if (!selectedFile || !currentEnvId) {
        placeholder.style.display = 'block';
        iframe.style.display = 'none';
        codePre.style.display = 'none';
        return;
      }

      placeholder.style.display = 'none';
      const rawUrl = `/sandbox/${encodeURIComponent(currentEnvId)}/${selectedFile.path}?t=${Date.now()}`;

      if (canvasMode === 'preview' && selectedFile.is_previewable) {
        codePre.style.display = 'none';
        iframe.style.display = 'block';
        iframe.src = rawUrl;
      } else {
        iframe.style.display = 'none';
        codePre.style.display = 'block';
        const res = await fetch(rawUrl);
        const text = await res.text();
        document.getElementById('code-content').textContent = text;
      }
    }

    function openInNewTab() {
      if (!selectedFile || !currentEnvId) return;
      window.open(`/sandbox/${encodeURIComponent(currentEnvId)}/${selectedFile.path}`, '_blank');
    }

    async function forkSandbox() {
      if (!currentEnvId || isStreaming) return;
      setStreamingState(true, 'FORKING SANDBOX FILESYSTEM...');
      try {
        const res = await fetch('/api/fork', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ env_id: currentEnvId }),
        });
        const data = await res.json();
        if (data.fork_env_id) {
          currentEnvId = data.fork_env_id;
          updateEnvUI(data.status);
          await refreshFiles(true);
          const container = document.getElementById('chat-messages');
          const note = document.createElement('div');
          note.className = 'step-pill';
          note.innerHTML = `<span class="step-tag">FORKED</span><span>Cloned filesystem into new sandbox ${data.fork_env_id.slice(0, 12)}…</span>`;
          container.appendChild(note);
          container.scrollTop = container.scrollHeight;
        }
      } finally {
        setStreamingState(false, 'READY');
      }
    }

    async function newSandbox() {
      if (isStreaming) return;
      const res = await fetch('/api/new', { method: 'POST' });
      const data = await res.json();
      currentEnvId = data.current_env_id;
      selectedFile = null;
      updateEnvUI(data);
      document.getElementById('file-list').innerHTML = `<div style="padding: 12px 8px; font-family: var(--font-mono); font-size: 10.5px; color: var(--text-dim);">No files in /workspace yet.</div>`;
      await renderCanvas();
    }

    async function switchEnvironment(envId) {
      if (!envId) return;
      const res = await fetch('/api/switch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ env_id: envId }),
      });
      const data = await res.json();
      currentEnvId = data.current_env_id;
      selectedFile = null;
      updateEnvUI(data);
      await refreshFiles(true);
    }

    async function changeModel(modelName) {
      await fetch('/api/model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelName }),
      });
    }

    function toggleTheme() {
      const root = document.documentElement;
      const next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
      root.setAttribute('data-theme', next);
      document.getElementById('theme-btn').textContent = next === 'light' ? 'DARK' : 'LIGHT';
    }

    document.getElementById('prompt-input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendPrompt();
      }
    });

    loadStatus();
  </script>
</body>
</html>
"""


def make_handler(state: StudioState):
    """Create the HTTP request handler bound to the current StudioState."""

    class StudioHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            # Keep terminal logs clean
            return

        def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/":
                raw = STUDIO_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return

            if path == "/api/status":
                self._send_json(state.to_status_dict())
                return

            if path == "/api/files":
                qs = parse_qs(parsed.query)
                env_id = (qs.get("env_id") or [state.current_env_id])[0] or ""
                files = list_sandbox_files(state.client, env_id)
                self._send_json({"env_id": env_id, "files": files})
                return

            if path.startswith("/sandbox/"):
                # Format: /sandbox/<env_id>/<remote_path>
                parts = path.removeprefix("/sandbox/").split("/", 1)
                if len(parts) == 2:
                    env_id = unquote(parts[0])
                    remote_path = unquote(parts[1])
                    downloaded = download_sandbox_bytes(state.client, env_id, remote_path)
                    if downloaded is not None:
                        data, mime = downloaded
                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", mime)
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                        return
                self.send_error(HTTPStatus.NOT_FOUND, "File not found in remote sandbox")
                return

            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/api/model":
                body = self._read_json()
                state.model = body.get("model") or DEFAULT_MODEL
                self._send_json(state.to_status_dict())
                return

            if path == "/api/new":
                with state.lock:
                    state.current_env_id = None
                self._send_json(state.to_status_dict())
                return

            if path == "/api/switch":
                body = self._read_json()
                target = body.get("env_id")
                if target:
                    state.register_env(target, created=False)
                self._send_json(state.to_status_dict())
                return

            if path == "/api/fork":
                body = self._read_json()
                source_env = body.get("env_id") or state.current_env_id
                if not source_env:
                    self._send_json({"error": "No active sandbox to fork"}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    fork_env = state.client.environments.create(from_environment=source_env)
                    fork_id = getattr(fork_env, "id", None)
                    if not fork_id:
                        raise RuntimeError("No environment id returned")
                    label = f"Fork ({fork_id[:8]} ← {source_env[:6]})"
                    state.register_env(fork_id, label=label, created=True)
                    print(f"🌿 Forked sandbox {source_env} -> {fork_id}")
                    self._send_json({"fork_env_id": fork_id, "status": state.to_status_dict()})
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return

            if path == "/api/chat":
                body = self._read_json()
                prompt = (body.get("prompt") or "").strip()
                target_env = body.get("env_id") or state.current_env_id or "remote"

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                def send_sse(event_dict: dict[str, Any]) -> None:
                    msg = f"data: {json.dumps(event_dict)}\n\n".encode("utf-8")
                    self.wfile.write(msg)
                    self.wfile.flush()

                t0 = time.monotonic()
                kwargs: dict[str, Any] = {
                    "agent": DEFAULT_AGENT,
                    "input": prompt,
                    "environment": target_env,
                    "stream": True,
                }
                if state.model and state.model != DEFAULT_MODEL:
                    kwargs["agent_config"] = {"type": "antigravity", "model": state.model}
                prev_id = (
                    state.last_interaction_by_env.get(target_env)
                    if target_env != "remote"
                    else None
                )
                if prev_id:
                    kwargs["previous_interaction_id"] = prev_id

                try:
                    events = state.client.interactions.create(**kwargs)
                    resolved_env: str | None = None if target_env == "remote" else target_env
                    interaction_id: str | None = None
                    tokens = 0

                    for event in events:
                        etype = getattr(event, "event_type", "")

                        if etype == "interaction.created":
                            inter = getattr(event, "interaction", None)
                            if inter:
                                interaction_id = getattr(inter, "id", interaction_id)
                                resolved_env = getattr(inter, "environment_id", resolved_env)
                            if resolved_env:
                                state.register_env(resolved_env, created=(target_env == "remote"))
                                send_sse(
                                    {
                                        "type": "env_ready",
                                        "env_id": resolved_env,
                                        "status": state.to_status_dict(),
                                    }
                                )

                        elif etype == "step.start":
                            badge = extract_step_summary(getattr(event, "step", None))
                            if badge:
                                send_sse({"type": "step", **badge})

                        elif etype == "step.delta":
                            delta = getattr(event, "delta", None)
                            dtype = getattr(delta, "type", "")
                            if dtype == "text":
                                text = getattr(delta, "text", "")
                                if text:
                                    send_sse({"type": "text", "text": text})
                            elif dtype == "thought_summary":
                                content = getattr(delta, "content", None)
                                ttext = getattr(content, "text", "") if content else ""
                                if ttext:
                                    send_sse(
                                        {
                                            "type": "thought",
                                            "text": summarize_one_line(ttext, max_len=95),
                                        }
                                    )

                        elif etype == "step.stop":
                            usage = getattr(event, "usage", None)
                            if usage and getattr(usage, "total_tokens", None):
                                tokens = int(usage.total_tokens)

                        elif etype == "interaction.completed":
                            inter = getattr(event, "interaction", None)
                            if inter:
                                interaction_id = getattr(inter, "id", interaction_id)
                                resolved_env = getattr(inter, "environment_id", resolved_env)
                                usage = getattr(inter, "usage", None)
                                if usage and getattr(usage, "total_tokens", None):
                                    tokens = int(usage.total_tokens)
                            if resolved_env:
                                state.register_env(resolved_env, created=(target_env == "remote"))
                                if interaction_id:
                                    state.last_interaction_by_env[resolved_env] = interaction_id

                            elapsed_s = round(time.monotonic() - t0, 1)
                            print(
                                f"[{elapsed_s:5.1f}s] ✓ Turn complete in {resolved_env} "
                                f"(tokens: {tokens:,})"
                            )
                            send_sse(
                                {
                                    "type": "done",
                                    "env_id": resolved_env,
                                    "interaction_id": interaction_id,
                                    "tokens": tokens,
                                    "elapsed_s": elapsed_s,
                                    "status": state.to_status_dict(),
                                }
                            )

                        elif etype == "error":
                            err = getattr(event, "error", None)
                            send_sse({"type": "error", "message": str(getattr(err, "message", err))})

                except Exception as exc:
                    send_sse({"type": "error", "message": str(exc)})
                return

            self.send_error(HTTPStatus.NOT_FOUND)

    return StudioHandler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Antigravity Web Studio (split-screen chat, sandbox file browser, and live preview)."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1).",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8080,
        help="Local port for the Web Studio server (default: 8080).",
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model",
        type=str,
        default=None,
        help="Initial Gemini model (e.g., gemini-3.8-flash, gemini-4-argon).",
    )
    model_group.add_argument(
        "--argon",
        action="store_true",
        help=f"Start with Gemini 4 Argon ({ARGON_MODEL}).",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default=None,
        help="Attach to an existing remote sandbox environment ID on startup.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Automatically open the Web Studio URL in your default browser.",
    )
    parser.add_argument(
        "--keep-env",
        action="store_true",
        help="Preserve created sandboxes when shutting down the server.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_name = ARGON_MODEL if args.argon else (args.model or DEFAULT_MODEL)
    client = genai.Client()

    state = StudioState(
        client,
        default_model=model_name,
        initial_env_id=args.env_id,
        keep_env=args.keep_env,
    )

    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    url = f"http://{args.host}:{args.port}"

    print(f"🚀 Antigravity Web Studio running at: {url}")
    print(f"   Default Model: {model_name}")
    if args.env_id:
        print(f"   Attached Sandbox: {args.env_id}")
    print("   Press Ctrl+C to stop the server.\n")

    if args.open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Antigravity Web Studio...")
    finally:
        server.server_close()
        state.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
