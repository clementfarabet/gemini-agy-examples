"""Example 3: Interactive local CLI for a remote Antigravity Managed Agent.

Provides a full Read-Eval-Print Loop (REPL) with command history, live SSE
streaming (thoughts, searches, code execution, and responses), direct sandbox
file transfers (/ls, /upload, /download), model switching (/argon, /model),
and independent control over conversation state (/clear) vs. sandbox state (/reset).

Usage:
    export GEMINI_API_KEY="..."

    # Launch interactive REPL (defaults to gemini-3.8-flash)
    uv run python test_snippet3_interactive_cli.py

    # Launch interactive REPL with Gemini 4 Argon
    uv run python test_snippet3_interactive_cli.py --argon

    # Attach to an existing remote sandbox ID
    uv run python test_snippet3_interactive_cli.py --env-id <ENVIRONMENT_ID>

    # Run a single prompt non-interactively and exit
    uv run python test_snippet3_interactive_cli.py -p "Create hello.py and run it"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import mimetypes
from pathlib import Path
try:
    import readline
except ImportError:  # pragma: no cover
    readline = None  # type: ignore[assignment]
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

SLASH_COMMANDS = (
    "/help",
    "/status",
    "/ls",
    "/download",
    "/upload",
    "/model",
    "/argon",
    "/clear",
    "/reset",
    "/envs",
    "/attach",
    "/quit",
    "/exit",
)


@dataclass
class SessionState:
    """Mutable state for the interactive agent session."""

    model: str | None = None
    environment_id: str | None = None
    previous_interaction_id: str | None = None
    keep_env: bool = False
    turns: int = 0
    total_session_tokens: int = 0
    created_env_ids: set[str] = field(default_factory=set)

    @property
    def model_display(self) -> str:
        return self.model or DEFAULT_MODEL_LABEL


class Logger:
    """Elapsed-time terminal logger."""

    def __init__(self) -> None:
        self._start = time.monotonic()

    def reset(self) -> None:
        self._start = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def log(self, icon: str, message: str) -> None:
        print(f"[{self.elapsed:5.1f}s] {icon} {message}", flush=True)


def setup_readline() -> None:
    """Configure tab-completion for slash commands and file paths if readline is available."""
    if readline is None:
        return

    def completer(text: str, state: int) -> str | None:
        line = readline.get_line_buffer()
        if line.startswith("/"):
            options = [cmd + " " for cmd in SLASH_COMMANDS if cmd.startswith(text)]
        else:
            options = []
        return options[state] if state < len(options) else None

    readline.set_completer(completer)
    readline.set_completer_delims(" \t\n")
    # macOS libedit vs GNU readline compatibility
    if "libedit" in getattr(readline, "__doc__", ""):
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive local CLI for a remote Antigravity Managed Agent."
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
        help=f"Shorthand to run with Gemini 4 Argon (--model {ARGON_MODEL}).",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default=None,
        help="Attach to an existing remote sandbox environment ID instead of provisioning a new one.",
    )
    parser.add_argument(
        "--keep-env",
        action="store_true",
        help="Preserve the remote sandbox on exit instead of deleting it.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        action="append",
         default=None,
        help="Run one or more prompts non-interactively (can be passed multiple times) and exit.",
    )
    return parser.parse_args(argv)


def format_step_event(step: Any, logger: Logger) -> None:
    """Log human-readable progress for an agent step.start event."""
    step_type = getattr(step, "type", "")
    args = getattr(step, "arguments", None)

    if step_type == "thought":
        logger.log("💭", "Thinking...")
    elif step_type == "google_search_call":
        queries = getattr(args, "queries", "") if args else ""
        logger.log("🔍", f"Google Search: {queries}")
    elif step_type == "code_execution_call":
        cmd = getattr(args, "command", getattr(args, "code", "")) if args else ""
        snippet = str(cmd).strip().replace("\n", " ")[:75]
        if snippet:
            logger.log("⚡", f"Exec: {snippet}")
    elif step_type == "function_call":
        fn_name = getattr(step, "name", "") or "tool"
        logger.log("🔧", f"Tool call: {fn_name}")
    elif step_type == "model_output":
        logger.log("📝", "Response:\n" + "-" * 60)
    elif step_type and not step_type.endswith("_result"):
        logger.log("🔧", f"Step: {step_type}")


def run_agent_turn(
    client: genai.Client,
    session: SessionState,
    prompt: str,
) -> bool:
    """Stream a single conversational turn with the remote Antigravity agent."""
    logger = Logger()
    target_env = session.environment_id or "remote"

    kwargs: dict[str, Any] = {
        "agent": DEFAULT_AGENT,
        "input": prompt,
        "environment": target_env,
        "stream": True,
    }
    if session.model:
        kwargs["agent_config"] = {"type": "antigravity", "model": session.model}
    if session.previous_interaction_id:
        kwargs["previous_interaction_id"] = session.previous_interaction_id

    try:
        events = client.interactions.create(**kwargs)
    except errors.APIError as exc:
        logger.log("❌", f"API Error ({exc.code}): {exc.message}")
        if exc.code == 429 or "RESOURCE_EXHAUSTED" in str(exc):
            print(
                "  Tip: Gemini 4 Argon requires a billed Google AI Studio / GCP project.\n"
                "  Use `/model gemini-3.8-flash` to switch back to the default model.",
                file=sys.stderr,
            )
        return False

    interaction_id: str | None = None
    env_id: str | None = session.environment_id
    turn_tokens: int | None = None

    try:
        for event in events:
            event_type = getattr(event, "event_type", "")

            if event_type == "interaction.created":
                interaction = getattr(event, "interaction", None)
                if interaction:
                    interaction_id = getattr(interaction, "id", interaction_id)
                    env_id = getattr(interaction, "environment_id", env_id)
                if env_id and env_id != session.environment_id:
                    session.environment_id = env_id
                    session.created_env_ids.add(env_id)
                    logger.log("🚀", f"Sandbox provisioned: {env_id}")
                else:
                    logger.log("🔄", f"Connected to sandbox: {env_id or 'remote'}")

            elif event_type == "step.start":
                format_step_event(getattr(event, "step", None), logger)

            elif event_type == "step.delta":
                delta = getattr(event, "delta", None)
                delta_type = getattr(delta, "type", "")

                if delta_type == "text":
                    text = getattr(delta, "text", "")
                    if text:
                        sys.stdout.write(text)
                        sys.stdout.flush()
                elif delta_type == "thought_summary":
                    content = getattr(delta, "content", None)
                    thought_text = getattr(content, "text", "") if content else ""
                    if thought_text:
                        logger.log("💡", f"Thought: {thought_text.strip()}")

            elif event_type == "step.stop":
                usage = getattr(event, "usage", None)
                if usage and getattr(usage, "total_tokens", None):
                    turn_tokens = usage.total_tokens

            elif event_type == "interaction.completed":
                interaction = getattr(event, "interaction", None)
                if interaction:
                    env_id = getattr(interaction, "environment_id", env_id)
                    interaction_id = getattr(interaction, "id", interaction_id)
                    usage = getattr(interaction, "usage", None)
                    if usage and getattr(usage, "total_tokens", None):
                        turn_tokens = usage.total_tokens

                if env_id:
                    if not session.environment_id:
                        session.created_env_ids.add(env_id)
                    session.environment_id = env_id
                if interaction_id:
                    session.previous_interaction_id = interaction_id

                session.turns += 1
                if turn_tokens:
                    session.total_session_tokens += turn_tokens

                print("\n" + "-" * 60)
                token_str = f" | {turn_tokens:,} tokens" if turn_tokens else ""
                logger.log(
                    "✓",
                    f"Turn {session.turns} complete ( Sandbox: {session.environment_id}{token_str} )",
                )

            elif event_type == "error":
                err = getattr(event, "error", None)
                code = getattr(err, "code", "unknown")
                msg = getattr(err, "message", str(err))
                logger.log("❌", f"Stream error [{code}]: {msg}")
                return False

    except KeyboardInterrupt:
        print("\n")
        logger.log("⚠️", "Turn interrupted by user (Ctrl+C). Sandbox remains active.")
        return False

    return True


def print_help() -> None:
    """Print available interactive slash commands."""
    print(
        "\nAvailable Slash Commands:\n"
        "  /help                             Show this help message\n"
        "  /status                           Show current sandbox, model, and session stats\n"
        "  /ls [remote_path]                 List files in the remote sandbox (default: workspace)\n"
        "  /download <remote> [local_path]   Download a file from the sandbox to local disk\n"
        "  /upload <local_path> [remote]     Upload a local file into the remote sandbox\n"
        "  /model [model_name]               Switch underlying model (or omit name to use default)\n"
        "  /argon                            Shorthand for `/model gemini-4-argon`\n"
        "  /clear                            Clear chat history while keeping sandbox files\n"
        "  /reset                            Delete current sandbox and start fresh on next turn\n"
        "  /envs                             List active remote sandboxes in your project\n"
        "  /attach <environment_id>          Switch session to an existing remote sandbox ID\n"
        "  /quit, /exit                      Exit CLI and clean up sandbox (unless --keep-env)\n"
    )


def handle_slash_command(
    client: genai.Client,
    session: SessionState,
    raw_input: str,
) -> bool:
    """Execute a slash command. Returns False if the REPL should exit, True otherwise."""
    parts = raw_input.strip().split()
    cmd = parts[0].lower()
    args = parts[1:]

    if cmd in ("/quit", "/exit"):
        return False

    if cmd == "/help":
        print_help()
        return True

    if cmd == "/status":
        print(
            f"\nSession Status:\n"
            f"  Agent:             {DEFAULT_AGENT}\n"
            f"  Underlying Model:  {session.model_display}\n"
            f"  Sandbox ID:        {session.environment_id or '(not provisioned yet — send a prompt first)'}\n"
            f"  Last Interaction:  {session.previous_interaction_id or '(none)'}\n"
            f"  Completed Turns:   {session.turns}\n"
            f"  Cumulative Tokens: {session.total_session_tokens:,}\n"
            f"  Keep Sandbox:      {session.keep_env}\n"
        )
        return True

    if cmd == "/argon":
        session.model = ARGON_MODEL
        print(f"⚙ Switched model to: {session.model_display}")
        return True

    if cmd == "/model":
        session.model = args[0] if args else None
        print(f"⚙ Switched model to: {session.model_display}")
        return True

    if cmd == "/clear":
        session.previous_interaction_id = None
        print(
            f"🧹 Conversation history cleared. Sandbox ({session.environment_id or 'none'}) "
            "files are preserved for your next prompt."
        )
        return True

    if cmd == "/reset":
        if session.environment_id:
            try:
                client.environments.delete(id=session.environment_id)
                print(f"🧹 Deleted sandbox: {session.environment_id}")
                session.created_env_ids.discard(session.environment_id)
            except errors.APIError as exc:
                print(f"⚠️ Could not delete sandbox ({exc.code}): {exc.message}")
        session.environment_id = None
        session.previous_interaction_id = None
        print("✨ Session reset. A fresh sandbox will be provisioned on your next prompt.")
        return True

    if cmd == "/envs":
        try:
            resp = client.environments.list(page_size=20)
            envs = getattr(resp, "environments", None) or []
            if not envs:
                print("  No active remote environments found.")
            else:
                print(f"\nActive Remote Environments ({len(envs)}):")
                for env in envs:
                    eid = getattr(env, "id", "")
                    status = getattr(env, "status", "")
                    marker = " (current)" if eid == session.environment_id else ""
                    print(f"  • {eid}  [{status}]{marker}")
                print()
        except errors.APIError as exc:
            print(f"❌ Failed to list environments ({exc.code}): {exc.message}")
        return True

    if cmd == "/attach":
        if not args:
            print("Usage: /attach <environment_id>")
            return True
        session.environment_id = args[0]
        session.previous_interaction_id = None
        print(f"🔗 Attached to sandbox: {session.environment_id} (conversation context reset)")
        return True

    if cmd == "/ls":
        if not session.environment_id:
            print("ℹ No active sandbox yet. Send a prompt or `/attach <env_id>` first.")
            return True
        remote_path = args[0] if args else "workspace"
        try:
            resp = client.environments.files.list(
                environment=session.environment_id,
                path=remote_path,
                recursive=True,
            )
            files = getattr(resp, "files", None) or []
            if not files:
                print(f"  (No files found under '{remote_path}')")
            else:
                print(f"\nFiles in {session.environment_id}:{remote_path}:")
                for f in files:
                    ftype = getattr(f, "type", "FILE")
                    fpath = getattr(f, "path", getattr(f, "name", ""))
                    fsize = getattr(f, "size_bytes", None)
                    size_str = f" ({int(fsize):,} B)" if fsize is not None else ""
                    icon = "📁" if str(ftype).upper() == "DIRECTORY" else "📄"
                    print(f"  {icon} {fpath}{size_str}")
                print()
        except errors.APIError as exc:
            print(f"❌ Could not list '{remote_path}' ({exc.code}): {exc.message}")
        return True

    if cmd == "/download":
        if not session.environment_id:
            print("ℹ No active sandbox yet. Send a prompt or `/attach <env_id>` first.")
            return True
        if not args:
            print("Usage: /download <remote_path> [local_path]")
            return True
        remote_req = args[0]
        local_dest = Path(args[1]) if len(args) > 1 else Path(Path(remote_req).name)

        # Try exact path first, then fallback to workspace/<path> if relative
        candidates = [remote_req]
        if not remote_req.startswith("workspace/"):
            candidates.append(f"workspace/{remote_req.lstrip('/')}")

        for candidate in candidates:
            try:
                data = client.environments.files.download(
                    environment=session.environment_id,
                    path=candidate,
                )
                local_dest.parent.mkdir(parents=True, exist_ok=True)
                local_dest.write_bytes(data)
                print(f"📥 Downloaded {candidate} -> {local_dest} ({len(data):,} bytes)")
                return True
            except errors.APIError:
                continue

        print(f"❌ File '{remote_req}' not found in sandbox {session.environment_id}.")
        return True

    if cmd == "/upload":
        if not session.environment_id:
            print("ℹ No active sandbox yet. Send a prompt or `/attach <env_id>` first.")
            return True
        if not args:
            print("Usage: /upload <local_path> [remote_path]")
            return True
        local_src = Path(args[0]).expanduser()
        if not local_src.is_file():
            print(f"❌ Local file not found: {local_src}")
            return True
        remote_dest = args[1] if len(args) > 1 else f"workspace/{local_src.name}"
        mime_type, _ = mimetypes.guess_type(local_src.name)
        try:
            with local_src.open("rb") as fp:
                res = client.environments.files.upload(
                    environment=session.environment_id,
                    path=remote_dest,
                    file=fp,
                    mime_type=mime_type or "application/octet-stream",
                    overwrite=True,
                )
            uploaded_files = getattr(res, "files", None) or []
            size_info = (
                f" ({uploaded_files[0].size_bytes} bytes)"
                if uploaded_files and getattr(uploaded_files[0], "size_bytes", None)
                else ""
            )
            print(f"📤 Uploaded {local_src} -> {remote_dest}{size_info}")
        except errors.APIError as exc:
            print(f"❌ Upload failed ({exc.code}): {exc.message}")
        return True

    print(f"Unknown slash command: {cmd}. Type `/help` for available commands.")
    return True


def cleanup_session_environments(client: genai.Client, session: SessionState) -> None:
    """Clean up any sandboxes provisioned during this CLI session unless --keep-env is set."""
    if not session.created_env_ids:
        return
    if session.keep_env:
        for env_id in session.created_env_ids:
            print(f"🔒 Sandbox retained (--keep-env): {env_id}")
        return

    for env_id in list(session.created_env_ids):
        try:
            client.environments.delete(id=env_id)
            print(f"🧹 Deleted sandbox: {env_id}")
        except errors.APIError as exc:
            print(f"ℹ Sandbox {env_id} retained ({exc.code}); auto-expires after 7 days.")


def main(argv: list[str] | None = None) -> int:
    """Entry point for the interactive Antigravity CLI."""
    args = parse_args(argv)
    model_name: str | None = ARGON_MODEL if args.argon else args.model

    session = SessionState(
        model=model_name,
        environment_id=args.env_id,
        keep_env=args.keep_env,
    )
    client = genai.Client()

    try:
        # Non-interactive batch mode if -p / --prompt was supplied
        if args.prompt:
            for prompt_item in args.prompt:
                text = prompt_item.strip()
                if not text:
                    continue
                if text.startswith("/"):
                    if not handle_slash_command(client, session, text):
                        break
                else:
                    if not run_agent_turn(client, session, text):
                        return 1
            return 0

        # Interactive REPL mode
        setup_readline()
        print("\n" + "=" * 65)
        print("🚀 Antigravity Remote Agent Interactive CLI")
        print(f"   Agent:   {DEFAULT_AGENT}")
        print(f"   Model:   {session.model_display}")
        print(f"   Sandbox: {session.environment_id or 'remote (provisioned on first prompt)'}")
        print("   Type a prompt to begin, `/help` for commands, or `/quit` to exit.")
        print("=" * 65 + "\n")

        while True:
            env_badge = session.environment_id[:8] if session.environment_id else "new"
            model_badge = "argon" if session.model == ARGON_MODEL else (session.model or "3.8-flash")
            try:
                user_input = input(f"\n[agy:{model_badge}|{env_badge}] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                should_continue = handle_slash_command(client, session, user_input)
                if not should_continue:
                    break
            else:
                run_agent_turn(client, session, user_input)

        return 0

    finally:
        cleanup_session_environments(client, session)


if __name__ == "__main__":
    sys.exit(main())
