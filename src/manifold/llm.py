"""Anthropic client construction with an explicit-credential guard.

Manifold's LLM stages (relevance judging, grounded generation) are the only parts that cost
money, and they are meant to run on an API key the operator has chosen explicitly. But
`anthropic.Anthropic()` silently reads whatever credentials are in the environment, and a
Claude Code session exports `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN`.
So running a stage from inside a Claude Code session with no key configured does not fail — it
quietly bills the session's account and stamps the artifact with judgments nobody can
reproduce. That happened once; this module makes it impossible to happen silently.

Three outcomes:

* explicit ``ANTHROPIC_API_KEY`` set  -> proceed (the intended path)
* no key, no session credentials      -> proceed and let the SDK raise its own clear error
* no key, session credentials present -> refuse, unless opted in

The opt-in exists because using the session is legitimate when done deliberately: the
``scripts/cc_judge_export.py`` -> subagents -> ``scripts/cc_judge_ingest.py`` route is the
project's designed $0 judge path and runs on exactly that auth. What is not legitimate is
*accidentally* spending it from the API path. Opt in with ``--allow-session-auth`` or
``MANIFOLD_ALLOW_SESSION_AUTH=1``; either way the choice is printed so it lands in the run log.
"""

from __future__ import annotations

import os

# Credentials a Claude Code session (or a proxy/gateway) exports into child processes.
SESSION_AUTH_VARS = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDECODE")

_REFUSAL = """\
[manifold] REFUSING to call the Anthropic API with inherited session credentials.

No ANTHROPIC_API_KEY is set, but these Claude Code / gateway variables are present:
    {found}
`anthropic.Anthropic()` would use them, billing whatever account is behind this session rather
than the one you intended, and producing judgments that cannot be reproduced from a clean
checkout.

Pick one:
  1. Explicit API key   — put a live ANTHROPIC_API_KEY in .env (check it is not commented out).
  2. Free judge route   — grade with Claude Code subagents instead, no API key and no spend:
                            python scripts/cc_judge_export.py --out <dir>
                            (subagents grade the task files)
                            python scripts/cc_judge_ingest.py --results <dir> --tasks <dir>
  3. Deliberate opt-in  — re-run with --allow-session-auth (or MANIFOLD_ALLOW_SESSION_AUTH=1)
                          if you really do want this session's account to pay.
"""


def session_auth_vars_present() -> list[str]:
    """Names (never values) of session-credential variables visible to this process."""
    return [v for v in SESSION_AUTH_VARS if os.environ.get(v)]


def assert_explicit_credentials(allow_session_auth: bool = False) -> None:
    """Raise SystemExit if we would spend inherited session credentials by accident."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    found = session_auth_vars_present()
    if not found:
        return  # the SDK's own "no api_key" error is clearer than anything we'd add
    if allow_session_auth or os.environ.get("MANIFOLD_ALLOW_SESSION_AUTH") == "1":
        print(f"[manifold] WARNING: no ANTHROPIC_API_KEY; using inherited session credentials "
              f"({', '.join(found)}) because session auth was explicitly allowed. Judgments from "
              f"this run are not reproducible from a clean checkout — record that in the gold set.")
        return
    raise SystemExit(_REFUSAL.format(found=", ".join(found)))


def client(allow_session_auth: bool = False):
    """Guarded `anthropic.Anthropic()`. Use this instead of constructing the client directly."""
    import anthropic

    assert_explicit_credentials(allow_session_auth)
    return anthropic.Anthropic()
