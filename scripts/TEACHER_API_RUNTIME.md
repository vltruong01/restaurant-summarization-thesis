# Teacher API runtime

The teacher uses an independent API key. It does not require changing Codex's
provider, model selection, configuration, authentication, or ChatGPT login.
Do not read or write `~/.codex/config.toml` or `~/.codex/auth.json` for teacher work.

The teacher endpoint remains `https://www.coletoken.de/v1`; the requested teacher
model remains `gpt-6.1-sol`, selected by the existing study configuration. The API
key authenticates the request; it does not select the model.

Use `stage_g_teacher_credentials.credentials()` in future teacher runners.
It reads `STAGE_G_TEACHER_API_KEY` from the process environment, or an explicitly
configured teacher-only file outside the repository and Codex home. Alternatively,
an explicitly requested execution can ask for the key through a hidden terminal
prompt. Keys are never printed, included in logs or placed in notebooks.
There is no fallback to Codex login files or the generic `OPENAI_API_KEY` variable.

Optional persistent teacher-only setup (hidden input, no API call):

```powershell
.\.venv-caption-pilot\Scripts\python.exe scripts\setup_stage_g_teacher_key.py
```

The default secret file is `thesis_teacher/api_key.txt` under LOCALAPPDATA (or
`~/.local/share` on other systems). `STAGE_G_TEACHER_API_KEY_FILE` can name another
private path. Repository and Codex-home paths are rejected. This secret file must
not be added to scientific exports, logs, Git or review bundles.

The new D-040 runner imports this isolated resolver directly. Its offline preparation
uses `scripts/run_stage_g_d040_focused_pilot.py --prepare`; the explicit bounded
execution uses `--execute-pilot`, with no automatic retries or production rollout.

Offline status from the project directory:

```powershell
.\.venv-caption-pilot\Scripts\python.exe scripts\run_stage_g_teacher_isolated.py --check-config
```

The isolated launcher can only resume the already approved bounded D-039 pilot,
using `--execute-d039-pilot`, and refuses if its completion report already exists.
It changes the credential resolver in memory only. It preserves historical runner
files, prepared-input hashes, prompts, generation settings and existing outputs.
Do not invoke old runner files directly: they are retained as historical evidence
and contain the former Codex-dependent resolver.

D-039 is currently complete. No API calls are needed for this operational change.
Production generation remains pending researcher review; this launcher does not
authorize it. A future approved producer must use the isolated resolver and record
its new runner hash without modifying previous pilot manifests or results.
