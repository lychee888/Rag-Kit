# rag-kit — Install Test Bug Tracker (TEMPORARY working doc)

Created 2026-08-07 during full start-to-finish install test (fresh clone @ 7b14749).
This file is a scratchpad for tracking defects found and their fixes; delete before release.
Full audit evidence: `~/rag-kit-test-notes.md`, `~/rag-install-win2.log`, `~/rag-install-win3.log`,
`~/rag-ingest-test.log`, plus `~/rag-install-linux.log` + `~/rag-download-models.log` inside WSL.

Goal when fixing: **rag-kit must install and run on ANY PC** — CPU-only, old NVIDIA
(Ampere/Ada), new NVIDIA (Blackwell sm_120), ARM (DGX Spark) — with no hard assumptions
about GPU, CUDA version, Python version, or inherited shell environment.

---

## BUG-1 — install-windows.bat crashes at Step 4 with `... was unexpected at this time`
- **Status:** FIXED (working tree) — pending re-verify.
- **File:** `scripts/install-windows.bat:109` (and latent twins at 156, 172, 189, 191; `download-models.bat:59`)
- **Cause:** `echo   Installing PyTorch (CPU)...` sits inside the `) else (` block of
  `if !USE_CUDA! EQU 1 (`. Unescaped `()` in an echo line inside a parenthesized cmd
  block breaks cmd's block parser → batch aborts with exit 255 before installing anything.
  Affects ALL systems (whole block parsed regardless of GPU branch). Reproduced with a
  minimal 6-line repro; escaping as `^(…)` fixes it.
- **Fix applied:** escape all 6 spots with `^(` `)`.
- **Proof:** Z2 (minimal repro): fail → Z2f (escaped): pass. Real install proceeded past Step 4 after fix.
- **Verify:** re-run Windows install.

## BUG-2 — venv package shadowing via inherited PYTHONPATH (Windows) → sentence_transformers unusable
- **Status:** FIXED — verified in-place. `rag_kit/__init__.py` now purges foreign
  site-packages from sys.path at import + pops PYTHONPATH for children; installers
  point PYTHONPATH at a dead temp dir (empty breaks some Python builds).
- **Proof:** with PYTHONPATH polluted to hermes, `import rag_kit` → site-packages = rag-kit-venv only,
  tokenizers 0.22.2 from the venv.
- **Evidence:** first `rag ingest` crashed `ImportError`; real cause (unmasked) =
  `tokenizers==0.23.1` (from `…\hermes-agent\venv\Lib\site-packages`) failing
  transformers 5.14.1's `tokenizers<=0.23.0` requirement, even though rag-kit-venv
  contains the correctly pinned `tokenizers 0.22.2`.
- **Root cause:** this Hermes session exports `PYTHONPATH=hermes-agent;hermes-agent\venv\Lib\site-packages`.
  Every `python` launched from that shell (installer's pip AND runtime `rag`) gets those
  paths inserted into `sys.path` ahead of the venv's own site-packages → wrong versions win.
  pip also *skipped* installing deps it thought were "already satisfied" (e.g. `packaging`).
  `env -u PYTHONPATH` → clean: tokenizers 0.22.2, sentence_transformers OK.
- **Why this is a real bug:** `rag` is normally driven from Hermes-spawned shells;
  a venv should behave like a venv regardless of inherited PYTHONPATH.
- **Fix plan:**
  - `rag_kit/__init__.py`: at import, drop from `sys.path` any `site-packages` entry that
    is not under this interpreter's prefix / base-prefix / user-site (i.e., foreign venv
    leak). Also `os.environ.pop("PYTHONPATH", None)` for child procs.
  - `install-windows.bat` + `install-dgx-spark.sh`: clear PYTHONPATH before `pip install`
    so pip resolves purely against the fresh venv.

## BUG-3 — misleading ImportError message in embed engine
- **Status:** FIXED — embed engine now surfaces the underlying error + a hint.
- **File:** `rag_kit/embed/__init__.py` _load_model (c. line 272)
- **Cause:** `except ImportError as exc: raise ImportError("sentence-transformers is not installed…")`
  masks the underlying cause (e.g. tokenizers version); CLI shows the generic text.
- **Fix:** raise with the real message: `f"sentence-transformers could not be imported ({exc})"`,
  keep `from exc`.

## BUG-4 — install-dgx-spark.sh always runs sudo apt (headless stall: "sudo: timed out")
- **Status:** FIXED — venv check now tests `$PYTHON -m venv --help`; apt only if
  module truly missing; failures are non-fatal with guidance.
- **File:** `scripts/install-dgx-spark.sh` step 2b (c. line 82)
- **Cause:** `if ! command -v python3-venv` can NEVER succeed (`python3-venv` is a package,
  not a command) → script ALWAYS `sudo apt-get install python3-venv python3-pip …`
  even when already installed; with no TTY / no passwordless sudo it hangs→times out.
- **Fix:** check `$PYTHON -m venv --help`; only apt if missing (and guard on apt-get/sudo existing).

## BUG-5 — installed torch can't run on GPUs (both installers hardcode cu124 → Blackwell sm_120 kerfail)
- **Status:** FIXED in scripts — cu128-first selection; pending final verify.
- **Fix applied (both installers):** GPU present → install **cu128** (CUDA 12.8, ships sm_120
  Blackwell kernels AND legacy sm≤90) → fallback **cu124** (old GPUs) → fallback **CPU**.
  CPU-only machines (no nvidia-smi) → small CPU wheel (unchanged).
  aarch64 (DGX) → cu126 → cu124 → CPU.
- **IMPORTANT discovered fact:** on Windows, plain `pip install torch` (PyPI default)
  now resolves to a **CPU-only wheel** (`torch ...+cpu`) — so "install default torch"
  silently loses the GPU. That's why the GPU branch MUST use an explicit CUDA index.
  (Install #4 proved this: GPU detected, default torch → `Torch not compiled with CUDA`.)

## BUG-6 — bash scripts exit non-zero post-success when run non-interactively
- **Status:** FIXED — all `read` guarded with `|| true` + `[ -t 0 ]` TTY check.
- **Files:** `install-dgx-spark.sh` (final `read -r -p "Press Enter…"`; Step 8 model prompt),
  `download-models.sh` (lines 127, 182, 198). `set -euo pipefail` + `read` on EOF → aborts.
- **Fix:** guard all `read` with `|| true` and skip prompting when not a TTY.

## Minor observations (tidied while fixing)
- **OBS-A:** Windows autostart needs admin — prints clear message. Leave as-is (expected).
- **OBS-B:** Step 7 prints "Backfill: complete" even when folder empty — FIXED (message now honest).
- **OBS-C:** Step 5 verify never imported the ML stack — FIXED (sentence_transformers import check added to both installers).
- **OBS-F:** `if %ERRORLEVEL%` immediately after an `if…` block carried stale errorlevel — FIXED (torch result now gated by explicit TORCH_OK flag).
- **OBS-G (new):** reinstalling while a watcher is running left old venv's python.exe locked → venv re-create Permission denied. FIXED — installer now `taskkill /im rag.exe` before rebuilding the venv.
- No `.gitattributes` → EOL depended on per-machine `core.autocrlf`. FIXED — added `.gitattributes` (`*.sh eol=lf`, `*.bat eol=crlf`, `*.py eol=lf`).

## BUG-7 — watcher crashes on stop: `AttributeError: '_WatchHandler' object has no attribute '_pending'`
- **Status:** FIXED + verified.
- **File:** `rag_kit/cli/main.py` `flush_and_stop()` (lines 683-684).
- **Cause:** referenced `self._pending` but the handler stores `self._pending_events`
  (and `_pending_deletes`); on graceful stop the missing attr crashed the watcher
  (systemd journal: `status=1/FAILURE`, `exit-code`). Reproduced in logs during installs.
- **Fix:** use `_pending_events`. Verified: live `rag watch` → SIGINT → clean
  "Stopping watcher... Done.", no traceback.

---

## Verification checklist (after fixes)
- [ ] Windows fresh install completes, Step 4 no parse error.
- [ ] `rag` works WITHOUT unsetting PYTHONPATH (from Hermes shell).
- [ ] `rag ingest` + `rag query` on GPU (RTX 5060 Ti) succeed (torch supports sm_120).
- [ ] CPU path still works (`CUDA_VISIBLE_DEVICES=-1`).
- [ ] Linux (WSL) install completes; non-interactive exit 0; no sudo hang; ingest+query OK.
- [ ] `download-models` scripts don't die on non-interactive input.
