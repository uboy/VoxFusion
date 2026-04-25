# VoxFusion — Architecture, Code & UX Review

**Date:** 2026-04-25
**Scope:** Full project review — architecture, code quality, UI/UX, testing, security
**Version:** 0.1.0 (commit 3b02e29 + uncommitted GigaAM multi-variant changes)

---

## 1. Architecture

### 1.1 Strengths

- **Protocol-based interfaces** (PEP 544) — correct approach for Python 3.11+; all component
  boundaries use `typing.Protocol`, not ABCs. Enables structural subtyping and easy mocking.
- **Modular pipeline:** Capture → Preprocessing → ASR → Diarization → Translation → Output.
  Each stage is independently replaceable and testable.
- **Graceful degradation:** Missing pyannote → fallback to channel diarization; no CUDA → CPU;
  no FFmpeg → warning banner. System never crashes due to missing optional dependencies.
- **6-level configuration hierarchy:** defaults.yaml → system → user → project → env vars → CLI
  flags. Covers all deployment scenarios.
- **`src/` layout** prevents accidental imports from the repo root.
- **Frozen dataclasses** for all data models — immutability prevents accidental mutations in pipeline.

### 1.2 Issues

#### ARCH-1: Streaming pipeline not implemented
**Severity:** Medium (design debt)
**Location:** `pipeline/streaming.py`, `live_gigaam/session.py`

The architecture document claims "streaming-first design" but `pipeline/streaming.py` is
minimal. Live GigaAM uses its own orchestration in `live_gigaam/session.py`, completely
bypassing the standard pipeline. Two parallel codepaths for the same conceptual operation.

**Options:**
1. **Rename to batch-first** — Update ARCHITECTURE.md to accurately describe the current state.
   Lowest effort, honest documentation. *Recommended for now.*
2. **Unify live and batch** — Refactor `live_gigaam/session.py` to use a streaming pipeline
   interface that `BatchPipeline` also implements. Higher effort, better long-term architecture.
3. **Extract streaming protocol** — Define a `StreamingPipeline` protocol that both live and
   batch implement, without full refactoring. Middle ground.

**Why fix:** Architectural docs that don't match reality mislead contributors and create
confusion about where to add new functionality.

---

#### ARCH-2: Duplicate revision mappings (DRY violation)
**Severity:** Low (maintenance risk)
**Location:** `asr/gigaam_engine.py:44` (`_GIGAAM_REVISIONS`), `cli/models_cmd.py:52` (`_GIGAAM_IDS`)

Both files define the mapping from model ID to HuggingFace revision branch. If a new variant
is added, both must be updated — easy to forget one.

**Options:**
1. **Move to `asr_catalog.py`** — Add a `revision` field to `ASRModelInfo` or a standalone
   `GIGAAM_REVISIONS` dict next to the catalog. Both `gigaam_engine.py` and `models_cmd.py`
   import from there. *Recommended.*
2. **Add a helper function** — `get_gigaam_revision(model_id) -> str | None` in the catalog
   module. Cleaner API but slightly more indirection.

**Why fix:** Single source of truth prevents divergence. Currently `_GIGAAM_IDS` in
`models_cmd.py` has an extra `"gigaam"` shorthand that `_GIGAAM_REVISIONS` doesn't, meaning
the engine can't handle the shorthand — a latent bug.

---

#### ARCH-3: GUI is a God Object (~1000+ lines)
**Severity:** Medium (maintainability)
**Location:** `gui/main.py`

`TranscriptionGUI` handles: window layout, tab construction, event handling, worker lifecycle,
settings persistence, LLM integration, device enumeration, model selection, file queuing, and
progress tracking — all in one class.

**Options:**
1. **Extract tab classes** — `LiveCaptureTab` and `FileTranscriptionTab` as separate classes,
   each managing their own widgets and worker interactions. `TranscriptionGUI` becomes a thin
   shell. *Recommended as first step.*
2. **Extract settings manager** — `SettingsManager` class handles JSON persistence, loading,
   saving. Removes ~150 lines from main class.
3. **Extract LLM panel** — `LLMPanel` handles Open WebUI configuration, prompt editing,
   streaming, chunking. Removes ~200 lines.
4. **Full MVP/presenter pattern** — Add an application service layer between GUI and pipeline.
   Highest effort, best testability.

**Why fix:** A 1000+ line class is hard to navigate, hard to test, and attracts further
bloat. Extracting tabs alone would reduce the class by ~60%.

---

#### ARCH-4: No service layer between GUI and pipeline
**Severity:** Low-Medium (testability, reusability)
**Location:** `gui/main.py`, `gui/runtime.py`

GUI directly creates `PipelineOrchestrator`, `RecordingWorker`, `FileTranscribeWorker`.
Business logic is coupled to the UI layer.

**Options:**
1. **Introduce `TranscriptionService`** — Encapsulates "transcribe these files with these
   settings" logic. GUI and CLI both call this service. *Recommended.*
2. **Keep current approach** — For a single-developer project, the direct coupling is
   pragmatic. Accept the trade-off but document it.

**Why fix:** Enables testing pipeline integration without Tkinter. Also makes it possible
to add alternative UIs (web, TUI) without duplicating orchestration logic.

---

#### ARCH-5: Megatron compat shim is fragile
**Severity:** Low (but hard to debug when it breaks)
**Location:** `asr/gigaam_engine.py:116` (`_install_megatron_compat_shim`)

Fake `megatron.core.num_microbatches_calculator` module injected into `sys.modules`. If
NeMo or pyannote update their Megatron expectations, this silently breaks.

**Options:**
1. **Add runtime warning** — Log at `WARNING` level when the shim activates, so it's
   visible in debug logs. *Recommended.*
2. **Add version guard** — Only install shim for known-compatible library versions.
3. **Pin NeMo version** — Avoid the problem by freezing the dependency version.

**Why fix:** Silent shims create debugging nightmares. A warning makes the cause obvious.

---

## 2. ASR Subsystem

### 2.1 Strengths

- 5 backends: faster-whisper, GigaAM (4 variants), Parakeet, Breeze, OpenVINO
- Model catalog with rich metadata (accuracy, speed, languages, required packages)
- Runtime availability detection via `is_model_available()` with lazy imports
- CUDA VRAM probe with automatic CPU fallback
- Quality presets (Fast/Balanced/Quality) for faster-whisper

### 2.2 Issues

#### ASR-1: GigaAM chunking through temp files
**Severity:** Low (performance)
**Location:** `asr/gigaam_engine.py:348-360`

Each 24-second chunk is written to a temp WAV file, then read back by the model's
`.transcribe()` method. This adds I/O overhead (~10-20ms per chunk).

**Options:**
1. **Pass numpy array directly** — If the GigaAM model accepts tensor/array input in
   addition to file paths, bypass the file I/O. Requires checking model API.
   *Recommended if feasible.*
2. **Use tmpfs/ramdisk** — Set `tempfile.tempdir` to `/dev/shm` on Linux. Eliminates
   disk I/O without changing the API. Simple and safe.
3. **Keep current approach** — 10-20ms overhead per 24s chunk is negligible compared to
   inference time. Pragmatic choice.

**Why fix:** Mostly a cleanliness issue. The I/O overhead is small relative to inference.

---

#### ASR-2: No retry for transient network errors
**Severity:** Medium (reliability)
**Location:** `asr/gigaam_engine.py:252`, `cli/models_cmd.py`

Model download fails immediately on network errors (500, timeout). No retry with backoff.

**Options:**
1. **Add retry with backoff** — Use `tenacity` or manual retry (3 attempts, exponential
   backoff) for `AutoModel.from_pretrained()` and `snapshot_download()`. *Recommended.*
2. **Wrap in user-facing retry prompt** — CLI asks "Download failed. Retry? [Y/n]".
   More interactive but doesn't help batch/headless usage.
3. **Rely on huggingface_hub retry** — The HF client has built-in retry for some errors.
   May be sufficient without additional logic.

**Why fix:** Network hiccups during multi-GB downloads are common. A single retry would
save many user-reported "download failed" issues.

---

#### ASR-3: Crude token estimation for LLM context
**Severity:** Low (UX)
**Location:** `gui/main.py` (LLM chunking logic)

Token count estimated as `len(text) / 2`. For Russian text this significantly undercounts
(1 Russian char ≈ 2-3 BPE tokens), leading to context overflow errors.

**Options:**
1. **Better heuristic** — `len(text.encode('utf-8')) / 4` is closer for multilingual text.
   Zero dependencies. *Recommended.*
2. **Use tiktoken** — Accurate BPE counting, but adds a dependency (~3 MB).
3. **Conservative multiplier** — `len(text) * 1.5` for Russian. Simple, overestimates
   slightly (better than underestimating).

**Why fix:** Context overflow triggers chunking fallback, which is slow and produces
worse results than getting the chunk size right on the first pass.

---

## 3. Diarization

### 3.1 Strengths

- 4 strategies: none, channel, ML (pyannote), hybrid
- Chunked diarization for large files (5-min chunks, 10s overlap)
- Alignment algorithm matches ASR segments to speaker turns
- Auto-strategy: ML for files, channel for live

### 3.2 Issues

#### DIAR-1: ML diarization requires HF token + license acceptance
**Severity:** Medium (UX barrier)
**Location:** `diarization/pyannote_engine.py`

Users must: (1) create HuggingFace account, (2) accept pyannote model license,
(3) generate token, (4) enter token in settings. This is a multi-step external process.

**Options:**
1. **Add setup wizard** — GUI dialog that guides through all steps with links and
   validation. *Recommended.*
2. **Document clearly** — Add a "Setting up ML diarization" section to the README
   with screenshots. Lowest effort.
3. **Bundle a non-gated alternative** — If a non-gated diarization model exists,
   offer it as a zero-config default. Research needed.

**Why fix:** ML diarization is a key differentiator. If it's hard to set up, users
won't use it.

---

## 4. UI/UX

### 4.1 Strengths

- Two-tab interface (Live + File) — logical separation
- Tooltips on every control with help icons
- Localization (EN, RU, ZH) with dynamic language switching
- Settings persistence in JSON
- LLM integration with chunking, streaming, custom prompts
- Workflow hints guide users through steps

### 4.2 Issues

#### UX-1: Tkinter framework limitations
**Severity:** Medium (user perception, accessibility)

No native screen reader support, no HiDPI scaling, no dark mode, dated appearance
on all platforms (especially macOS).

**Options:**
1. **CustomTkinter** — Drop-in replacement with modern look. Minimal code changes,
   adds dark mode, better scaling. *Recommended as next step.*
2. **Keep Tkinter, add DPI awareness** — Call `ctypes.windll.shcore.SetProcessDpiAwareness(1)`
   on Windows, `root.tk.call('tk', 'scaling', factor)` on others. Partial fix.
3. **Qt/PySide6 migration** — Best cross-platform UI, native look, full accessibility.
   But complete rewrite of GUI layer (~weeks of work).
4. **Web UI (Gradio/Streamlit)** — Zero desktop dependencies, runs in browser. Good for
   remote/server deployments. Different UX paradigm.

**Why fix:** First impressions matter. A modern-looking UI increases user trust and adoption.

---

#### UX-2: Errors shown only by color
**Severity:** Low-Medium (accessibility)
**Location:** `gui/main.py` (status labels)

Red text for errors, orange for warnings. Colorblind users (~8% of men) may not
distinguish these from normal text.

**Options:**
1. **Add text prefixes** — `[ERROR]`, `[WARNING]` before messages. Zero effort,
   universal. *Recommended.*
2. **Add icons** — Unicode symbols (⚠️, ❌, ✅) before messages. Visually clearer.
3. **Both** — Prefix + icon + color. Belt and suspenders.

**Why fix:** Accessibility with zero effort. No reason not to do it.

---

#### UX-3: No cancel button for long operations
**Severity:** Medium (UX)
**Location:** `gui/main.py` (file transcription flow)

`FileTranscribeWorker` has `cancel()` method but the GUI has no Cancel button.
Users must close the entire app to stop a long transcription.

**Options:**
1. **Add Cancel button** — Show it during transcription, wire to `worker.cancel()`.
   *Recommended.* Simple implementation.
2. **Add Cancel to progress bar area** — Inline cancel icon next to the progress bar.
   Cleaner UX but slightly more layout work.

**Why fix:** Transcription of large files can take minutes. Users need a way out.

---

#### UX-4: No drag-and-drop for files
**Severity:** Low-Medium (UX convenience)
**Location:** `gui/main.py` (file tab)

Desktop transcription app without drag-and-drop feels incomplete.

**Options:**
1. **tkinterdnd2** — Third-party package that adds native drag-and-drop to Tkinter.
   Well-maintained, MIT license. *Recommended.*
2. **Accept it** — File dialog is functional. Drag-and-drop is nice-to-have.
3. **Add with CustomTkinter migration** — CustomTkinter has better DnD support.
   Bundle with UX-1 fix.

**Why fix:** Drag-and-drop is the expected workflow for file-based desktop apps.

---

#### UX-5: LLM configuration is complex
**Severity:** Low (affects power users only)
**Location:** `gui/main.py` (LLM panel)

User must manually configure: URL, API key, model, context window. Many steps
for a feature that should "just work" with a local Open WebUI instance.

**Options:**
1. **Auto-detect localhost** — On first use, probe `localhost:8000`, `localhost:3000`,
   `localhost:11434` for Open WebUI / Ollama. Pre-fill URL if found. *Recommended.*
2. **Sensible defaults** — Default URL to `http://localhost:8000/api`, context to 4096.
   Reduces manual steps.
3. **Setup wizard** — Step-by-step dialog: "Do you have Open WebUI running?" → probe
   → configure → test.

**Why fix:** Reduces friction for the most common deployment (local Open WebUI).

---

## 5. Code Quality

### 5.1 Strengths

- Zero TODO/FIXME/HACK comments — clean codebase
- ruff + mypy strict — enforced code quality
- Frozen dataclasses — immutable data models
- Consistent error handling with `voxfusion.exceptions` hierarchy
- Structured logging via structlog
- Pre-commit hooks configured (ruff, mypy, whitespace, YAML, merge conflicts)

### 5.2 Issues

#### CODE-1: Broad exception catches
**Severity:** Low (masks real errors)
**Location:** Multiple files

Despite CLAUDE.md guideline "never catch bare Exception", the codebase has many
`except Exception` blocks. Examples:
- `gigaam_engine.py:252` — model loading catch-all
- `gigaam_engine.py:86` — GPU probe
- GUI workers — everywhere

**Options:**
1. **Narrow where safe** — Model loading: catch `ImportError | ConnectionError |
   RuntimeError | OSError`. GPU probe: catch `RuntimeError`. *Recommended.*
2. **Keep for GUI workers** — GUI must never crash; broad catches are acceptable
   in the UI layer as long as they log the exception.
3. **Add `# noqa` or `# type: ignore` comments** — Make the intentionality explicit.

**Why fix:** Broad catches can mask programming errors (AttributeError, TypeError)
that should be caught during development.

---

#### CODE-2: Weak typing (`object` as parameter/attribute type)
**Severity:** Low (developer ergonomics)
**Location:** `cli/transcribe_cmd.py:175` (`config: object`), `gigaam_engine.py:186`
(`self._model: object | None`)

Using `object` loses all type information — no autocomplete, no type checking on
method calls.

**Options:**
1. **Use concrete types** — `config: PipelineConfig`, define a `GigaAMModel` protocol
   with `.transcribe(path: str) -> str`. *Recommended.*
2. **Use `Any`** — At least signals "I know this is untyped" rather than "I forgot".
   Marginal improvement.

**Why fix:** Type information is the primary documentation for Python code. `object`
communicates nothing.

---

#### CODE-3: No type stub for GigaAM model
**Severity:** Low (developer ergonomics)
**Location:** `asr/gigaam_engine.py`

The HuggingFace `AutoModel.from_pretrained()` returns a dynamic type. All method
calls on `self._model` use `# type: ignore[attr-defined]`.

**Options:**
1. **Define a Protocol** — `class GigaAMModelProtocol(Protocol): def transcribe(self,
   path: str) -> str: ...` and cast the loaded model. *Recommended.*
2. **Accept it** — Dynamic HF models are inherently untyped. The `# type: ignore` is
   the pragmatic choice for trust_remote_code models.

**Why fix:** A protocol makes the expected interface explicit and catches API changes
at type-check time.

---

## 6. Testing

### 6.1 Strengths

- 85 test files covering unit (79), integration (4), hardware (2)
- Comprehensive conftest.py fixtures for audio, transcription, diarization, translation
- Good mocking of heavy dependencies (PyTorch, pyannote, HuggingFace)
- Hardware tests are opt-in with `--run-hardware` flag
- ~20 GUI-specific tests (rare for Tkinter apps)
- 70% minimum coverage threshold configured

### 6.2 Issues

#### TEST-1: No CI/CD
**Severity:** High (quality assurance gap)
**Location:** (missing `.github/workflows/`)

Tests only run locally. No automated verification on push/PR.

**Options:**
1. **GitHub Actions** — Simple workflow: lint (ruff) + type check (mypy) + unit tests
   (pytest tests/unit/) on push to main and PRs. *Recommended.* Example below.
2. **Pre-push hook** — `pre-commit` hook that runs `pytest tests/unit/ -x -q`.
   Catches issues before push but doesn't provide shared visibility.
3. **Both** — CI for shared verification, pre-push for fast local feedback.

Minimal GitHub Actions workflow:
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]"
      - run: ruff check src/ tests/
      - run: mypy src/
      - run: pytest tests/unit/ -x -q --tb=short
```

**Why fix:** CI is the single highest-impact improvement. Catches regressions
before they reach production.

---

#### TEST-2: No coverage reporting in CI
**Severity:** Low-Medium
**Location:** `pyproject.toml` (coverage config)

`fail_under = 70` is configured but never checked automatically.

**Options:**
1. **Add `--cov --cov-fail-under=70` to CI** — Enforces the threshold on every push.
   *Recommended.*
2. **Codecov integration** — Visual coverage reports on PRs. Nice but adds complexity.

**Why fix:** A coverage threshold that's never enforced is just a comment.

---

#### TEST-3: Some backends lack dedicated unit tests
**Severity:** Low
**Location:** `asr/faster_whisper.py`, `asr/breeze_engine.py`, `asr/parakeet_engine.py`,
`asr/openvino_engine.py`

These engines are tested indirectly through integration tests and pipeline tests,
but don't have dedicated unit tests with mocked dependencies.

**Options:**
1. **Add mocked unit tests** — Mock the underlying model and test the engine's
   chunking, error handling, and normalization logic. *Recommended for faster_whisper
   at minimum* (it's the primary backend).
2. **Accept integration-only coverage** — These backends have different deps on each
   platform. Integration tests are more realistic.

**Why fix:** Unit tests run fast and don't need models. They catch logic errors
in chunking, normalization, and error handling.

---

## 7. Security

### 7.1 Strengths

- Optional output encryption
- HF token via env vars (not in config files)
- `auto_delete_temp_files` for cleanup
- No telemetry by default
- File permission checks on Unix

### 7.2 Issues

#### SEC-1: `trust_remote_code=True` undocumented risk
**Severity:** Medium (security awareness)
**Location:** `asr/gigaam_engine.py:224`

GigaAM models use `trust_remote_code=True` which executes arbitrary Python from
the HuggingFace model repo. This is required but should be documented.

**Options:**
1. **Document in README and ARCHITECTURE.md** — Explain what `trust_remote_code`
   means, why it's needed for GigaAM, and what the risks are. *Recommended.*
2. **Add a runtime warning** — First-time warning in logs when `trust_remote_code`
   is used.
3. **Pin model revision hash** — Use specific commit hashes instead of branch names
   for downloaded models. Prevents supply-chain attacks on the HF repo.

**Why fix:** Users should understand they're running code from a third party.
Transparency builds trust.

---

## 8. Documentation

### 8.1 Strengths

- ARCHITECTURE.md is thorough (72 KB) with ADRs
- Russian-language guides (QUICK_START_RU.md, STREAMING_GUIDE_RU.md)
- BINARY_BUILD.md for packaging
- REQUIREMENTS_TRACEABILITY.md for compliance

### 8.2 Issues

#### DOC-1: ARCHITECTURE.md claims streaming-first
**Severity:** Low (misleading)
**Same as ARCH-1.** Update to reflect batch-first reality.

#### DOC-2: No CHANGELOG
**Severity:** Low
No changelog file tracking releases and changes.

**Options:**
1. **Add CHANGELOG.md** — Keep it manually updated with each release. *Recommended.*
2. **Generate from git** — Use `git-cliff` or similar to auto-generate from
   conventional commits.

---

## 9. Priority Matrix

| ID | Issue | Severity | Effort | Impact | Priority |
|----|-------|----------|--------|--------|----------|
| TEST-1 | No CI/CD | High | Low | High | **P0** |
| ARCH-2 | Duplicate revision mappings | Low | Low | Medium | **P1** |
| UX-2 | Error color-only indication | Low-Med | Minimal | Medium | **P1** |
| UX-3 | No cancel button | Medium | Low | High | **P1** |
| CODE-2 | Weak typing (object) | Low | Low | Medium | **P1** |
| ASR-3 | Crude token estimation | Low | Minimal | Medium | **P1** |
| SEC-1 | trust_remote_code undocumented | Medium | Minimal | Medium | **P1** |
| ARCH-3 | GUI God Object | Medium | Medium | High | **P2** |
| ARCH-1 | Streaming not implemented | Medium | Low-Med | Medium | **P2** |
| UX-1 | Tkinter limitations | Medium | Medium | High | **P2** |
| ASR-2 | No retry for downloads | Medium | Low | Medium | **P2** |
| DIAR-1 | ML diarization setup barrier | Medium | Medium | Medium | **P2** |
| UX-4 | No drag-and-drop | Low-Med | Low | Medium | **P3** |
| UX-5 | LLM config complexity | Low | Low | Low | **P3** |
| ARCH-4 | No service layer | Low-Med | Medium | Medium | **P3** |
| ARCH-5 | Megatron shim fragility | Low | Minimal | Low | **P3** |
| CODE-1 | Broad exception catches | Low | Low | Low | **P3** |
| CODE-3 | No GigaAM type stub | Low | Low | Low | **P3** |
| ASR-1 | Temp file chunking | Low | Low | Low | **P3** |
| TEST-2 | No coverage in CI | Low-Med | Minimal | Low | **P3** |
| TEST-3 | Backend unit tests missing | Low | Medium | Low | **P3** |
| DOC-1 | Architecture doc mismatch | Low | Minimal | Low | **P3** |
| DOC-2 | No CHANGELOG | Low | Low | Low | **P3** |

---

## 10. Overall Assessment

| Area | Score | Notes |
|------|-------|-------|
| Architecture | 7/10 | Modular, protocol-based; streaming gap, GUI monolith |
| ASR | 9/10 | 5 backends, catalog, auto-detect, quality presets |
| Diarization | 8/10 | 4 strategies, chunked ML, alignment |
| Translation | 7/10 | 4 backends, registry, cache; not all fully implemented |
| UI/UX | 6/10 | Feature-complete but Tkinter-limited; no cancel, no DnD |
| Testing | 8/10 | 85 tests, good fixtures; no CI/CD is the critical gap |
| Code Quality | 8/10 | Clean, strict mypy, structlog; minor typing gaps |
| Documentation | 8/10 | Thorough ARCHITECTURE.md, guides, ADRs |
| Security | 7/10 | Adequate for offline tool; trust_remote_code needs docs |
| **Overall** | **7.5/10** | Solid foundation, ready for production use with targeted fixes |
