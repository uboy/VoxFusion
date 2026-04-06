# Requirements Traceability

## Live GigaAM v1

| Requirement | Description | Implementation | Verification |
|---|---|---|---|
| `LG-1` | Durable session audio spool for live GigaAM | `src/voxfusion/live_gigaam/spool.py`, `src/voxfusion/live_gigaam/session.py` | `tests/unit/test_live_gigaam_spool.py` |
| `LG-2` | Deterministic ordered transcript commit | `src/voxfusion/live_gigaam/commit.py`, `src/voxfusion/live_gigaam/session.py` | `tests/unit/test_live_gigaam_commit.py`, `tests/unit/test_live_gigaam_session.py` |
| `LG-3` | Warm worker-process pool for live utterances | `src/voxfusion/live_gigaam/dispatcher.py`, `src/voxfusion/live_gigaam/worker.py` | `tests/unit/test_live_gigaam_dispatcher.py`, `tests/unit/test_live_gigaam_worker.py` |
| `LG-4` | Selective stop-time finalization from spooled session audio, with optional full second pass | `src/voxfusion/live_gigaam/session.py`, `src/voxfusion/config/models.py` | `tests/unit/test_live_gigaam_session.py`, `tests/unit/test_config.py` |
| `LG-5` | Finalization fallback to draft text on error | `src/voxfusion/live_gigaam/session.py` | `tests/unit/test_live_gigaam_session.py::test_live_gigaam_session_uses_draft_text_when_finalize_fails` |
| `LG-6` | GUI live timestamps anchored to actual capture start | `src/voxfusion/gui/runtime.py` | `tests/unit/test_live_gigaam_gui_runtime.py` |
| `LG-7` | CLI/GUI contract alignment for live GigaAM support and no live translation | `src/voxfusion/asr_catalog.py`, `src/voxfusion/gui/main.py`, `src/voxfusion/cli/capture_cmd.py` | `tests/unit/test_live_gigaam_catalog.py`, `tests/unit/test_capture_cli.py`, `tests/unit/test_live_gigaam_cli.py`, `tests/unit/test_live_gigaam_gui_contract.py` |
| `LG-8` | Controlled draft deferral under overload with final recovery from spool | `src/voxfusion/live_gigaam/session.py`, `src/voxfusion/config/models.py` | `tests/unit/test_live_gigaam_session.py::test_live_gigaam_session_defers_drafts_under_backlog_and_finalizes_all`, `tests/unit/test_config.py`, `tests/unit/test_live_gigaam_dispatcher.py::test_concurrent_dispatch_uses_multiple_workers_under_load` |
