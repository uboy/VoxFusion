#!/bin/sh
# Локальное зеркало CI (.github/workflows/ci.yml: lint ruff+mypy, pytest tests/unit).
# Вызывается автоматически pre-push хуком bm1; можно запускать вручную перед пушем.
# Скоуп шире CI (server/ CI не линтует - здесь линтуется), падение здесь = падение CI.
set -e
cd "$(dirname "$0")/.."
PY=".venv/bin/python"; [ -x "$PY" ] || PY=python3
RUFF=""
[ -x ".venv/bin/ruff" ] && RUFF=".venv/bin/ruff"
[ -z "$RUFF" ] && RUFF="$(command -v ruff || true)"
[ -z "$RUFF" ] && RUFF="$PY -m ruff"

echo "[ci-mirror] ruff check src/ tests/ server/"
$RUFF check src/ tests/ server/
echo "[ci-mirror] ruff format --check src/ tests/ server/"
$RUFF format --check src/ tests/ server/
echo "[ci-mirror] mypy src/ server/"
$PY -m mypy src/ server/
echo "[ci-mirror] pytest tests/unit/"
if ! $PY -m pytest tests/unit/ -q --tb=short > /tmp/ci-mirror-pytest.txt 2>&1; then
  # известный артефакт: segfault при shutdown ПОСЛЕ всех passed (обход в CI такой же)
  if grep -q "passed" /tmp/ci-mirror-pytest.txt && ! grep -q "FAILED" /tmp/ci-mirror-pytest.txt; then
    echo "[ci-mirror] pytest passed (shutdown segfault ignored, как в CI)"
  else
    tail -20 /tmp/ci-mirror-pytest.txt
    exit 1
  fi
else
  tail -2 /tmp/ci-mirror-pytest.txt
fi
echo "[ci-mirror] OK - можно пушить"
