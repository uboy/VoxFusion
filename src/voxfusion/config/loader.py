"""Configuration loading with hierarchical resolution.

Resolution order (later overrides earlier):
1. Built-in defaults (defaults.yaml)
2. System config
3. User config (~/.voxfusion/config.yaml)
4. Project config (.voxfusion.yaml in cwd)
5. Environment variables (via Pydantic)
6. Explicit overrides (CLI flags)
"""

import copy
import importlib.resources
import json
import os
import sys
from pathlib import Path

import yaml

from voxfusion.config.models import PipelineConfig
from voxfusion.exceptions import ConfigurationError
from voxfusion.logging import get_logger

log = get_logger(__name__)


def _gui_settings_path() -> Path:
    override = os.environ.get("VOXFUSION_GUI_SETTINGS_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".voxfusion" / "gui_settings.json"


def _load_gui_settings_hf_token() -> str | None:
    target = _gui_settings_path()
    if not target.is_file():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    token = str(data.get("hf_token", "")).strip()
    return token or None


def _apply_gui_settings_token_environment_fallback() -> None:
    """Expose GUI-saved HF token to CLI/runtime code when no stronger source exists."""
    if os.environ.get("VOXFUSION_DIARIZATION__ML__HF_AUTH_TOKEN"):
        return
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    token = _load_gui_settings_hf_token()
    if not token:
        return
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def _apply_environment_compatibility_overrides(merged: dict) -> dict:  # type: ignore[type-arg]
    """Backfill env-driven settings that nested dict merging can shadow.

    Pydantic BaseSettings does not reliably inject nested env values once the
    corresponding nested object is already provided via our merged dict layers.
    Keep this helper narrowly scoped to documented env knobs that must behave
    consistently in both config preflight and runtime code paths.
    """
    token = os.environ.get("VOXFUSION_DIARIZATION__ML__HF_AUTH_TOKEN")
    if token:
        merged.setdefault("diarization", {})
        merged["diarization"].setdefault("ml", {})
        merged["diarization"]["ml"]["hf_auth_token"] = token
    return merged


def _resolve_diarization_token_source(config: PipelineConfig) -> str | None:
    if config.diarization.ml.hf_auth_token:
        return "config-or-env:VOXFUSION"
    for env_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(env_name):
            return f"env:{env_name}"
    return None


def _deep_merge(base: dict, override: dict) -> dict:  # type: ignore[type-arg]
    """Recursively merge *override* into *base* (returns a new dict)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml(path: Path) -> dict | None:  # type: ignore[type-arg]
    """Load a YAML file, returning ``None`` if the file does not exist."""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else None
    except (yaml.YAMLError, OSError) as exc:
        log.warning("failed_to_load_config", path=str(path), error=str(exc))
        return None


# -- Public helpers -----------------------------------------------------------


def load_defaults() -> dict:  # type: ignore[type-arg]
    """Load the bundled ``defaults.yaml``."""
    ref = importlib.resources.files("voxfusion.config").joinpath("defaults.yaml")
    text = ref.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ConfigurationError("Bundled defaults.yaml is invalid")
    return data


def _system_config_path() -> Path:
    if sys.platform == "win32":
        base = Path(r"C:\ProgramData")
    else:
        base = Path("/etc")
    return base / "voxfusion" / "config.yaml"


def load_system_config() -> dict | None:  # type: ignore[type-arg]
    """Load system-wide configuration."""
    return _load_yaml(_system_config_path())


def load_user_config() -> dict | None:  # type: ignore[type-arg]
    """Load user configuration from ``~/.voxfusion/config.yaml``."""
    return _load_yaml(Path.home() / ".voxfusion" / "config.yaml")


def load_project_config(cwd: Path | None = None) -> dict | None:  # type: ignore[type-arg]
    """Load project-level configuration from ``.voxfusion.yaml`` in *cwd*."""
    cwd = cwd or Path.cwd()
    return _load_yaml(cwd / ".voxfusion.yaml")


def merge_configs(*configs: dict) -> dict:  # type: ignore[type-arg]
    """Deep-merge multiple config dicts (later overrides earlier)."""
    result: dict = {}  # type: ignore[type-arg]
    for cfg in configs:
        result = _deep_merge(result, cfg)
    return result


def load_config(overrides: dict | None = None) -> PipelineConfig:  # type: ignore[type-arg]
    """Load the fully-resolved configuration.

    Args:
        overrides: Optional dict of CLI-level overrides (highest priority).

    Returns:
        Validated ``PipelineConfig`` instance.

    Raises:
        ConfigurationError: If the merged config fails validation.
    """
    _apply_gui_settings_token_environment_fallback()
    layers = [load_defaults()]
    layer_names = ["defaults"]
    for name, loader in (
        ("system", load_system_config),
        ("user", load_user_config),
        ("project", load_project_config),
    ):
        layer = loader()
        if layer:
            layers.append(layer)
            layer_names.append(name)
    if overrides:
        layers.append(overrides)
        layer_names.append("overrides")

    merged = _apply_environment_compatibility_overrides(merge_configs(*layers))

    try:
        config = PipelineConfig(**merged)
    except Exception as exc:
        raise ConfigurationError(f"Invalid configuration: {exc}") from exc
    token_source = _resolve_diarization_token_source(config)
    log.info(
        "config.loaded",
        layers=layer_names,
        overrides_applied=bool(overrides),
        asr_model=config.asr.model_size,
        asr_engine=config.asr.engine,
        language=config.asr.language,
        diarization_strategy=config.diarization.strategy,
        diarization_token_present=token_source is not None,
        diarization_token_source=token_source,
        min_speakers=config.diarization.ml.min_speakers,
        max_speakers=config.diarization.ml.max_speakers,
        output_format=config.output.format,
    )
    return config


def save_user_config(config: PipelineConfig) -> None:
    """Persist *config* to ``~/.voxfusion/config.yaml``."""
    path = Path.home() / ".voxfusion" / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.dump(config.model_dump(), fh, default_flow_style=False, sort_keys=False)


def get_config_path(level: str) -> Path:
    """Return the filesystem path for the given config level.

    Args:
        level: One of ``"system"``, ``"user"``, or ``"project"``.
    """
    match level:
        case "system":
            return _system_config_path()
        case "user":
            return Path.home() / ".voxfusion" / "config.yaml"
        case "project":
            return Path.cwd() / ".voxfusion.yaml"
        case _:
            raise ConfigurationError(f"Unknown config level: {level!r}")


def show_config(config: PipelineConfig, fmt: str = "yaml") -> str:
    """Serialize *config* to a human-readable string.

    Args:
        fmt: ``"yaml"`` or ``"json"``.
    """
    if fmt == "json":
        return config.model_dump_json(indent=2)
    return yaml.dump(config.model_dump(), default_flow_style=False, sort_keys=False)
