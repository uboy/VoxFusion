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
import sys
from pathlib import Path

import yaml

from voxfusion.config.models import PipelineConfig
from voxfusion.exceptions import ConfigurationError
from voxfusion.logging import get_logger

log = get_logger(__name__)


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
    layers = [load_defaults()]
    for loader in (load_system_config, load_user_config, load_project_config):
        layer = loader()
        if layer:
            layers.append(layer)
    if overrides:
        layers.append(overrides)

    merged = merge_configs(*layers)

    try:
        return PipelineConfig(**merged)
    except Exception as exc:
        raise ConfigurationError(f"Invalid configuration: {exc}") from exc


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
