"""
Utility helpers to load the shared analysis configuration and compute
its content hash for provenance tracking.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")


def load_config(path: Path | None = None) -> dict[str, Any]:
    """
    Load the YAML configuration used across analysis scripts.

    Parameters
    ----------
    path:
        Optional override path for the config file. Defaults to
        analysis/config.yaml located alongside this module.
    """
    config_path = path or CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {config_path} must be a mapping.")
    return data


def config_hash(config: dict[str, Any]) -> str:
    """
    Compute a stable SHA256 hash for the provided configuration mapping.

    The mapping is normalised via JSON dumping with sorted keys to ensure
    reproducibility regardless of key ordering.
    """
    normalized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_config_with_hash(path: Path | None = None) -> tuple[dict[str, Any], str]:
    """
    Convenience helper returning both the parsed configuration and its hash.
    """
    config = load_config(path)
    return config, config_hash(config)

