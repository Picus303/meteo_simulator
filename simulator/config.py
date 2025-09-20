from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass, field

CONFIG_GROUPS = ("constants", "numerics", "physics", "outputs")


@dataclass(slots=True)
class RunConfig:
    """
    Top-level configuration holder.
    (for now, stays generic, but could be specialized later)
    """

    constants: Dict[str, Any] = field(default_factory=dict)
    numerics: Dict[str, Any] = field(default_factory=dict)
    physics: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def load(path: str | Path) -> "RunConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")

        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            # Ensure sections exist
            for k in CONFIG_GROUPS:
                data.setdefault(k, {})

            print(data)
            return RunConfig(**data)