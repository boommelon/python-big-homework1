"""
通用工具：读取配置、路径管理、简单可视化占位。
"""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    assert path.exists(), f"YAML 不存在: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def data_yaml_path() -> Path:
    return project_root() / "data" / "hand_gesture" / "data.yaml"

