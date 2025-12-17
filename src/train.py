"""
训练入口：封装 YOLOv8 训练，读取 data.yaml 与超参。
需先安装 ultralytics，并确保 data/hand_gesture/data.yaml 已就绪。
"""

from pathlib import Path
from typing import Optional

from ultralytics import YOLO


def train(
    data_yaml: Path,
    model: str = "yolov8n.pt",
    imgsz: int = 640,
    epochs: int = 60,
    batch: int = 16,
    workers: int = 4,
    lr0: float = 0.01,
    weight_decay: float = 0.0005,
    project: str = "runs",
    name: str = "hand_gesture",
    device: Optional[str] = None,
):
    """
    运行 YOLO 训练。
    """
    assert data_yaml.exists(), f"data.yaml 不存在: {data_yaml}"
    model_obj = YOLO(model)
    model_obj.train(
        data=str(data_yaml),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        workers=workers,
        lr0=lr0,
        weight_decay=weight_decay,
        project=project,
        name=name,
        device=device,
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    data_yaml = root / "data" / "hand_gesture" / "data.yaml"
    train(data_yaml)

