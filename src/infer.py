"""
推理入口：加载最佳权重，支持图片/目录/摄像头推理。
"""

from pathlib import Path
from typing import Optional, Union

from ultralytics import YOLO


def predict(
    weights: Path,
    source: Union[str, int, Path],
    conf: float = 0.25,
    iou: float = 0.45,
    save: bool = True,
    show: bool = False,
    project: str = "runs",
    name: str = "predict",
    device: Optional[str] = None,
):
    assert weights.exists(), f"权重文件不存在: {weights}"
    model = YOLO(str(weights))
    model.predict(
        source=str(source),
        conf=conf,
        iou=iou,
        save=save,
        show=show,
        project=project,
        name=name,
        device=device,
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    # 使用训练脚本生成的 best.pt 作为默认权重
    weights = root / "runs" / "hand_gesture" / "weights" / "best.pt"

    # 示例1：对 demo_images 文件夹内的图片批量推理并保存结果
    predict(weights, source=root / "demo_images", save=True, show=False, project="runs", name="infer_demo")

    # 示例2：摄像头实时推理（本地运行时可解开注释）
    # predict(weights, source=0, save=False, show=True, device="cuda")

