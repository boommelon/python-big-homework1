import random
from pathlib import Path

from PIL import Image, ImageDraw


# 路径配置：以 data.yaml 所在目录为根，根据其中的 train/val/test 路径来找数据
DATA_ROOT = Path("../data/hand_gesture")
OUT_DIR = Path("../demo_images")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_names(yaml_path: Path):
    """从 data.yaml 读取类别 names。"""
    import yaml

    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    return y["names"]


def draw_bbox(img_path: Path, label_path: Path, names):
    """在图像上绘制 YOLO 标签框并保存。"""
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    draw = ImageDraw.Draw(im)
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cid, cx, cy, bw, bh = map(float, parts)
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), names[int(cid)] if names else str(int(cid)), fill="red")
    return im


def get_split_dirs(split: str):
    """根据 data.yaml 中的路径推断 images/labels 目录。"""
    import yaml

    yaml_path = DATA_ROOT / "data.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    key = {"train": "train", "val": "val", "test": "test"}[split]
    img_root = Path(cfg[key])  # 例如 ./train/images
    if not img_root.is_absolute():
        img_root = (yaml_path.parent / img_root).resolve()
    # train/images -> train/labels
    split_dir = img_root.parent  # ./train
    lbl_root = split_dir / "labels"
    return img_root, lbl_root


def sample_split(split: str, k: int, names):
    """随机抽样指定划分，绘制标注并保存到 OUT_DIR。"""
    img_dir, lbl_dir = get_split_dirs(split)
    imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not imgs:
        print(f"[warn] {split} 无图片，跳过")
        return
    sample_imgs = random.sample(imgs, min(k, len(imgs)))
    for img_path in sample_imgs:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            print(f"[warn] 缺少标签：{lbl_path}")
            continue
        im = draw_bbox(img_path, lbl_path, names)
        out_path = OUT_DIR / f"{split}_{img_path.name}"
        im.save(out_path)


def main():
    yaml_path = DATA_ROOT / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError("未找到 data.yaml，请先放置数据集并确认路径。")
    names = load_names(yaml_path)
    sample_split("train", k=8, names=names)
    sample_split("val", k=4, names=names)
    print("done, 请查看 ../demo_images")


if __name__ == "__main__":
    main()

