from collections import Counter
from pathlib import Path


DATA_ROOT = Path("../data/hand_gesture")


def load_yaml():
    import yaml

    yaml_path = DATA_ROOT / "data.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_names(cfg):
    return cfg.get("names", None)


def get_label_dir(cfg, split: str) -> Path:
    """根据 data.yaml 中的路径推断 labels 目录。"""
    key = {"train": "train", "val": "val", "test": "test"}[split]
    img_root = Path(cfg[key])  # 例如 ./train/images
    yaml_path = DATA_ROOT / "data.yaml"
    if not img_root.is_absolute():
        img_root = (yaml_path.parent / img_root).resolve()
    # train/images -> train/labels
    split_dir = img_root.parent  # ./train
    lbl_root = split_dir / "labels"
    return lbl_root


def count_split(cfg, split: str):
    lbl_dir = get_label_dir(cfg, split)
    cnt = Counter()
    if not lbl_dir.exists():
        return cnt
    for txt in lbl_dir.glob("*.txt"):
        with open(txt, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = parts[0]
                if cls_id.isdigit():
                    cnt[int(cls_id)] += 1
    return cnt


def main():
    cfg = load_yaml()
    names = load_names(cfg)
    splits = ["train", "val", "test"]
    total = Counter()
    for sp in splits:
        total += count_split(cfg, sp)

    if not total:
        print("未统计到标签，请确认 labels/*/*.txt 是否存在。")
        return

    print("类别计数：")
    for cid, num in sorted(total.items()):
        name = names[cid] if names and cid < len(names) else str(cid)
        print(f"{cid}\t{name}\t{num}")


if __name__ == "__main__":
    main()

