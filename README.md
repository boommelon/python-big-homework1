# 手势检测（YOLO）

## 准备
1. 安装依赖：`pip install -r requirements.txt`
2. 放置数据集到 `data/hand_gesture/`，包含 `images/`、`labels/`、`data.yaml`（YOLO 标注）。
3. 可运行 `scripts/visualize_samples.py` 抽样查看标注；`scripts/stats_labels.py` 统计类别数量。

## 训练
- 直接用 ultralytics CLI：
  ```
  yolo detect train model=yolov8n.pt data=./data/hand_gesture/data.yaml imgsz=640 epochs=60 batch=16
  ```
- 或运行封装脚本：
  ```
  python src/train.py
  ```

## 推理
```
python src/infer.py
```
或 CLI：
```
yolo detect predict model=./runs/detect/hand_gesture/weights/best.pt source=./demo_images conf=0.25
```

## 演示界面
```
streamlit run app.py
```

