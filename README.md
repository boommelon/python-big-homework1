# 手势检测项目（YOLOv8 + 自制数据集）

一个基于 **Ultralytics YOLOv8** 的手势检测项目，完成了从数据集构建、模型训练到命令行推理与 Web 图形界面展示的完整流程，能够区分两类手势：`Grab`（握拳）与 `Release`（张开手掌）。

---

## 一、项目概述与方法

- **方法**：采用 Ultralytics YOLOv8 目标检测模型，在公开手势数据集基础上进行迁移学习与微调。  
- **目标**：实现 Grab / Release 手势的实时检测，用于课堂演示与简单人机交互。  
- **模型结构简述**：Backbone + FPN/PAFPN 颈部 + Detect Head，输入尺寸默认 640，轻量化 `yolov8n.pt` 作为预训练权重。  
- **训练策略**：
  - 优化器：SGD（由 Ultralytics 自动选择）  
  - 学习率：`lr0=0.01`，`weight_decay=0.0005`  
  - Epoch：60  
  - 数据增强：随机翻转、尺度缩放、颜色扰动等（由 YOLO 内置）

---

## 二、项目结构（Software 结构设计）

```text
python大作业1/
├── data/hand_gesture/          # YOLO 数据集（data.yaml, images, labels）
├── runs/                       # 训练输出（hand_gesture*, detect/predict 等）
├── src/
│   ├── train.py                # 训练入口，封装 YOLO 训练
│   ├── infer.py                # 推理脚本（批量图片 / 摄像头）
│   └── utils.py                # 路径与配置工具
├── scripts/
│   ├── visualize_samples.py    # 抽样可视化标注结果
│   └── stats_labels.py         # 统计各类别样本数量
├── app.py                      # Streamlit Web GUI（图片上传 + 摄像头）
├── demo_images/                # 演示用样例图（含自采集图片）
├── requirements.txt            # 依赖列表
└── README.md                   # 使用说明（本文件）
```

---

## 三、运行环境与依赖

- **Python**：3.11（已在 3.11.9 上验证）
- **操作系统**：Windows 10/11 64 位
- **GPU（推荐）**：NVIDIA 显卡 + 对应 CUDA 驱动  
  - 本项目已在 **RTX 3060 Laptop GPU + torch 2.3.0+cu121** 上测试通过。

---

## 安装步骤

1. **克隆 / 下载代码**
   ```bash
   git clone <your-repo-url>
   cd python大作业1
   ```

2. **安装基础依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **安装 GPU 版 PyTorch（推荐）**
   - 根据自己显卡和 CUDA 版本，从 PyTorch 官网获取安装命令，或使用已下载好的 whl，例如：
   ```bash
   pip install D:\Desktop\pkgs\torch-2.3.0+cu121-cp311-cp311-win_amd64.whl
   pip install D:\Desktop\pkgs\torchvision-0.18.0+cu121-cp311-cp311-win_amd64.whl
   pip install D:\Desktop\pkgs\torchaudio-2.3.0+cu121-cp311-cp311-win_amd64.whl
   ```

4. **确保 NumPy 版本兼容**
   ```bash
   pip uninstall -y numpy
   pip install "numpy<2.0,>=1.23.5"
   ```

---

## 数据集准备

- 将标注好的手势检测数据集放到：
  - `data/hand_gesture/`
    - `images/`  训练与验证图片
    - `labels/`  YOLO txt 标注
    - `data.yaml` 数据集配置文件

- 可选辅助脚本：
  - `scripts/visualize_samples.py`：随机可视化标注，检查是否正确。
  - `scripts/stats_labels.py`：统计各类别样本数量。

**当前数据集情况（示例）**：

- 公开数据集：Grab / Release 两类手势，共约 800 张图（train/valid/test 划分）。  
- 自采集数据：在宿舍场景下额外拍摄若干张自己的手势图片，并使用 makesense.ai 标注后加入 `train` 集，增强模型在真实使用场景下的表现。

---

## 四、训练（模型设计与调参）

### 方式一：直接使用 YOLO CLI

```bash
yolo detect train model=yolov8n.pt data=./data/hand_gesture/data.yaml imgsz=640 epochs=60 batch=16
```

### 方式二：使用封装好的训练脚本（推荐）

```bash
python src/train.py
```

- `src/train.py` 中默认：
  ```python
  train(data_yaml, device="cuda")
  ```
  会自动在 **GPU（cuda）** 上训练。  
- 如需改回 CPU，只需改成：
  ```python
  train(data_yaml)
  ```

训练完成后，权重会保存在 `runs/hand_gesture*` 目录下（以 Ultralytics 默认命名为准）。

**一次典型训练结果（hand_gesture6）**：

- 验证集总体：`mAP50 ≈ 0.995`，`mAP50-95 ≈ 0.915`，`P ≈ 0.999`，`R ≈ 1.0`  
- 按类别：
  - Grab：`mAP50 ≈ 0.995`，`mAP50-95 ≈ 0.893`  
  - Release：`mAP50 ≈ 0.995`，`mAP50-95 ≈ 0.937`  

说明在当前数据集上，模型能够稳定区分两类手势。

---

## 五、推理 / 预测（命令行接口）

### 使用项目脚本

```bash
python src/infer.py
```

> 默认会加载训练好的 best 权重，并对指定图片 / 目录做检测（根据脚本内部配置）。

### 使用 YOLO CLI

```bash
yolo detect predict model=./runs/hand_gesture/weights/best.pt source=./demo_images conf=0.25
```

---

## 可视化演示界面

使用 Streamlit 启动一个简单的 Web UI（图片上传 + 摄像头实时预览）：

```bash
streamlit run app.py
```

在浏览器中打开提示的本地地址，即可：
- 上传图片并查看检测框与置信度；
- 在“摄像头实时预览”页中启动本机摄像头，实时展示检测结果。

---

## 六、测试与使用建议

- **功能测试**：  
  - 模型能在 `demo_images` 与验证集中正确画出握拳 / 张开手掌的检测框，并给出类别与置信度。  
  - Streamlit 界面支持图片上传检测与摄像头实时预览。  
- **性能测试**：  
  - 在 RTX 3060 Laptop GPU 上，640×480 分辨率下单帧推理约 15ms，可满足实时演示需求。  
- **局限性与改进方向**：  
  - 当摄像头场景与训练数据差异较大（如光照极暗、视角极端、手部严重模糊）时，可能出现误判或不出框。  
  - 可通过继续采集并标注更多真实使用场景数据、增加数据增强、对不同模型尺寸（如 yolov8s）进行对比实验来进一步提升泛化能力。

---

## 常见问题

- **训练时提示 `Numpy is not available`**
  - 按照上文“确保 NumPy 版本兼容”一节，将 NumPy 降到 `<2.0` 再重新训练。

- **训练太慢**
  - 确认命令行显示中 `CUDA` 可用，并且 `device` 为 `cuda`。
  - 适当调大 `batch`（显存允许的情况下），并减小 `epochs` 做调试。 

