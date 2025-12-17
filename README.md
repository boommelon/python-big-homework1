# 手势检测项目（YOLOv8 + 自制数据集）

一个基于 **Ultralytics YOLOv8** 的手势检测项目，支持自定义数据集训练、命令行推理和可视化演示界面。

---

## 环境要求

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

---

## 训练

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

---

## 推理 / 预测

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

## 常见问题

- **训练时提示 `Numpy is not available`**
  - 按照上文“确保 NumPy 版本兼容”一节，将 NumPy 降到 `<2.0` 再重新训练。

- **训练太慢**
  - 确认命令行显示中 `CUDA` 可用，并且 `device` 为 `cuda`。
  - 适当调大 `batch`（显存允许的情况下），并减小 `epochs` 做调试。 

