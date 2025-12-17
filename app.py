"""
Streamlit 简易演示界面：
- Tab1：上传图片检测
- Tab2：页面内摄像头实时预览
运行：streamlit run app.py
"""

import tempfile
from pathlib import Path

import cv2
import streamlit as st
from ultralytics import YOLO


def load_model(weights: Path):
    return YOLO(str(weights))


def run_inference(model, source, conf: float, iou: float):
    return model.predict(source=source, conf=conf, iou=iou, save=False, stream=True)


def main():
    st.title("Hand Gesture Detection (YOLOv8)")
    root = Path(__file__).resolve().parent
    # 使用当前项目训练得到的 best.pt 作为默认权重
    default_weights = root / "runs" / "hand_gesture" / "weights" / "best.pt"

    weights_path = st.text_input("权重路径", str(default_weights))
    conf = st.slider("置信度阈值", 0.1, 0.9, 0.25, 0.05)
    iou = st.slider("NMS IoU 阈值", 0.1, 0.9, 0.45, 0.05)

    if not Path(weights_path).exists():
        st.warning("权重文件不存在，请先训练或放置 best.pt")
        return

    model = load_model(Path(weights_path))

    tab1, tab2 = st.tabs(["图片上传", "摄像头实时预览"])

    # Tab1：图片上传检测
    with tab1:
        uploaded = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded.read())
                tmp_path = Path(tmp.name)
            results = list(run_inference(model, str(tmp_path), conf, iou))
            if results:
                res = results[0]
                st.image(res.plot(), caption="检测结果")

    # Tab2：摄像头实时预览（在页面内显示）
    with tab2:
        st.write("在本地电脑上启用摄像头进行实时手势检测。")
        start = st.button("启动摄像头实时检测")

        if start:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("无法打开摄像头，请检查设备连接。")
            else:
                frame_placeholder = st.empty()
                st.info("正在从摄像头读取画面，大约显示 10 秒。如需停止可关闭页面或重新运行。")

                # 显示有限帧，避免死循环卡住页面
                for _ in range(300):  # 约 10 秒，取决于实际 FPS
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 对单帧做推理并绘制结果
                    results = model.predict(source=frame, conf=conf, iou=iou, verbose=False, device=0)
                    annotated = results[0].plot()  # BGR

                    frame_placeholder.image(annotated, channels="BGR")

                cap.release()


if __name__ == "__main__":
    main()

