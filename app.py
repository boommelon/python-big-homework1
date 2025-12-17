"""
Streamlit 简易演示界面：上传图片或摄像头实时检测。
运行：streamlit run app.py
"""

import tempfile
from pathlib import Path

import streamlit as st
from ultralytics import YOLO


def load_model(weights: Path):
    return YOLO(str(weights))


def run_inference(model, source, conf: float, iou: float):
    return model.predict(source=source, conf=conf, iou=iou, save=False, stream=True)


def main():
    st.title("Hand Gesture Detection (YOLOv8)")
    root = Path(__file__).resolve().parent
    default_weights = root / "runs" / "detect" / "hand_gesture" / "weights" / "best.pt"

    weights_path = st.text_input("权重路径", str(default_weights))
    conf = st.slider("置信度阈值", 0.1, 0.9, 0.25, 0.05)
    iou = st.slider("NMS IoU 阈值", 0.1, 0.9, 0.45, 0.05)

    if not Path(weights_path).exists():
        st.warning("权重文件不存在，请先训练或放置 best.pt")
        return

    model = load_model(Path(weights_path))

    tab1, tab2 = st.tabs(["图片上传", "摄像头演示"])

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

    with tab2:
        st.info("在本地运行时可启用摄像头：修改 source=0 的推理调用。")
        st.code("yolo detect predict model=best.pt source=0 conf=0.25 show=True")


if __name__ == "__main__":
    main()

