
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="Skin Lesion AI", page_icon="🔬",
                   layout="wide")

ROOT = Path(__file__).parent

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        ROOT / "skin_disease_model.keras", compile=False
    )

model = load_model()
classes = np.load(ROOT / "class_names.npy", allow_pickle=True).tolist()

st.title("Skin Lesion Classification Using Deep Learning")
st.write("EfficientNetB0 | 7-class skin lesion classification")
st.warning(
    "Educational project only. This model is not a medical diagnosis "
    "and must not replace assessment by a qualified dermatologist."
)

st.subheader("Upload a skin lesion image")
uploaded = st.file_uploader(
    "Choose a JPG, JPEG or PNG image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    left, right = st.columns(2)

    with left:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with right:
        size = model.input_shape[1:3]
        image_resized = image.resize((size[1], size[0]))
        x = np.array(image_resized, dtype=np.float32)
        x = np.expand_dims(x, axis=0)

        if st.button("Classify Image", type="primary"):
            with st.spinner("Analysing image..."):
                scores = model.predict(x, verbose=0)[0]
                idx = int(np.argmax(scores))

            st.subheader("Prediction result")
            st.success(f"Predicted class: {classes[idx]}")
            st.metric("Model score", f"{float(scores[idx]) * 100:.2f}%")
            st.caption(
                "This score is the model's output, not a confirmed "
                "probability that the lesion has this disease."
            )

            st.subheader("Scores for all classes")
            st.bar_chart({
                "Model score (%)": {
                    str(classes[i]): float(scores[i] * 100)
                    for i in range(len(classes))
                }
            })

st.divider()
st.subheader("Model evaluation outputs")

for filename, title in [
    ("accuracy_graph.png", "Training accuracy"),
    ("loss_graph.png", "Training loss"),
    ("confusion_matrix.png", "Confusion matrix"),
    ("gradcam_example.png", "Grad-CAM example")
]:
    path = ROOT / filename
    if path.exists():
        st.write(title)
        st.image(str(path), use_container_width=True)

report = ROOT / "classification_report.csv"
if report.exists():
    import pandas as pd
    st.subheader("Classification report")
    st.dataframe(pd.read_csv(report), use_container_width=True)

metrics = ROOT / "metrics.json"
if metrics.exists():
    import json
    st.subheader("Evaluation metrics")
    st.json(json.loads(metrics.read_text()))

st.caption("Project demonstration — not intended for clinical use.")
