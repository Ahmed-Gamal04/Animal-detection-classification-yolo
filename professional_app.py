import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from pathlib import Path

st.set_page_config(page_title='Animal Detection & Classification', page_icon='🦁', layout='wide')

# ---------- Paths ----------
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / 'models'
MODEL_PATHS = {
    'YOLOv8 (Detection)': MODEL_DIR / 'yolov8_best.pt',
    'YOLOv5 (Detection)': MODEL_DIR / 'yolov5_best.pt',
    'YOLO11 (Classification)': MODEL_DIR / 'yolo11_best.pt'
}

# ---------- Cache model loading ----------
@st.cache_resource
def load_model(path):
    return YOLO(str(path))

# ---------- UI ----------
st.title('🦁 Animal Detection & Classification')
st.caption('YOLO-based computer vision project with Streamlit interface')

with st.sidebar:
    st.header('Settings')
    model_choice = st.selectbox('Choose model', list(MODEL_PATHS.keys()))
    conf = st.slider('Confidence threshold', 0.10, 0.90, 0.25, 0.05)
    st.markdown('---')
    st.info('Upload an image to run animal detection or classification.')

uploaded = st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])

# ---------- Load selected model ----------
model_path = MODEL_PATHS[model_choice]
if not model_path.exists():
    st.error(f'Model file not found: {model_path.name}')
    st.stop()

model = load_model(model_path)

# ---------- Prediction ----------
if uploaded:
    img = Image.open(uploaded).convert('RGB')
    img_np = np.array(img)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption='Original Image', use_container_width=True)

    with st.spinner('Running model...'):
        result = model.predict(img_np, conf=conf, save=False)[0]

    with col2:
        if 'Classification' in model_choice:
            label = result.names[result.probs.top1]
            score = float(result.probs.top1conf)
            st.image(img, caption='Classification Result', use_container_width=True)
            st.success(f'Predicted: {label}')
            st.progress(score)
            st.write(f'Confidence: {score:.1%}')
        else:
            plotted = result.plot()[:, :, ::-1]
            st.image(plotted, caption='Detection Result', use_container_width=True)
            st.success(f'Detected objects: {len(result.boxes)}')

    st.markdown('---')
    st.subheader('Project Info')
    st.write('This application compares multiple YOLO models for animal detection and classification.')
else:
    st.info('Upload an image to begin.')
