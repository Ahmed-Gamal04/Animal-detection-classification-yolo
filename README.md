# Animal Detection & Classification using YOLO

A visual computing project that detects and classifies animals using multiple YOLO models with an interactive Streamlit web app.

## Features
- Animal object detection with bounding boxes
- Animal image classification
- Confidence threshold control
- Multiple model comparison
- Clean web interface using Streamlit

## Models Used
- YOLOv8 Detection
- YOLOv5 Detection
- YOLO Classification Model

## Dataset Work
- Raw dataset cleaning and class merging
- Image preprocessing (resize, blur, grayscale)
- Data augmentation (flip, rotate, crop, brightness)

## Results
| Model | Task | Score |
|------|------|------|
| YOLOv8 | Detection | mAP50 78.2% |
| YOLOv5 | Detection | mAP50 78.9% |
| YOLO11 | Classification | Accuracy 96.3% |

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
