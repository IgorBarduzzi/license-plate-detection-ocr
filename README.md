# 🚗 License Plate Detection & OCR Preprocessing Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow%20Lite-Edge%20AI-orange.svg)](https://www.tensorflow.org/lite)
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-red.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end computer vision pipeline for **Brazilian Vehicle License Plate Recognition (ALPR/LPR)**. The repository includes:
1. **Model Training Pipeline:** Efficient object detection training using `tflite-model-maker` (EfficientDet-Lite0) with Pascal VOC XML annotations.
2. **Inference & Bounding Box Extraction:** Deep learning inference with margin-adjusted dynamic cropping.
3. **Advanced Image Preprocessing for OCR:** OpenCV pipeline designed to maximize text contrast on small, noisy, or distorted plates using CLAHE, Bilateral filtering, and adaptive Gaussian thresholding.

---

## 🏗️ Pipeline Architecture

```mermaid
graph LR
    Input[📸 Raw Vehicle Image] --> Model[Object Detector: YOLO / EfficientDet]
    Model --> Crop[Margin-Expanded Crop]
    Crop --> Bicubic[Cubic Super-Resolution Scale]
    Bicubic --> Filter[Bilateral Filter: Edge-Preserving Denoise]
    Filter --> CLAHE[CLAHE: Local Contrast Equalization]
    CLAHE --> Sharp[Unsharp Mask: Edge Enhancement]
    Sharp --> Binary[Adaptive Gaussian Binarization]
    Binary --> OCR([🔤 Clean Binary Image for OCR Engine])
```

---

## 🌟 Key Features

- **🎯 Real-Time Edge Detection:** Designed for deployment on low-power devices and edge servers using lightweight EfficientDet-Lite and YOLO architectures.
- **📐 Margin-Aware Cropping:** Automatically expands detected bounding boxes by a configurable margin percentage to prevent character cut-offs caused by tight predictions.
- **✨ OCR Preprocessing Engine:**
  - **Super-Resolution Upscaling:** Resizes small or distant plates to improve character definition.
  - **Bilateral Filtering:** Smooths noise while strictly preserving sharp text edges.
  - **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Corrects uneven lighting and harsh shadows across the plate.
  - **Adaptive Gaussian Thresholding:** Robust binarization that adapts to varying ambient brightness levels.

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/IrguCap/license-plate-detection-ocr.git
cd license-plate-detection-ocr
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Running Inference & OCR Preprocessing
To detect plates and generate preprocessed crops from an image:
```bash
python inference_ocr.py --weights path/to/model.pt --source test_image.jpg --out-dir output/
```

### 3. Training a Custom TFLite Model
To train with Pascal VOC format datasets (Images + XML annotations):
```bash
python train_plate_inplace.py
```

---

## 📁 Repository Structure

```
├── inference_ocr.py         # YOLO inference, crop with margin & OpenCV adaptive filters
├── train_plate_inplace.py   # Pascal VOC dataloader & TFLite Model Maker training pipeline
├── requirements.txt         # Project dependencies
├── .gitignore               # Excludes raw datasets and large model weights
└── README.md
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
