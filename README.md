# Document-classification-model
Document-classification-model
## Overview
This project implements a **Machine Learning pipeline** to classify document images into three categories:

1. **Letter**
2. **Resume**
3. **Scientific**
Manually classify some data from the dataset as letters , resumes and scintific. After that we split the data into test train and got the accuracy of test train.

The pipeline uses **OCR (EasyOCR)** to extract text from images and **TF-IDF vectorization + ML model** for classification. It supports both **batch folder processing** and **single-image inference**.

---

## Features
- OCR-based text extraction from images (`.jpg`, `.jpeg`, `.png`).
- TF-IDF vectorization to convert text into numerical features.
- Classification using **SVM / Logistic Regression / Random Forest**.
- Batch segregation: automatically moves images into class-specific folders.
- Single-image inference: predict the class of a single document.
- Basic text cleaning (removing special characters, lowercasing).

---

## Installation
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Ensure you have **EasyOCR** and **PyTorch** installed for OCR processing.

---

## Usage

### Training
1. Organize your training dataset in `dataset/` with subfolders `letter/`, `resume/`, `scientific/`.
2. Run the training script:
```bash
python train.py
```
3. This will:
   - Extract text from images using EasyOCR
   - Vectorize the text using TF-IDF
   - Train the ML model (SVM / Logistic Regression / Random Forest)
   - Save the trained model and vectorizer in `Source_code/`

### Batch Inference & Segregation
1. Place all images you want to classify in `inference_data/`.
2. Run segregation script:
```bash
python segregation.py
```
3. The script will:
   - Extract text from each image
   - Predict the class using the trained model
   - Move images into `Segregated_Data/<class>/` folders

### Single Image Inference
You can modify `segregation.py` to provide a single image path:
```python
image_path = 'inference_data/test_image.jpg'
prediction = predict_label(image_path)
print(f'Prediction: {prediction}')
```

## Dependencies
Listed in `requirements.txt`, main libraries include:
- easyocr
- pandas
- scikit-learn
- joblib
- opencv-python
- numpy
- torch

---

