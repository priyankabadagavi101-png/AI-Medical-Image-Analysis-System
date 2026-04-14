# AI-Powered Medical Image Analysis System

## Overview

This project develops a deep learning system that detects **pneumonia from chest X-ray images** using convolutional neural networks. The model is trained using **transfer learning with MobileNetV2**, enabling efficient and accurate classification of medical images.

The system includes model training, prediction, evaluation metrics, visualization, and a simple web application for real-time predictions.

---

## Features

* Pneumonia detection from chest X-ray images
* Transfer learning using MobileNetV2
* Image preprocessing and normalization
* Prediction script for new images
* Confusion matrix evaluation
* Accuracy visualization
* Streamlit web application for interactive predictions

---

## Project Structure

```
AI-Medical-Image-Analysis-System
│
├── data                    # Chest X-ray dataset
│
├── models                  # Saved trained models
│   └── pneumonia_model.h5
│
├── outputs                 # Generated outputs
│   ├── accuracy.png
│   └── confusion_matrix.png
│
├── src
│   ├── train_model.py      # Model training
│   ├── predict.py          # Predict pneumonia for a new image
│   ├── evaluate_model.py   # Model evaluation and metrics
│
├── test_images             # Sample images for testing
│
├── app.py                  # Streamlit web application
├── requirements.txt
└── README.md
```

---

## Dataset

The project uses the **Chest X-Ray Pneumonia Dataset**, which contains labeled chest X-ray images categorized into:

* **NORMAL**
* **PNEUMONIA**

Dataset structure:

```
data/chest_xray
│
├── train
├── test
└── val
```

---

## Model Architecture

The model uses **transfer learning with MobileNetV2**.

Pipeline:

```
Chest X-ray Image
        ↓
Resize to 224 × 224
        ↓
Image Normalization
        ↓
MobileNetV2 Feature Extraction
        ↓
Dense Layers
        ↓
Binary Classification (Normal / Pneumonia)
```

---

## Results

Model performance:

* **Accuracy:** ~83%
* High recall for pneumonia detection

Example confusion matrix:

![Confusion Matrix](outputs/confusion_matrix.png)

---

## How to Run the Project

### 1. Install dependencies

```
pip install -r requirements.txt
```

---

### 2. Train the model

```
python src/train_model.py
```

---

### 3. Predict using a new X-ray image

```
python src/predict.py
```

---

### 4. Evaluate the model

```
python src/evaluate_model.py
```

---

### 5. Run the web application

```
streamlit run app.py
```

Upload a chest X-ray image and the AI model will predict whether it shows **Normal lungs or Pneumonia**.

---

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* Seaborn
* Streamlit

---

## Future Improvements

* Improve accuracy using EfficientNet or ResNet
* Add Grad-CAM visualization for explainable AI
* Deploy the model as a cloud-based web application

---

## Author

**Priyanka Badagavi**
Electronics & Communication Engineering Student
Interested in Artificial Intelligence, Machine Learning, and Computer Vision
