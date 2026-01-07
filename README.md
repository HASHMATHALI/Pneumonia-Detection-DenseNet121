# Pneumonia Detection using DenseNet121 with Uncertainty Estimation

This project implements a deep learning–based system for automated pneumonia detection from chest X-ray images using DenseNet121. The model integrates uncertainty estimation and interpretability techniques to enhance clinical reliability.

## 📌 Features
- DenseNet121 (ImageNet pretrained)
- Monte Carlo Dropout for uncertainty estimation
- Temperature Scaling for probability calibration
- Grad-CAM for explainability
- Binary classification: Normal vs Pneumonia

## 🗂 Dataset
- Kaggle Chest X-Ray (Pneumonia) Dataset
- This is the link so you can dowmload the dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia 
- Dataset not included due to size constraints

## 🛠 Technologies Used
- Python
- PyTorch
- Torchvision
- OpenCV
- Scikit-learn
- Matplotlib

## 🚀 How to Run
1. Install dependencies:
```bash
pip install -r requirements.txt
