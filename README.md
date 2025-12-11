Fish Species Classification — Deep Learning Project
This repository contains a complete end‑to‑end deep learning project that classifies different species of fish using Convolutional Neural Networks (CNN) and multiple pre‑trained transfer‑learning models.

Project Overview
The goal of this project is to build an accurate and robust fish image classification system using modern computer vision techniques.
The project covers everything — from data preprocessing to model deployment using Streamlit.

Contents of This Repository
1. Dataset_____
The dataset contains images of multiple fish species.
It is organized in the format:
train/
val/
test/
Each folder contains sub‑folders of fish categories.

2. Model Training_____
This project trains six models:

From Scratch:
  Custom CNN model
Transfer Learning Models:
  VGG16
  ResNet50
  MobileNetV2
  InceptionV3
  EfficientNetB0

All models are:
  Fine‑tuned on the fish dataset
  Evaluated on validation and test sets
  Saved as "final_fish_classifier.h5" for future usage

3. Evaluation
  The project includes:
  Accuracy, Precision, Recall, F1‑Score
  Confusion Matrix
  Training curves (Accuracy vs Loss)
  Model comparison report
  Selection of the best model automatically

4. Streamlit Application
  An interactive app that allows users to:
  Upload a fish image
  Predict the fish category
  Display confidence score
  Show uploaded image

5. Deliverables
  This repository provides:
  Trained ("final_fish_classifier.h5") models
  Jupyter Notebook ("fish_classification.ipynb") for full training workflow
  app.py for Streamlit deployment
  Evaluation graphs
  README documentation

Tech Stack
  Python
  TensorFlow / Keras
  NumPy, Matplotlib, Seaborn
  Streamlit
  Jupyter Notebook

Key Features
  Full machine learning pipeline
  Hyperparameter‑tuned transfer learning models
  Clean, modular, and reusable code
  Real‑time prediction using Streamlit UI
  Production‑ready saved model

How to Run
Train Models
  Fish_Classification.ipynb

Start Streamlit App
  streamlit run app.py

Author,
Praveenkumar P
Data Analyst & Data Science
  
