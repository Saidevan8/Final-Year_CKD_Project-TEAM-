
---

# Chronic Kidney Disease (CKD) Prediction and Image-Based Analysis

![CKD Prediction](images/ckd_logo.png)

## Overview
This project is dedicated to predicting Chronic Kidney Disease (CKD) risk using machine learning, incorporating both clinical data and ultrasound imaging. The dual approach leverages traditional health metrics and recent advances in image analysis, with the goal of providing a comprehensive prediction model for CKD assessment.

## Table of Contents
1. [Project Overview](#overview)
2. [Data Sources](#data-sources)
3. [Project Structure](#project-structure)
4. [Setup and Installation](#setup-and-installation)
5. [Training and Evaluation](#training-and-evaluation)
6. [Streamlit App Deployment](#streamlit-app-deployment)
7. [License](#license)

## Data Sources
This project relies on two main datasets:
1. **Clinical Data**: Contains various demographic and health indicators such as age, BMI, blood pressure, and lab results. Sourced from Kaggle's CKD datasets.
2. **Ultrasound Image Data**: Ultrasound images sourced from [Open Kidney Ultrasound Data](https://ubc.ca1.qualtrics.com/jfe/form/SV_1TfBnLm1wwZ9srk), containing over 500 B-mode ultrasound images with fine-grained annotations.

## Project Structure
Here's an overview of the project's structure:

```plaintext
CKD_Prediction_Project/
├── data/
│   ├── clinical_data.csv           # CKD clinical dataset
│   ├── ultrasound_images/          # Folder for ultrasound images
│   └── model_comparison.csv        # Model performance comparison
├── images/                         # Visual assets for documentation
├── models/
│   ├── ckd_model.joblib            # Trained CKD model
│   ├── lsvm_model.joblib           # Trained Linear SVM model
├── src/
│   ├── train_model.py              # Training script for the CKD model
│   ├── predict_clinical.py         # Prediction using clinical data
│   ├── predict_image.py            # Image-based prediction script
│   ├── image_processing.py         # Functions for preprocessing ultrasound images
│   ├── evaluation/                 # Scripts for model evaluation
│       ├── model_metrics.py        # Calculates metrics like accuracy, F1, ROC-AUC
│       ├── visualizations.py       # Generates plots and confusion matrices
│   └── utils.py                    # Utility functions
├── streamlit_app/                  # Streamlit application for model deployment
│   ├── app.py                      # Main Streamlit app
│   └── evolution/                  # Directory for evaluation visualizations
├── plots/                          # Directory for saving visualization plots
├── venv/                           # Virtual environment for dependencies
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── LICENSE                         # Project license information

## Setup and Installation
1. **Clone the Repository**
   ```bash
   git clone <repo_link>
   cd CKD_Prediction_Project
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate       # For macOS/Linux
   venv\Scripts\activate          # For Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Training and Evaluation
1. **Model Training**  
   Run `train_model.py` to train both the CKD clinical and image-based models.

2. **Evaluation**  
   Use `evaluation/model_metrics.py` for performance metrics and `visualizations.py` for generating plots, saved in the `evolution` folder for documentation.

## Streamlit App Deployment
Deploy the prediction models on a Streamlit web app with `app.py` in `streamlit_app/`. The app supports prediction based on clinical data and ultrasound images.

## License
This project is licensed under the [CC BY-NC-SA License](LICENSE), permitting non-commercial use with attribution.

---

