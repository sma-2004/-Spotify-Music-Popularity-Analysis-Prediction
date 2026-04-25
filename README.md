# Spotify Music Popularity Analysis & Prediction

> An end-to-end machine learning project focused on predicting song popularity using Spotify audio features and metadata.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)]()
[![Scikit-learn](https://img.shields.io/badge/MachineLearning-Scikit--learn-orange.svg)]()
[![Streamlit](https://img.shields.io/badge/WebApp-Streamlit-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Project Overview

This project analyzes a dataset of **114,000+ Spotify tracks** to identify the key factors that influence song popularity. By leveraging audio attributes such as danceability, energy, loudness, tempo, valence, and acousticness, multiple machine learning models were developed to classify whether a song is likely to become popular.

The final optimized model achieved a **75% ROC-AUC score** and was integrated into an interactive Streamlit web application for real-time predictions.

---

## Objectives

- Understand patterns behind successful songs  
- Explore relationships between audio features and popularity  
- Build predictive machine learning models  
- Compare multiple algorithms and optimize performance  
- Deploy an interactive prediction system  

---

## Key Features

### Data Analysis

- Cleaned and processed large-scale Spotify dataset  
- Performed exploratory data analysis with visual insights  
- Studied feature correlations and popularity trends  
- Identified influential song characteristics  

### Machine Learning Pipeline

- Feature engineering and preprocessing  
- Train-test split with cross-validation  
- Model comparison across multiple algorithms  
- Hyperparameter tuning for better performance  
- Classification threshold optimization  
- Performance tracking using ROC-AUC, Precision, Recall, F1-score  

### Interactive Web Application

- Input custom song audio features  
- Get instant popularity predictions  
- View model confidence scores  
- Easy-to-use dashboard built with Streamlit  

---

## Model Performance

| Model | ROC-AUC Score |
|------|---------------|
| XGBoost | **0.75** |
| Random Forest | 0.74 |
| Gradient Boosting | 0.73 |
| Logistic Regression | 0.71 |

**Best Performing Model:** XGBoost

---

## Key Insights

- Songs with higher **energy** often perform better  
- **Danceability** shows strong positive impact on popularity  
- **Genre** remains an important predictive factor  
- Balanced duration tracks tend to perform well  
- Loud and engaging tracks show stronger popularity trends  

---

## Technology Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib  
- Seaborn  
- Plotly  
- Streamlit  

---

## Installation & Usage

```bash
git clone https://github.com/your-username/spotify-popularity-prediction.git
cd spotify-popularity-prediction
pip install -r requirements.txt
streamlit run app.py
