# K-pop Album Sales Prediction

## Project Overview

This project aims to predict **K-pop album sales** using machine learning models based on artist popularity, album characteristics, and social media metrics.

The goal is to identify key factors that influence commercial success and build a model capable of forecasting future releases.

---

## Objectives

* Collect and preprocess K-pop album data
* Perform exploratory data analysis (EDA)
* Engineer meaningful features
* Train and compare multiple ML models
* Interpret model predictions

---

## Models Used

We experiment with several regression models:

* Linear Regression
* Random Forest
* Support Vector Machine (SVM)
* XGBoost
* CatBoost

---

## Features (examples)

* Number of social media followers
* Number of album versions
* Time since last release
* Fan engagement indicators
* Custom engineered features

---

## Project Structure

```
project/
├── data/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── data/
│   │   └── preprocess.py
│   ├── models/
│   │   ├── train.py
│   │   └── predict.py
│   └── utils.py
├── config.yaml
├── models/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt
```

---

## Usage

### 1. Run EDA

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 2. Train models

```bash
python src/models/train.py
```

### 3. Make predictions

```bash
python src/models/predict.py
```

---

## Evaluation Metrics

* MAE (Mean Absolute Error)
* MSE / RMSE
* R² Score

---

## Inspiration

This project is inspired by research on music industry analytics and popularity prediction:
Kim J., Kim J., Seo M., Song J. Analysis of K-pop album sales using machine learning models // Korean J Appl Stat, 2025, 38(2), p. 227-249.

DOI:
 [https://doi.org/10.5351/KJAS.2025.38.2.227]

---

## Team

* Team Lead, Data Engineer: @polly842
* ML Engineer, документация/тесты: @aliences
