# **Predicting Autism Spectrum Disorder (ASD) Using Machine Learning**

This project focuses on **predicting Autism Spectrum Disorder (ASD)** based on **behavioral screening responses and demographic features**. The analysis includes **exploratory data analysis (EDA), feature importance evaluation, machine learning model comparison, hyperparameter tuning**, and model deployment preparation.

---

## **Table of Contents**
1. [Introduction](#introduction)  
2. [Dataset Description](#dataset-description)  
3. [Project Workflow](#project-workflow)  
4. [Results](#results)  
5. [Installation & Running the Project](#installation--running-the-project)  
6. [Future Enhancements](#future-enhancements)  
7. [Acknowledgments](#acknowledgments)  
8. [Contact](#contact)  

---

## **Introduction**
Autism Spectrum Disorder (ASD) is a **neurodevelopmental condition** that affects communication, behavior, and social interaction. Early detection is **critical** for timely interventions and improving outcomes. This project leverages **machine learning techniques** to develop a predictive model capable of **identifying individuals at risk for ASD** using **behavioral screening scores and demographic data**.

### **Objectives**
- **Perform data preprocessing** and feature engineering to prepare the dataset for analysis.
- **Evaluate multiple machine learning algorithms** to determine the best-performing model.
- **Interpret model decisions** using feature importance and SHAP analysis.
- **Prepare the selected model for deployment** as a prototype for ASD detection.

---

## **Dataset Description**
The dataset contains **behavioral screening responses and demographic information** for **adult ASD screening**. 

### **Key Features:**
- **Behavioral Screening Responses**: Answers to 10 standardized screening questions (`A1_Score` to `A10_Score`).
- **Demographic Attributes**:
  - **Age**
  - **Gender**
  - **Ethnicity**
  - **Jaundice history** (Yes/No)
  - **Family history of ASD** (Yes/No)
  - **Relation to ASD individual** (e.g., Self, Parent)
- **Target Variable**:  
  - **ASD Classification** (`YES` = ASD, `NO` = No ASD)

The dataset was **imbalanced** with a higher number of **non-ASD cases**, which was addressed using **SMOTE (Synthetic Minority Oversampling Technique)**.

---

## **Project Workflow**

### **1. Exploratory Data Analysis (EDA)**
- **Feature distribution analysis** to understand data characteristics.
- **Correlation analysis** to assess relationships between input variables and ASD classification.
- **Class imbalance detection** and **oversampling using SMOTE** to ensure fair model learning.

### **2. Data Preprocessing**
- **Missing values handled** and **duplicate entries removed**.
- **Categorical encoding applied** (label encoding & one-hot encoding).
- **Feature scaling** to ensure numerical consistency.

### **3. Model Development & Evaluation**
- **Trained multiple models**, including:
  - **Logistic Regression**
  - **Random Forest**
  - **Gradient Boosting**
  - **XGBoost**
  - **LightGBM**
  - **CatBoost** (Final Model)
- **Evaluated models based on**:
  - **Accuracy**
  - **ROC-AUC**
  - **F1-Score**
  - **Confusion Matrix**

### **4. Feature Importance & SHAP Analysis**
- **Tree-based feature importance** identified key predictive variables.
- **SHAP (Shapley Additive Explanations)** provided interpretability for model predictions.
- **Key finding**: **Behavioral screening scores were the strongest predictors of ASD.**

### **5. Confusion Matrix Insights**
- **CatBoost model demonstrated the lowest false negative rate**, making it ideal for ASD screening.
- **Minimal false positives ensured fewer unnecessary ASD evaluations**.
- **Overall balanced precision-recall performance**, supporting clinical reliability.

### **6. Model Deployment Preparation**
- **The final model (CatBoost) was saved using `joblib`** for efficient deployment.
- **Validated predictions with sample test cases** to ensure correctness.

---

## **Results**

### **Best Model: CatBoost**
| Metric                 | Score  |
|------------------------|--------|
| **Cross-Validation Accuracy** | 96.70% |
| **Test Accuracy**      | 97.56% |
| **ROC-AUC**            | 0.9982 |
| **Precision (ASD)**    | 0.9619 |
| **Recall (ASD)**       | 0.9902 |
| **F1-Score (ASD)**     | 0.9758 |

### **Key Insights**
- **Behavioral screening scores** (`A1_Score` to `A10_Score`) were the most important predictors of ASD.
- **CatBoost achieved the highest recall**, making it **preferable for early ASD detection**.
- **False negatives were minimized**, reducing the risk of undiagnosed ASD cases.

---

## **Installation & Running the Project**

### **1. Clone the Repository**
```
bash
git clone https://github.com/YourUsername/ASD-Detection-ML
cd ASD-Detection-ML
```

### **2. Set Up a Virtual Environment**
```
bash
python -m venv asd_env
source asd_env/bin/activate  # macOS/Linux
asd_env\Scripts\activate     # Windows
```

### **3. Install Dependencies**
```
bash
pip install -r requirements.txt
```

### **4. Running Predictions**
Load the saved **CatBoost model** and test predictions:
```
python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("refined_catboost_model.pkl")

# Example input (ensure this matches the feature order in training)
sample_data = pd.DataFrame([[35, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1]], 
                           columns=['Age', 'Sex', 'Ethnicity_Asian', ..., 'A10_Score'])

# Make a prediction
prediction = model.predict(sample_data)
print(f"Predicted ASD Status: {prediction[0]} (0 = No ASD, 1 = ASD)")
```

---

## **Future Enhancements**
Potential improvements include:
1. **Validating the model on external datasets** to confirm generalizability.
2. **Exploring ensemble learning techniques** for even higher predictive accuracy.
3. **Deploying the model via a web-based API** (Flask/Django) for real-world use.
4. **Developing a user-friendly UI** for seamless interaction.

---

## **Acknowledgments**
- **Dataset Source**: Public ASD screening dataset ([Kaggle Link](https://www.kaggle.com/datasets/umeradnaan/autism-screening/data)).  
- **Libraries Used**: **Scikit-learn, XGBoost, LightGBM, CatBoost, Pandas, SHAP, Matplotlib, Seaborn, imbalanced-learn**  
- **Tools**: Python, Jupyter Notebook, joblib  

---

## **Contact**
For any questions or collaboration opportunities, reach out at:
- **Email**: devin.shrode@proton.me  
- **LinkedIn**: [linkedin.com/in/DevinShrode](https://www.linkedin.com/in/DevinShrode)  
- **GitHub**: [github.com/Devin-Shrode](https://github.com/Devin-Shrode)  

---
