# Predicting Autism Spectrum Disorder (ASD) Using Machine Learning

This project focuses on predicting Autism Spectrum Disorder (ASD) using behavioral screening data and demographic features. The analysis includes **exploratory data analysis (EDA), feature importance evaluation, machine learning model comparison, hyperparameter tuning**, and model deployment preparation.

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
Autism Spectrum Disorder (ASD) is a developmental condition that affects communication, behavior, and social interaction. Early detection is critical for providing timely interventions and improving outcomes. This project utilizes machine learning algorithms to develop a model capable of detecting ASD based on behavioral screening scores and demographic data.

### **Objectives**
- Perform **data preprocessing and feature engineering** to prepare the dataset for analysis.
- Develop and evaluate multiple **machine learning algorithms** to identify the most effective predictive model.
- Prepare the best-performing model for **deployment** as a prototype for real-world use.

---

## **Dataset Description**
The dataset used in this project focuses on **adult ASD screening** and includes behavioral assessment scores and relevant demographic information.

### **Features:**
- **Behavioral Screening Scores**: Answers to ten screening questions (`A1_Score` to `A10_Score`).
- **Demographic Features**:
  - **Age**
  - **Gender**
  - **Ethnicity**
  - **Jaundice history** (yes/no)
  - **Family history of ASD** (yes/no)
  - **Relation** (e.g., self, parent)
- **Target Variable**:  
  - **Class/ASD**: Binary classification (YES/NO)

The dataset is imbalanced, with a higher prevalence of non-ASD cases, addressed using **SMOTE (Synthetic Minority Oversampling Technique)**.

---

## **Project Workflow**

### **1. Exploratory Data Analysis (EDA)**
- Analyzed distributions of numerical and categorical features.
- Investigated correlations between features and the target variable (`Class/ASD`).
- Detected class imbalance and applied oversampling techniques.

### **2. Data Preprocessing**
- Handled missing values and removed duplicate entries.
- Encoded categorical features using **label encoding** and **one-hot encoding**.
- Scaled numerical features for consistency.

### **3. Model Development & Evaluation**
- Trained multiple models:
  - **Baseline Models**: Logistic Regression
  - **Tree-Based Models**: Random Forest, Gradient Boosting
  - **Advanced Models**: XGBoost, LightGBM, CatBoost
- Evaluated models using:
  - **Accuracy**
  - **ROC-AUC**
  - **F1-Score**
  - **Confusion Matrix**

### **4. Hyperparameter Tuning**
- Optimized the **Random Forest** model using **GridSearchCV** to improve accuracy and stability.
- Validated model performance using cross-validation.

### **5. Feature Importance Analysis**
- Evaluated feature contributions using **permutation importance** from tree-based models.
- Identified key features influencing ASD detection.

### **6. Deployment Preparation**
- Saved the final Random Forest model using **joblib** for future deployment.
- Demonstrated prediction functionality with sample inputs.

---

## **Results**

### **Best Model: Random Forest (Refined)**
| Metric                 | Score  |
|------------------------|--------|
| **Cross-Validation Score** | 97.43% |
| **Test Accuracy**      | 98.05% |
| **ROC-AUC**            | 0.9986 |
| **Precision (ASD)**    | 0.98   |
| **Recall (ASD)**       | 0.98   |
| **F1-Score (ASD)**     | 0.98   |

### **Key Insights**
- **Behavioral screening scores** (`A1_Score` to `A10_Score`) were the most significant predictors of ASD likelihood.
- Demographic factors such as **age** and **ethnicity** contributed moderate predictive value.
- The model demonstrated balanced precision and recall across both ASD and non-ASD classes.

---

## **Installation & Running the Project**

### **1. Clone the Repository**
``` bash
git clone https://github.com/YourUsername/ASD-Detection-ML
cd ASD-Detection-ML
```

### **2. Set Up a Virtual Environment**
``` bash
python -m venv asd_env
source asd_env/bin/activate  # For macOS/Linux
asd_env\Scripts\activate     # For Windows
```

### **3. Install Dependencies**
``` bash
pip install -r requirements.txt
```

### **4. Run the Project**
``` bash
# Open Jupyter Notebook and run the analysis step by step
jupyter notebook ASD_Detection_Project.ipynb
```

---

## **Future Enhancements**
Potential improvements to this project include:
1. **Validating the model on external datasets** to ensure generalizability.
2. **Exploring ensemble methods** to further boost model accuracy.
3. **Deploying the model via an API** using **FastAPI** or **Flask** for real-time predictions.
4. **Developing a user interface** for ease of use in clinical settings.

---

## **Acknowledgments**
- **Dataset Source**: Public ASD screening dataset (specific source to be cited if applicable).  
- **Libraries Used**: **Scikit-learn, XGBoost, LightGBM, CatBoost, Pandas, Matplotlib, Seaborn, imbalanced-learn**  
- **Tools**: Python, Jupyter Notebook, joblib  

---

## **Contact**
For any questions or collaboration opportunities, reach out at:
- **Email**: devin.shrode@proton.me  
- **LinkedIn**: [linkedin.com/in/DevinShrode](https://www.linkedin.com/in/DevinShrode)  
- **GitHub**: [github.com/Devin-Shrode/Wine-Quality](https://github.com/Devin-Shrode/Autism-Spectrum)  

---
