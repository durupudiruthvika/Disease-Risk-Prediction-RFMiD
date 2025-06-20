# 🧠 Leveraging Machine Learning to Predict Disease Risk Levels from Multiple Eye Condition Indicators

## 📑 Table of Contents

1. [📋 Project Overview](#-overview)
2. [🎯 Key Features](#-features)
3. [⚙️ Tech Stack & Dataset](#-technologies-used)
4. [🚀 Installation & Usage](#-usage)
5. [🔄 Methodology & Workflow](#-project-workflow)
6. [📊 Results & Visualizations](#-sample-outputs)
7. [📈 Model Performance](#-evaluation-metrics)
8. [🔮 Future Roadmap](#-future-enhancements)
9. [Published Paper](#-published-paper)
10. [👥 Development Team](#-team-members-amrita-school-of-computing-bengaluru)
11. [👨‍🏫 Academic Supervisors](#-mentor)
12. [🙏 Acknowledgments](#-acknowledgment)

---

## 📋 Overview

This project applies machine learning to predict disease risk levels using ocular indicators from the **RFMiD (Retinal Fundus MultiDisease Image)** dataset. It combines multiple classification algorithms and Explainable AI (XAI) to improve diagnostic precision and interpretability in medical applications.

---

## ✨ Features

### 🔍 Comprehensive Analysis
- Multi-model comparison with rich evaluation metrics
- Feature similarity via Minkowski distance
- Histogram and scatter plot-based insights

### 🤖 Robust ML Pipeline
- Stratified cross-validation for unbiased results
- Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
- Ensemble approaches for performance boosting

### 📊 Explainable AI Integration
- SHAP for global feature impact analysis
- LIME for instance-level interpretability

### 🏥 Healthcare Applications
- Early risk identification
- Aid in clinical decisions
- Resource prioritization based on risk scores

---

## 🛠️ Technologies Used

### Languages & Libraries
- Python 3.x
- pandas, numpy
- scikit-learn
- matplotlib, seaborn, plotly
- xgboost
- SHAP, LIME

### Dataset
- **RFMiD Dataset** from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification)
- 1,920 samples, 47 features, multiple disease risk classes

---

## 🚀 Usage

### Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install shap lime xgboost plotly
```

### Dataset Setup

1. Download the dataset from Kaggle
2. Place it in the root directory
3. Ensure it includes all 47 numerical features

### Sample Code to Run Naive Bayes

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load data into X, y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = GaussianNB()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

## 🔄 Project Workflow

1. **Data Preprocessing**
   - Standard scaling
   - Stratified train-test split

2. **Feature Analysis**
   - Histograms, scatter plots
   - Minkowski distance calculations

3. **Model Training**
   - Decision Trees, Random Forest, Gradient Boosting, Naive Bayes

4. **Evaluation & XAI**
   - SHAP & LIME for explainability
   - Metric-based comparison

---

## 📊 Sample Outputs

- ROC-AUC plots for all models
- Confusion matrices
- SHAP value bar charts
- LIME local explanation charts
- Feature correlation graphs

---

## 📈 Evaluation Metrics

| Model                | Accuracy  | Precision  | Recall    | F1-Score  | ROC-AUC    |
| -------------------- | --------- | ---------- | --------- | --------- | ---------- |
| 🏆 **Naive Bayes**   | **99.7%** | **1.0000** | **99.6%** | **99.8%** | **0.9982** |
| 🌳 Decision Tree     | 98.7%     | 1.0000     | 98.3%     | 99.1%     | 0.9919     |
| 🌲 Random Forest     | 98.4%     | 1.0000     | 98.0%     | 99.0%     | 0.9996     |
| 🚀 Gradient Boosting | 97.4%     | 1.0000     | 96.7%     | 98.3%     | 0.9884     |

---

## 🔮 Future Enhancements

### 🚧 Planned Improvements

- CNN integration for retinal image pattern detection
- Mobile-friendly diagnostic interface
- Real-time patient data integration

### 🔬 Advanced Research Directions

- Federated learning for secure collaboration
- AutoML for optimized model selection
- EHR integration via secure APIs
- Edge & cloud deployment for scalability

---
## 📄 Published Paper

**Title:** Leveraging Machine Learning to Predict Disease Risk Levels from Multiple Eye Condition Indicators

**Authors:** Mudumala Varnika Narayani, Naga Ruthvika Durupudi, Nunnaguppala Rohit, Tejashwini Vadeghar, Jyotsna C , Aiswariya Milan K

**Publication:** 5th International Conference on Pervasive Computing and Social Networking (ICPCSN-2025)
**Conference Date:** 14th–16th May 2025
**Conference Location:** R P Sarathy Institute of Technology, Salem, Tamil Nadu, India (Hybrid Mode)

**Publisher:**  IEEE Computational Intelligence Society
## 👥 Team Members (Amrita School of Computing, Bengaluru)

**B.Tech CSE, Batch of 2022–2026**  
**Amrita School of Computing, Bengaluru**  
**Amrita Vishwa Vidyapeetham, India**

- **Naga Ruthvika Durupudi**
- **Mudumala Varnika Narayani**
- **Nunnaguppala Rohit**

---

## 👨‍🏫 Mentor

- **Dr. Jyotsna C.**  
  Associate Professor  
  Department of Computer Science and Engineering  
  Amrita School of Computing, Bengaluru

- **Dr. Aiswariya Milan K.**  
  Assistant Professor  
  Department of Computer Science and Engineering  
  Amrita School of Computing, Bengaluru

---

## 🙏 Acknowledgment

This project was developed as part of the academic curriculum for the **B.Tech CSE Batch of 2022–2026**, under the mentorship of **Dr. Jyotsna C.** and **Dr. Aiswariya Milan K.**, at the **Amrita School of Computing**, **Amrita Vishwa Vidyapeetham**, Bengaluru campus.

---

## 🏛️ Developed At

**Amrita School of Computing, Bengaluru**  
**Amrita Vishwa Vidyapeetham, India**
