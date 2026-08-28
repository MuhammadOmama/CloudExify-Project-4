# Disease Prediction (Classification)

## 📌 Project Overview
This project develops an end-to-end medical classification system to predict patient disease status based on clinical diagnostics. The pipeline handles class imbalance, normalizes feature distributions, trains both Logistic Regression and Decision Tree classifiers, evaluates performance across multiple classification metrics, and provides an interactive Streamlit web application for real-time patient assessment.

---

## 📊 Dataset Description
The model is trained on diagnostic health metrics from the **Pima Indians Diabetes Database**:
* **Dataset Size**: 768 patient records (500 Healthy, 268 Disease Cases)
* **Target Variable**: `Disease` (`0` = Healthy / No Disease, `1` = Disease Likely)
* **Clinical Features**:
  * `Pregnancies`: Number of times pregnant
  * `Glucose`: Plasma glucose concentration after 2 hours in an oral glucose tolerance test
  * `BloodPressure`: Diastolic blood pressure ($mm\ Hg$)
  * `SkinThickness`: Triceps skin fold thickness ($mm$)
  * `Insulin`: 2-Hour serum insulin ($\mu U/ml$)
  * `BMI`: Body mass index ($weight\ in\ kg / (height\ in\ m)^2$)
  * `DiabetesPedigreeFunction`: Diabetes pedigree function score
  * `Age`: Patient age in years

---

## 🛠️ Tech Stack & Key Libraries
* **Language**: Python 3
* **Machine Learning & Preprocessing**: `scikit-learn` (`StandardScaler`, `LogisticRegression`, `DecisionTreeClassifier`), `joblib`
* **Data Manipulation**: `pandas`, `numpy`
* **Data Visualization**: `matplotlib`, `seaborn`
* **Web Deployment**: `streamlit`

---

## 📈 Model Performance & Evaluation

Models were evaluated using an 80/20 stratified split to preserve class proportions:

| Metric | Logistic Regression | Decision Tree (Best Model) | Description |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 74.7% | **88.3%** | Percentage of correct predictions |
| **Precision** | 60.9% | **78.1%** | True positive rate among predicted positives |
| **Recall** | 77.8% | **92.6%** | Proportion of actual disease cases identified |
| **F1-Score** | 0.683 | **0.847** | Harmonic mean of precision and recall |

### Confusion Matrix (Decision Tree)
```text
[[86  14]   <- [True Negative: 86,  False Positive: 14]
 [ 4  50]]  <- [False Negative: 4,  True Positive: 50]
```

### Top 3 Predictive Features
1. **Insulin**: 71.8%
2. **SkinThickness**: 7.5%
3. **Glucose**: 7.1%

---

## 📉 Visualizations

### 1. Receiver Operating Characteristic (ROC) Curve
Evaluates the diagnostic trade-off between True Positive Rate (Sensitivity) and False Positive Rate across decision thresholds.

<img width="800" height="600" alt="roc_curve" src="https://github.com/user-attachments/assets/dfbe92c8-7b1e-4c59-baaa-714b6178858d" />


### 2. Feature Importance
Highlights the clinical features with the strongest predictive weight in diagnosing disease risk (e.g., `Insulin`, `SkinThickness`, and `Glucose`).

<img width="1000" height="600" alt="feature_importance_disease" src="https://github.com/user-attachments/assets/5280fea9-2aae-49db-899c-4ca0219e76c6" />


---

## 🚀 Interactive Web Application (Streamlit)

The trained Decision Tree model and feature artifacts are serialized using `joblib` and served via Streamlit for clinical decision support.

### Run the App Locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MuhammadOmama/CloudExify-Project-4
   ```

2. **Install dependencies**:
   ```bash
   pip install streamlit pandas scikit-learn joblib matplotlib
   ```

3. **Launch the application**:
   ```bash
   streamlit run app.py
   ```

---

## 📁 Repository Structure

```text
├── diabetes.csv                   # Medical diagnostics dataset
├── disease_prediction.ipynb       # Jupyter notebook with EDA, training & evaluation
├── disease_model.pkl              # Serialized classification model
├── disease_scaler.pkl             # Serialized StandardScaler artifact
├── disease_features.pkl           # Expected column ordering artifact
├── app.py                         # Streamlit interactive application
├── roc_curve.png                  # Saved ROC evaluation plot
├── feature_importance_disease.png # Saved feature importance chart
└── README.md                      # Project documentation
```

---
