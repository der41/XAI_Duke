# Explainable AI Project — XAI_Duke

This repository contains work on:

1. **Interpretable machine learning** applied to a churn prediction case study using the Telco dataset. The focus  is on comparing **linear regression, logistic regression, and generalized additive models (GAMs)** for both **predictive performance** and **interpretability**.
2. **Explainable machine learning (XAI)** analysis applied to a cancer detection model. Suggesting way to improve the model alignment to diagnostic needs using **SHAP, LIME, ICE and Dependecy Plots**.


---

## 📂 Project Structure
```
XAI_DUKE/
│
├── data/
│ └── Telco.csv # Telco churn dataset (Kaggle)
│
├── Notebooks/
│ ├──Machine_Learning_Court # Prosecutor on Cancer diagnostic ML model
│ └── Explainable_ML.ipynb # Interpretable analysis of Linear models
│
├── .gitignore
├── README.md
└── requirements.txt # Python dependencies to run the Notebooks
```
---

## 📓 Notebook: `Explainable_ML.ipynb`

The notebook walks through:

1. **Data preparation** with the Telco churn dataset (Kaggle).  
2. **Modeling approaches**:  
   - Linear Probability Model (OLS)  
   - Logistic Regression  
   - Generalized Additive Models (pyGAM)  
3. **Evaluation metrics**: AUC, log-loss, Brier score, and calibration plots.  
4. **Explainability methods**:  
   - Coefficient interpretation  
   - Partial dependence plots   
5. **Business insights**: Quantifying model differences in terms of churn capture and revenue impact.  

## 📓 Notebook: `Machine_Learning_Court.ipynb`

The notebook walks through:

1. **Modeling** with the breast cancer dataset (Sci-kit Learn).  
2. **Modeling approaches**:  
   - Random Forest
3. **Evaluation metrics**: precision, recall, f1-score   
4. **Explainability methods**:  
   - SHAP for Global Explainability
   - LIME for Local Explainability
   - ICE Plot
   - Partial Dependecy Plot   
5. **Business insights**: Model require refinement. Optimization based on Malign recall should be be pursuit. 

---

## ⚙️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Also is possible to run the notebooks on Google Colab directly.
