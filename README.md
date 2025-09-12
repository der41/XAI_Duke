# Explainable AI Project — XAI_Duke

This repository contains work on **explainable machine learning (XAI)** applied to a churn prediction case study using the Telco dataset.  
The focus is on comparing **linear regression, logistic regression, and generalized additive models (GAMs)** for both **predictive performance** and **interpretability**.

---

## 📂 Project Structure
```
XAI_DUKE/
│
├── data/
│ └── Telco.csv # Telco churn dataset (Kaggle)
│
├── Notebooks/
│ └── Explainable_ML.ipynb # Main notebook with modeling & XAI analysis
│
├── .gitignore
├── README.md
└── requirements.txt # Python dependencies
```
---


---

## 📓 Notebook: `Explainable_ML.ipynb`

The notebook walks through:

1. **Data preparation** with the Telco churn dataset.  
2. **Modeling approaches**:  
   - Linear Probability Model (OLS)  
   - Logistic Regression  
   - Generalized Additive Models (pyGAM)  
3. **Evaluation metrics**: AUC, log-loss, Brier score, and calibration plots.  
4. **Explainability methods**:  
   - Coefficient interpretation  
   - Partial dependence plots   
5. **Business insights**: Quantifying model differences in terms of churn capture and revenue impact.  

---

## ⚙️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

