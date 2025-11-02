# Explainable AI Project — XAI_Duke

This repository collects a series of explainable AI experiments that span classical machine learning, modern deep learning, and model robustness investigations. The notebooks document hands-on explorations of:

1. **Interpretable machine learning** applied to a churn prediction case study using the Telco dataset. The focus is on comparing **linear regression, logistic regression, and generalized additive models (GAMs)** for both **predictive performance** and **interpretability**.
2. **Explainable machine learning (XAI)** analyses applied to a cancer detection model, highlighting how **SHAP, LIME, ICE, and partial dependence plots** can surface issues in model alignment with diagnostic needs.
3. **Explainable deep learning** with Grad-CAM and Grad-CAM variants for understanding where a ResNet-50 model focuses when classifying images.
4. **Robustness and adversarial attacks** through universal adversarial patches applied to ImageNet-class CNNs.
5. **Global explainability techniques** such as PDP, ICE, and ALE to probe feature behaviour in tabular models.

---

## 📂 Project Structure
```
XAI_Duke/
│
├── data/
│   └── Telco.csv              # Telco churn dataset (Kaggle)
│
├── Notebooks/
│   ├── Adversarial_attacks_patches.ipynb   # Universal adversarial patch attacks on ImageNet models
│   ├── Explainable_DL_Pytorch.ipynb        # Grad-CAM explainability for ResNet-50 image models
│   ├── Explainable_ML.ipynb                # Interpretable analysis of linear and additive models for churn
│   ├── Explainable_ML_II.ipynb             # PDP, ICE, and ALE exploration for tabular models
│   └── Machine_learning_court.ipynb        # Case-based explainability exercises for critical ML systems
│
├── README.md
└── requirements.txt            # Python dependencies to run the notebooks
```
---

## 📓 Notebook: `Explainable_ML.ipynb`

The notebook walks through:

1. **Data preparation** with the Telco churn dataset (Kaggle).
2. **Modelling approaches**:
   - Linear Probability Model (OLS)
   - Logistic Regression
   - Generalized Additive Models (pyGAM)
3. **Evaluation metrics**: AUC, log-loss, Brier score, and calibration plots.
4. **Explainability methods**:
   - Coefficient interpretation
   - Partial dependence plots
5. **Business insights**: Quantifying model differences in terms of churn capture and revenue impact.

## 📓 Notebook: `Explainable_ML_II.ipynb`

This follow-on notebook builds on the churn case study to concentrate on **global explainability plots** for tabular models:

- Generates updated exploratory analysis and correlation diagnostics.
- Compares **Partial Dependence Plots (PDP)**, **Individual Conditional Expectation (ICE)** curves, and **Accumulated Local Effects (ALE)** to highlight heterogeneous feature behaviour.
- Reuses helper utilities from earlier notebooks to ensure consistent preprocessing and visualization.

## 📓 Notebook: `Machine_learning_court.ipynb`

The "Machine Learning Court" notebook frames explainability as forensic analysis across three sensitive scenarios (loan approval, breast cancer diagnosis, and recidivism prediction):

1. **Model setup**: Baseline models such as random forests for tabular classification problems.
2. **Evaluation metrics**: Precision, recall, and F1-score with attention to class-dependent performance (e.g., malignant recall).
3. **Explainability workflow**: Prompts the practitioner to apply SHAP, LIME, ICE, and partial dependence plots to audit the model’s decisions and uncover potential bias or blind spots.
4. **Guided reflection**: Discusses risk trade-offs and improvements needed to make the models trustworthy in high-stakes settings.

## 📓 Notebook: `Explainable_DL_Pytorch.ipynb`

This notebook investigates **visual explanations for deep learning classifiers**:

- Uses a pre-trained **ResNet-50** network from torchvision.
- Applies **Grad-CAM and Grad-CAM variants** to both synthetic (Sora-generated) and Kaggle car detection images.
- Studies how explanations change when masking or blurring regions, and when targeting different classes in multi-object scenes.
- Concludes with reflections on when model attention aligns—or fails to align—with human expectations.

## 📓 Notebook: `Adversarial_attacks_patches.ipynb`

The adversarial notebook examines **universal adversarial patch attacks** on ImageNet-trained CNNs:

- Loads ImageNet-pretrained architectures (ResNet-34 by default) via torchvision.
- Demonstrates how pre-computed adversarial patches can consistently fool classifiers across diverse images.
- Provides utilities for downloading sample data, applying attacks, and visualizing both the patched inputs and resulting misclassifications.
- Encourages experimentation with alternative architectures and patches to understand robustness limits.

---

## ⚙️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

The notebooks can be executed locally or opened directly in Google Colab via the embedded badges.
