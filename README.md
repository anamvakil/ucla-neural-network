# UCLA Admission Neural Network Classifier

**CST2216 — Modularising and Deploying ML Code | Algonquin College**

A deployed neural network web application that predicts graduate admission outcomes for UCLA applicants based on academic profile features using scikit-learn's MLPClassifier. Built as part of a modular ML deployment project, converted from a Jupyter notebook into a production-ready Streamlit app.

🔗 **Live App:** [ucla-anam.streamlit.app](https://ucla-anam.streamlit.app)

---

## Overview

This app uses a Multi-Layer Perceptron (MLP) classifier to predict whether a student is admitted to UCLA based on their academic profile. No TensorFlow or Keras dependency — the full neural network is implemented using scikit-learn's MLPClassifier, keeping deployment lightweight and cloud-friendly.

| Metric | Value |
|---|---|
| Algorithm | MLPClassifier (MLP Neural Network) |
| Train Accuracy | 0.8525 |
| Test Accuracy | 0.8100 |
| CV Mean | 0.8375 |
| CV Std Dev | 0.0262 |
| Target Accuracy | 90% |
| Status | Below target — tanh activation recommended |

---

## Project Structure

```
ucla-neural-network/
├── app.py                  # Streamlit entry point
├── requirements.txt        # Dependencies (no pinned versions)
├── data/
│   └── dataset.csv         # UCLA applicant records
└── src/
    ├── __init__.py
    ├── data_loader.py      # Loads and validates the dataset
    ├── preprocessor.py     # Feature encoding and scaling
    ├── model.py            # MLPClassifier training and evaluation
    └── utils.py            # Shared helper functions
```

---

## App Features

**Tab 1 — Data Overview**
- Dataset shape, sample rows and summary statistics
- Feature distribution visualisations

**Tab 2 — Train & Evaluate**
- Adjustable hyperparameters: neurons per hidden layer, number of hidden layers, activation function, batch size, max iterations, admission threshold
- One-click training pipeline with training complete confirmation
- Accuracy metrics: train accuracy, test accuracy, CV mean and CV std dev
- Confusion matrix visualisation
- Training loss curve showing convergence across iterations
- Automatic below-target warning when test accuracy falls under 90%

**Tab 3 — Predict**
- Input sliders for applicant academic profile features
- Instant admission prediction from the trained MLP model
- Admission probability output with adjustable threshold

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Streamlit | UI framework and cloud deployment |
| scikit-learn | MLPClassifier neural network |
| pandas / NumPy | Data loading and preprocessing |
| Matplotlib / Seaborn | Confusion matrix, loss curve, scatter visualisation |
| GitHub | Version control and auto-deploy trigger |

---

## Running Locally

```bash
git clone https://github.com/anamvakil/ucla-neural-network
cd ucla-neural-network
pip install -r requirements.txt
streamlit run app.py
```

> Requires Python 3.9 or higher. No TensorFlow installation needed.

---

## Key Technical Notes

- **No TensorFlow dependency** — MLPClassifier from scikit-learn handles the full neural network; keeps requirements.txt lightweight and avoids cloud build failures
- **Automatic accuracy warning** — app detects when test accuracy falls below 90% and displays a banner prompting hyperparameter adjustment
- **tanh recommended over relu** — switching activation function is the primary recommendation to approach the 90% target
- **No pinned versions** in `requirements.txt` — compatible with Streamlit Cloud's Python 3.14 runtime
- **Pandas 2.x compatible** — all `inplace=True` calls replaced with explicit assignment syntax
- **data/ folder committed to repo** — required for Streamlit Cloud to locate the CSV at runtime
- `plt.close(fig)` called after every `st.pyplot()` call — resolves the infinite restart loop caused by matplotlib memory accumulation on Streamlit Cloud

---

## Dataset

UCLA applicant dataset with student academic and profile features for binary admission classification. Preprocessed through the modular src/ pipeline with encoding and standard scaling applied before training.

---

## Results

| Metric | Value |
|---|---|
| Train Accuracy | 0.8525 |
| Test Accuracy | 0.8100 |
| CV Mean Accuracy | 0.8375 |
| CV Std Dev | 0.0262 |

Test accuracy of 81% falls below the 90% target. The low CV standard deviation (0.0262) confirms the model is stable and generalising consistently — the gap is a tuning issue, not overfitting. Switching from relu to tanh activation is the recommended next step.

---

## Author

**Anam Vakil**  
BISI Graduate Certificate — Algonquin College  
[github.com/anamvakil](https://github.com/anamvakil)
