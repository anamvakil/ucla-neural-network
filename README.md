# UCLA Admission Prediction — Neural Network

**Course:** Machine Learning  
**College:** Algonquin College  
**Project:** Modularizing and Deploying ML Code (Week 12)

---

## Project Overview

Predicts the probability of a student being admitted to UCLA using a Multi-Layer Perceptron (MLP) neural network. The target variable `Admit_Chance` is binarised at a 0.80 threshold — students with ≥80% predicted chance are classified as admitted.

**Success criterion:** test accuracy ≥ 90%.

---

## Project Structure

```
ucla-neural-network/
├── app.py                  # Streamlit application
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── Admission.csv       # Raw dataset (add manually)
├── models/                 # Saved .pkl files (auto-created)
├── logs/                   # Runtime logs (auto-created)
└── src/
    ├── __init__.py
    ├── data_loader.py      # CSV loading and validation
    ├── preprocessor.py     # Binarisation, encoding, scaling
    ├── model.py            # MLP training, evaluation, activation comparison
    ├── utils.py            # Logging, chart helpers, model persistence
    └── pipeline.py         # CLI-runnable end-to-end orchestrator
```

---

## Setup Instructions

```bash
git clone https://github.com/anamvakil/ucla-neural-network.git
cd ucla-neural-network
py -m venv venv
venv\Scripts\activate
py -m pip install -r requirements.txt
```

Place `Admission.csv` in the `data/` folder, then:

```bash
streamlit run app.py
```

---

## Dataset Features

| Feature | Range | Description |
|---|---|---|
| GRE_Score | 0–340 | Graduate Record Exam score |
| TOEFL_Score | 0–120 | English language test score |
| University_Rating | 1–5 | Bachelor's university quality |
| SOP | 1–5 | Statement of Purpose strength |
| LOR | 1–5 | Letter of Recommendation strength |
| CGPA | 0–10 | Undergraduate GPA |
| Research | 0 / 1 | Research experience |
| Admit_Chance | Target | Binarised at 0.80 threshold |

---

## Streamlit App Tabs

| Tab | Description |
|---|---|
|  Data Overview | Dataset stats, GRE vs TOEFL scatter, admission rate |
|  Train & Evaluate | Configure and train MLP, view loss curve, confusion matrix, CV scores, activation comparison |
|  Predict | Enter student profile, get live admission prediction with confidence |

---

## Live Demo

Deployed on Streamlit Cloud: **[INSERT LINK AFTER DEPLOYMENT]**
