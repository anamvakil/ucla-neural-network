"""
app.py
------
Streamlit app — UCLA Admission Prediction via Neural Network
Tabs: Data Overview | Train & Evaluate | Predict
Dataset loads automatically from data/Admission.csv
"""
 
import os
import sys
import logging
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
 
sys.path.insert(0, os.path.dirname(__file__))
 
from src.data_loader import load_data
from src.preprocessor import (
    binarise_target, drop_unnecessary_columns,
    encode_features, split_features_target, scale_features,
)
from src.model import (
    train_mlp, evaluate_model, evaluate_train_accuracy,
    cross_validate_model, compare_activations, get_loss_curve,
)
from src.utils import (
    setup_logging, plot_confusion_matrix, plot_loss_curve,
    plot_activation_comparison, plot_cv_scores, plot_scatter,
)
 
setup_logging()
logger = logging.getLogger(__name__)
 
DATA_PATH = os.path.join("data", "Admission.csv")
 
st.set_page_config(
    page_title="UCLA Admission Predictor",
    page_icon=None,
    layout="wide",
)
 
st.title("UCLA Admission Prediction — Neural Network")
st.caption("CST2216 — Business Intelligence System Infrastructure | Algonquin College")
st.markdown("---")
 
@st.cache_data
def load_raw_data():
    return pd.read_csv(DATA_PATH)
 
try:
    raw_df = load_raw_data()
except FileNotFoundError:
    st.error("Dataset not found at data/Admission.csv. Please ensure the file is committed to the repository.")
    st.stop()
 
tab1, tab2, tab3 = st.tabs(["Data Overview", "Train & Evaluate", "Predict"])
 
 
# TAB 1
with tab1:
    st.header("Dataset Overview")
    st.write("UCLA Admission dataset — 500 graduate applicants with 7 academic and research features.")
 
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", raw_df.shape[0])
    col2.metric("Features", raw_df.shape[1] - 1)
    col3.metric("Missing Values", int(raw_df.isnull().sum().sum()))
 
    st.subheader("Sample Records")
    st.dataframe(raw_df.head(10), use_container_width=True)
 
    st.subheader("Summary Statistics")
    st.dataframe(raw_df.describe(), use_container_width=True)
 
    st.subheader("GRE Score vs TOEFL Score by Admission Outcome")
    st.caption("Admission threshold applied at 0.80.")
    try:
        df_bin = binarise_target(raw_df.copy())
        fig = plot_scatter(df_bin)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Could not render scatter plot: {e}")
 
    admitted = int((raw_df["Admit_Chance"] >= 0.8).sum())
    total = len(raw_df)
    st.subheader("Admission Rate at 0.80 Threshold")
    st.metric("Students Meeting Threshold", f"{admitted} / {total}  ({admitted/total*100:.1f}%)")
 
 
# TAB 2
with tab2:
    st.header("Train & Evaluate Neural Network")
    st.write("Configure the MLP hyperparameters below and run the training pipeline.")
 
    with st.expander("Hyperparameter Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            neurons = st.slider("Neurons in hidden layer", 2, 20, 3)
            hidden_layers = st.selectbox("Number of hidden layers", [1, 2, 3])
            activation = st.selectbox("Activation function", ["relu", "tanh", "logistic"])
        with col2:
            batch_size = st.slider("Batch size", 10, 100, 50, step=10)
            max_iter = st.slider("Max iterations", 100, 500, 200, step=50)
            threshold = st.slider("Admission threshold", 0.5, 0.9, 0.8, step=0.05)
 
    compare_acts = st.checkbox("Compare all activation functions (relu / tanh / logistic)", value=True)
 
    if st.button("Run Training Pipeline"):
        try:
            with st.spinner("Preprocessing data..."):
                df2 = binarise_target(raw_df.copy(), threshold=threshold)
                df2 = drop_unnecessary_columns(df2)
                df2 = encode_features(df2)
                X, y = split_features_target(df2)
                feat_names = X.columns.tolist()
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=123
                )
                X_tr_sc, X_te_sc, scaler = scale_features(X_train, X_test)
 
            with st.spinner("Training neural network..."):
                hidden = tuple([neurons] * hidden_layers)
                mlp = train_mlp(X_tr_sc, y_train,
                                hidden_layer_sizes=hidden,
                                activation=activation,
                                batch_size=batch_size,
                                max_iter=max_iter)
 
            st.success("Training complete.")
 
            train_acc = evaluate_train_accuracy(mlp, X_tr_sc, y_train)
            ev = evaluate_model(mlp, X_te_sc, y_test, model_name="MLP")
            cv = cross_validate_model(mlp, X_tr_sc, y_train)
 
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Train Accuracy", f"{train_acc:.4f}")
            m2.metric("Test Accuracy", f"{ev['accuracy']:.4f}")
            m3.metric("CV Mean", f"{cv['mean_accuracy']:.4f}")
            m4.metric("CV Std Dev", f"{cv['std_deviation']:.4f}")
 
            if ev["accuracy"] >= 0.9:
                st.success("Model meets the 90% accuracy target.")
            else:
                st.warning(f"Model is below the 90% target at {ev['accuracy']:.2%}. Try adjusting hyperparameters.")
 
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Confusion Matrix")
                fig_cm = plot_confusion_matrix(ev["confusion_matrix"], "MLP")
                st.pyplot(fig_cm)
                plt.close(fig_cm)
            with col_b:
                st.subheader("Training Loss Curve")
                fig_lc = plot_loss_curve(get_loss_curve(mlp), "MLP")
                st.pyplot(fig_lc)
                plt.close(fig_lc)
 
            st.subheader("5-Fold Cross-Validation Scores")
            fig_cv = plot_cv_scores(cv["scores"], "MLP")
            st.pyplot(fig_cv)
            plt.close(fig_cv)
 
            if compare_acts:
                with st.spinner("Comparing activation functions..."):
                    act_df = compare_activations(X_tr_sc, y_train, X_te_sc, y_test)
                st.subheader("Activation Function Comparison")
                st.dataframe(act_df, use_container_width=True)
                fig_ac = plot_activation_comparison(act_df)
                st.pyplot(fig_ac)
                plt.close(fig_ac)
 
            st.session_state["nn_state"] = {
                "model": mlp, "scaler": scaler,
                "feature_names": feat_names, "threshold": threshold,
            }
 
        except FileNotFoundError:
            st.error("Dataset not found. Please ensure Admission.csv is in the data/ folder.")
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            logger.exception("Tab2 error")
 
 
# TAB 3
with tab3:
    st.header("Predict Admission for a New Applicant")
 
    if "nn_state" not in st.session_state:
        st.info("Train the model first in the Train & Evaluate tab.")
    else:
        ns = st.session_state["nn_state"]
        st.write("Enter the applicant's academic profile below.")
 
        col1, col2 = st.columns(2)
        with col1:
            gre = st.slider("GRE Score (out of 340)", 260, 340, 310)
            toefl = st.slider("TOEFL Score (out of 120)", 90, 120, 107)
            uni_rating = st.selectbox("University Rating (1 to 5)", [1, 2, 3, 4, 5], index=2)
            sop = st.select_slider("Statement of Purpose Strength", options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], value=3.0)
        with col2:
            lor = st.select_slider("Letter of Recommendation Strength", options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], value=3.0)
            cgpa = st.slider("Undergraduate CGPA (out of 10)", 6.0, 10.0, 8.5, step=0.1)
            research = st.selectbox("Research Experience", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
 
        if st.button("Predict Admission"):
            try:
                input_dict = {
                    "GRE_Score": gre, "TOEFL_Score": toefl,
                    "SOP": sop, "LOR": lor, "CGPA": cgpa,
                    "University_Rating": str(uni_rating),
                    "Research": str(research),
                }
                input_df = pd.DataFrame([input_dict])
                input_enc = pd.get_dummies(input_df,
                                           columns=["University_Rating", "Research"],
                                           dtype=int)
 
                for col in ns["feature_names"]:
                    if col not in input_enc.columns:
                        input_enc[col] = 0
                input_enc = input_enc[ns["feature_names"]]
                input_scaled = ns["scaler"].transform(input_enc)
 
                pred = ns["model"].predict(input_scaled)[0]
                proba = ns["model"].predict_proba(input_scaled)[0]
 
                st.markdown("---")
                if pred == 1:
                    st.success(f"Admission Likely — Model Confidence: {proba[1]*100:.1f}%")
                else:
                    st.error(f"Admission Unlikely — Model Confidence: {proba[0]*100:.1f}%")
 
                st.caption(f"Admission threshold used during training: {ns['threshold']:.2f}")
 
            except Exception as e:
                st.error(f"Prediction error: {e}")
                logger.exception("Tab3 prediction error")
 
