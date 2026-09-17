import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure utils can be imported whether script is run from project root or app dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils import get_df, get_features, split_train_test, fit_model, evaluate, predict

st.set_page_config(page_title="Ligue 1 Match Analysis", layout="wide", page_icon="⚽")

st.title("⚽ Ligue 1 Match Analysis & Prediction")
st.markdown("""
This application analyzes Ligue 1 match statistics (Elo ratings, recent form, rest days) and trains an **XGBoost Classifier** using functions defined in `utils.py`.
""")

fields = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
data_csv = "data/Ligue1_2024-2025.csv"
if not os.path.exists(data_csv):
    # Fallback path if run inside app directory
    data_csv = "../data/Ligue1_2024-2025.csv"

@st.cache_data
def load_data(data_path, fields):
    df = pd.read_csv(data_path, usecols=fields)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.rename(columns={'FTHG': 'Home_goals', 'FTAG': 'Away_goals', 'FTR': 'Result'})
    return df

data = load_data(data_csv, fields)

# Section 1: Dataset Exploration
st.header("1. 📊 2024-2025 French Championship Matches")
data_dir = "data" if os.path.exists("data") else "../data"
df_2024 = get_df(data)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Match Dataset with Engineered Features")
    st.dataframe(df_2024, height=350)

with col2:
    st.subheader("Match Outcome Distribution (Home vs Away)")
    hist_values = df_2024['Result'].value_counts()
    st.bar_chart(hist_values)

X_all_teams = get_features(df_2024)

# Sidebar options for model training
st.sidebar.header("⚙️ Model Hyperparameters")
test_size = st.sidebar.slider("Test set size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.05, step=0.01)
n_estimators = st.sidebar.slider("N Estimators", min_value=50, max_value=1000, value=500, step=50)
max_depth = st.sidebar.slider("Max Depth", min_value=2, max_value=10, value=5, step=1)

# Section 3: Model Training and Evaluation
st.header("3. 🏋️ Model Training & Evaluation")

X_train, X_test, y_train, y_test = split_train_test(X_all_teams, test_size=test_size)

st.write(f"**Dataset split:** Training samples: `{len(X_train)}`, Test samples: `{len(X_test)}`")

train_clicked = st.button("🚀 Train & Evaluate XGBoost Model", type="primary")

if train_clicked or 'model' in st.session_state:
    if train_clicked:
        with st.spinner("Training model with XGBoost..."):
            model = fit_model(
                X_train, y_train,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                max_depth=max_depth
            )
            st.session_state['model'] = model
            st.session_state['params'] = (test_size, learning_rate, n_estimators, max_depth)
    else:
        model = st.session_state['model']

    st.success("Model trained successfully using `utils.fit_model`!")

    # Evaluation
    eval_res = evaluate(model, X_test, y_test)
    y_pred, cm, disp = predict(model, X_test, y_test)

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Log Loss", f"{eval_res['log_loss']:.4f}")
    with m2:
        st.metric("Accuracy Score", f"{eval_res['accuracy'] * 100:.2f}%")

    # Visualizations
    st.subheader("📈 Evaluation Visualizations")
    vcol1, vcol2 = st.columns(2)

    with vcol1:
        st.write("##### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        plt.title("Confusion Matrix (H: Home, D: Draw, A: Away)")
        st.pyplot(fig)

    with vcol2:
        st.write("##### Feature Importance")
        feature_names = X_train.columns
        importances = model.feature_importances_
        feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=True)
        
        fig_feat, ax_feat = plt.subplots(figsize=(5, 4))
        ax_feat.barh(feat_df["Feature"], feat_df["Importance"], color="skyblue")
        ax_feat.set_xlabel("Importance Score")
        ax_feat.set_title("XGBoost Feature Importances")
        plt.tight_layout()
        st.pyplot(fig_feat)

    # Predictions breakdown
    st.subheader("🔍 Sample Predictions on Test Set")
    res_map = {0: "H (Home Win)", 1: "D (Draw)", 2: "A (Away Win)"}
    pred_df = df_2024.loc[X_test.index, ['HomeTeam', 'AwayTeam']].rename(columns={'HomeTeam': 'Home_team', 'AwayTeam': 'Away_team'}).copy()
    for col in X_test.columns:
        pred_df[col] = X_test[col]
    pred_df["Actual Result"] = [res_map[val] for val in y_test.values]
    pred_df["Predicted Result"] = [res_map[val] for val in y_pred]
    st.dataframe(pred_df)

