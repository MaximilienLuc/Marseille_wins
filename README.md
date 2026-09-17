# ⚽ Marseille Wins - Match Predictor & Ligue 1 Analysis

We (Oliver Smith and Maximilien Lucille) are Marseille fans. We've spent too many evenings watching our team lose in the past years. This project will allow us to determine whether or not the next game played by Marseille is worth the watch depending on how likely we are to win.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/Marseille_wins.git
   cd Marseille_wins
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Streamlit Application

Start the Streamlit web dashboard:
```bash
streamlit run app/streamlit.py
```

---

## 🧪 Testing & Code Coverage

Run the unit test suite with coverage reporting:
```bash
pytest tests/ --cov=app --cov-report=term-missing
```

### Test Coverage Highlights:
- **`tests/test_data_processing.py`**: Tests data loading, Elo formulas (`expected_score`, `update_elo`), rolling form calculation (`features`), rest days integration (`get_rest_days`), and result ordinal encoding (`get_features`).
- **`tests/test_model_pipeline.py`**: Tests non-shuffled time-series train/test splitting (`split_train_test`), model fitting (`fit_model`), metric evaluation (`evaluate`), and deterministic predictions (`predict`).

---

## 🔁 Reproducibility & Best Practices

To ensure reliable, deterministic, and scientific experiment results:
- **Fixed Random Seeds**: XGBoost model training and data splitting utilize a fixed seed (`seed=42`).
- **Data Leakage Prevention**: Split functions enforce `shuffle=False` for match time-series data to avoid future match information leaking into training sets.
- **Dependency Locking**: Environment requirements are pinned in `requirements.txt`.

---

## ⚙️ Continuous Integration (CI)

This repository uses **GitHub Actions** (`.github/workflows/ci.yml`) to automatically:
1. Run `pytest` and compute code coverage on every push & pull request.
2. Validate compatibility across Python `3.10`, `3.11`, and `3.12`.
3. Check syntax compilation for `app/streamlit.py` and `app/utils.py`.
