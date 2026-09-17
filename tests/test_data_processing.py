import pytest
import pandas as pd
import numpy as np
import os
import sys

# Ensure app package is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))

from utils import (
    expected_score,
    update_elo,
    get_2024_elo,
    features,
    get_rest_days,
    get_df,
    get_features
)


@pytest.fixture
def sample_raw_data():
    """Provides sample Ligue 1 match data for testing."""
    return pd.DataFrame({
        "Date": pd.to_datetime(["2024-08-16", "2024-08-17", "2024-08-18", "2024-08-24", "2024-08-25"]),
        "HomeTeam": ["Marseille", "Brest", "PSG", "Lyon", "Marseille"],
        "AwayTeam": ["Lyon", "PSG", "Marseille", "Brest", "PSG"],
        "Home_goals": [3, 0, 1, 2, 2],
        "Away_goals": [1, 2, 1, 1, 0],
        "Result": ["H", "A", "D", "H", "H"]
    })


def test_expected_score():
    """Test Elo expected score probabilities."""
    # Equal rating should yield 50% chance
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    
    # Higher rating team should have higher expected score
    prob_A = expected_score(1600, 1400)
    prob_B = expected_score(1400, 1600)
    assert prob_A > 0.5
    assert prob_B < 0.5
    assert prob_A + prob_B == pytest.approx(1.0)


def test_update_elo():
    """Test Elo rating update after match result."""
    # Home win
    new_A, new_B = update_elo(1500, 1500, score_A=1, K=30)
    assert new_A > 1500
    assert new_B < 1500
    assert (new_A - 1500) == pytest.approx(1500 - new_B)

    # Draw
    draw_A, draw_B = update_elo(1500, 1500, score_A=0.5, K=30)
    assert draw_A == 1500
    assert draw_B == 1500


def test_get_2024_elo(sample_raw_data):
    """Test Elo calculations across multiple matches."""
    df_elo = get_2024_elo(sample_raw_data.copy())
    
    # Check that required columns exist
    assert "Elo_Home" in df_elo.columns
    assert "Elo_Away" in df_elo.columns
    assert "Diff_Elo" in df_elo.columns
    
    # First match starting Elos should be 1500
    assert df_elo.iloc[0]["Elo_Home"] == 1500
    assert df_elo.iloc[0]["Elo_Away"] == 1500
    assert df_elo.iloc[0]["Diff_Elo"] == 0
    
    # Check that diff is computed correctly
    for idx, row in df_elo.iterrows():
        assert row["Diff_Elo"] == row["Elo_Home"] - row["Elo_Away"]


def test_features_current_form(sample_raw_data):
    """Test current form calculation (points won in last 3 matches)."""
    df_form = features(sample_raw_data.copy())
    
    assert "current_form_Home" in df_form.columns
    assert "current_form_Away" in df_form.columns
    
    # First matches should have form = 0
    assert df_form.iloc[0]["current_form_Home"] == 0
    assert df_form.iloc[0]["current_form_Away"] == 0
    
    # Marseille won match 0 (3 points), drew match 2 (1 point)
    # Marseille's match 4 should reflect previous form
    m_match4 = df_form[(df_form["HomeTeam"] == "Marseille") & (df_form.index == 4)]
    assert m_match4.iloc[0]["current_form_Home"] == 4  # 3 + 1


def test_get_rest_days(sample_raw_data, tmp_path):
    """Test rest days integration with mock team files."""
    # Create temp data directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create sample rest day files for teams
    teams = ["Marseille", "Lyon", "PSG", "Brest"]
    for team in teams:
        csv_file = data_dir / f"{team}_matches_with_rest_days_2024.csv"
        df_team = pd.DataFrame({
            "date": ["2024-08-16", "2024-08-17"],
            "team": [team, team],
            "opponent": ["Lyon" if team == "Marseille" else "Marseille", "Brest"],
            "venue_team": ["Home", "Away"],
            "rest_days_team": [7, 6]
        })
        df_team.to_csv(csv_file, index=False)
        
    df_res = get_rest_days(sample_raw_data.copy(), data_dir=str(data_dir))
    
    # Check output dataframe structure
    assert isinstance(df_res, pd.DataFrame)
    assert len(df_res) == len(sample_raw_data)


def test_get_df(sample_raw_data):
    """Integration test for get_df function."""
    df_processed = get_df(sample_raw_data.copy())
    
    # Check that all features are added
    expected_cols = ["Elo_Home", "Elo_Away", "Diff_Elo", "current_form_Home", "current_form_Away"]
    for col in expected_cols:
        assert col in df_processed.columns


def test_get_features_encoding(sample_raw_data):
    """Test feature extraction and result encoding."""
    df_processed = get_df(sample_raw_data.copy())
    feat_df = get_features(df_processed)
    
    # Verify dropped non-feature columns
    forbidden_cols = ["Date", "HomeTeam", "AwayTeam", "Home_goals", "Away_goals", "Result"]
    for col in forbidden_cols:
        assert col not in feat_df.columns
        
    # Verify result_encoded exists and values match H=0, D=1, A=2
    assert "result_encoded" in feat_df.columns
    unique_vals = set(feat_df["result_encoded"].unique())
    assert unique_vals.issubset({0.0, 1.0, 2.0})
