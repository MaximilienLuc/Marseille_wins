import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import os


def get_df(df_2024):
    df_2024 = get_2024_elo(df_2024)
    df_2024 = features(df_2024)
    df_2024 = get_rest_days(df_2024)
    return df_2024


def expected_score(rating_A, rating_B):
    """Odds that A wins against B"""
    return 1 / (1 + 10 ** ((rating_B - rating_A) / 400))

def update_elo(rating_A, rating_B, score_A, K=30):
    """
    Update Elo ratings after a match
    score_A : 1 = A wins, 0.5 = draw, 0 = A loses
    We chose K=30 as of now based on common practice for this hyper_parameter in football 
    """
    
    expected_A = expected_score(rating_A, rating_B)
    expected_B = 1 - expected_A
    
    new_rating_A = round(rating_A + K * (score_A - expected_A), 0)
    new_rating_B = round(rating_B + K * ((1 - score_A) - expected_B), 0)
    
    return new_rating_A, new_rating_B


def get_2024_elo(df_2024):
    # Initialisation
    teams = pd.concat([df_2024['HomeTeam']]).unique()
    elo = {team: 1500 for team in teams}  # Initial score
    elo_A, elo_B = [], []

    # Iteration
    for i, row in df_2024.iterrows():
        team_A, team_B = row['HomeTeam'], row['AwayTeam']
        res = row['Result']

        # Results
        if res == 'H': score_A, score_B = 1, 0
        elif res == 'A': score_A, score_B = 0, 1
        else: score_A, score_B = 0.5, 0.5
        
        # Save before update
        elo_A.append(elo[team_A])
        elo_B.append(elo[team_B])
        
        # Update
        new_A, new_B = update_elo(elo[team_A], elo[team_B], score_A)
        elo[team_A], elo[team_B] = new_A, new_B

    df_2024['Elo_Home'] = elo_A
    df_2024['Elo_Away'] = elo_B
    df_2024['Diff_Elo'] = df_2024['Elo_Home'] - df_2024['Elo_Away']
    return df_2024


def features(df_2024): 
    # Compute current form of the teams

    # Initialisation
    teams = pd.concat([df_2024['HomeTeam']]).unique()
    perf_last_3_games = {team: 0 for team in teams} # Number of points won in the last 3 games
    perf_A, perf_B = [], []
    memory = {team: [0, 0, 0] for team in teams}

    # Iteration
    for i, row in df_2024.iterrows():
        team_A, team_B = row['HomeTeam'], row['AwayTeam']
        res = row['Result']

        # Results
        if res == 'H': score_A, score_B = 3, 0
        elif res == 'A': score_A, score_B = 0, 3
        else: score_A, score_B = 1, 1

        # Save before update
        perf_A.append(perf_last_3_games[team_A])
        perf_B.append(perf_last_3_games[team_B])
        
        # Update memory
        memory[team_A] = memory[team_A][1:]
        memory[team_A].append(score_A)
        memory[team_B] = memory[team_B][1:]
        memory[team_B].append(score_B)

        # Update perf
        perf_last_3_games[team_A] = sum(memory[team_A])
        perf_last_3_games[team_B] = sum(memory[team_B])

    df_2024['current_form_Home'] = perf_A
    df_2024['current_form_Away'] = perf_B

    return df_2024

def get_rest_days(df_2024, data_dir="data"):
    # adding the number of rest days for each team before game 

    teams = [
        "Angers", "Auxerre", "Brest", "Le Havre", "Lens", "Lille",
        "Lyon", "Marseille", "Monaco", "Montpellier", "Nantes", "Nice",
        "Paris SG", "Reims", "Rennes", "St Etienne", "Strasbourg", "Toulouse"
    ]

    frames = []

    for team in teams:
        path = os.path.join(data_dir, f"{team}_matches_with_rest_days_2024.csv")
        if os.path.exists(path):
            df_temp = pd.read_csv(path)
            frames.append(df_temp)

    if not frames:
        return df_2024

    df_rest_days_all_teams = pd.concat(frames)

    # reset the index to avoid duplicate indices after the concat
    df_rest_days_all_teams = df_rest_days_all_teams.reset_index(drop=True)

    # Create a new empty dataframe for matches
    match_df = pd.DataFrame(columns=['date', 'home_team', 'away_team', 'home_team_rest_days', 'away_team_rest_days'])

    # Group matches by date and process each match
    processed_matches = set() 

    for _, row in df_rest_days_all_teams.iterrows():
        date = row['date']
        team = row['team']
        opponent = row['opponent']
        venue = row['venue_team']
        rest_days = row['rest_days_team']
        
        match_id = tuple(sorted([team, opponent]) + [date])
        
        if match_id in processed_matches:
            continue
        
        opponent_rows = df_rest_days_all_teams[
            (df_rest_days_all_teams['date'] == date) & 
            (df_rest_days_all_teams['team'] == opponent) & 
            (df_rest_days_all_teams['opponent'] == team)
        ]
        
        if len(opponent_rows) > 0:
            opponent_row = opponent_rows.iloc[0]
            opponent_rest_days = opponent_row['rest_days_team']
            
            if venue == 'Home':
                home_team, away_team = team, opponent
                home_rest_days, away_rest_days = rest_days, opponent_rest_days
            else:
                home_team, away_team = opponent, team
                home_rest_days, away_rest_days = opponent_rest_days, rest_days
            
            match_df.loc[len(match_df)] = [date, home_team, away_team, home_rest_days, away_rest_days]
            processed_matches.add(match_id)

    match_df = match_df.rename(columns={"date": "Date", "home_team": "HomeTeam", "away_team": "AwayTeam"})
    match_df["Date"] = pd.to_datetime(match_df["Date"])
    df_2024["Date"] = pd.to_datetime(df_2024["Date"])

    df_2024 = pd.merge(df_2024, match_df, how="left", on=["Date", "HomeTeam", "AwayTeam"])

    return df_2024


def get_features(df_2024):
    X_all_teams = df_2024.drop(columns=['Date', 'HomeTeam', 'AwayTeam', 'Home_goals', 'Away_goals'])

    order = [["H", "D", "A"]]

    encoder = OrdinalEncoder(categories=order)
    X_all_teams["result_encoded"] = encoder.fit_transform(X_all_teams[["Result"]])
    X_all_teams.drop(columns='Result', inplace=True)
    return X_all_teams


def split_train_test(X_all_teams, test_size=0.2):
    X = X_all_teams[['Elo_Home', 'Elo_Away', 'Diff_Elo', 
            'current_form_Home', 'current_form_Away',
            'home_team_rest_days', 'away_team_rest_days']]
    y = X_all_teams["result_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    return X_train, X_test, y_train, y_test


def fit_model(X_train, y_train, learning_rate=0.05, n_estimators=500, max_depth=5):
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        seed=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    y_pred_prob = model.predict_proba(X_test)
    y_pred = y_pred_prob.argmax(axis=1)

    loss = log_loss(y_test, y_pred_prob)
    acc = accuracy_score(y_test, y_pred)

    return {
        "log_loss": loss,
        "accuracy": acc,
        "y_pred": y_pred,
        "y_pred_prob": y_pred_prob
    }


def predict(model, X_test, y_test=None):
    y_pred = model.predict(X_test)
    if y_test is not None:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["H", "D", "A"])
        return y_pred, cm, disp
    return y_pred


