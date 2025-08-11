import json
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

SELECTED_FEATURES = [
    'blueTeamTotalKills', 'blueTeamTotalGold', 'blueTeamTowersDestroyed',
    'blueTeamDragonKills', 'blueTeamHeraldKills', 'blueTeamFirstBlood',
    'redTeamTotalKills', 'redTeamTotalGold', 'redTeamTowersDestroyed',
    'redTeamDragonKills', 'redTeamHeraldKills'
]

def load_normalization_stats():
    with open("normalization_stats.json", "r") as f:
        stats = json.load(f)
    return np.array(stats["mean"]), np.array(stats["std"])

def get_user_input():
    print("\nEnter match stats for the following features for the first 15 minuites")
    values = {}

    base_features = [
        'blueTeamTotalKills', 'blueTeamTotalGold', 'blueTeamTowersDestroyed',
        'blueTeamDragonKills', 'blueTeamHeraldKills', 'blueTeamFirstBlood',
        'redTeamTotalKills', 'redTeamTotalGold', 'redTeamTowersDestroyed',
        'redTeamDragonKills', 'redTeamHeraldKills'
    ]

    for feature in base_features:
        while True:
            try:
                val = float(input(f"{feature}: "))
                values[feature] = val
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
    return pd.DataFrame([values])

def engineer_features(df):
    df = df.copy()
    
    df['goldDifference'] = df['blueTeamTotalGold'] - df['redTeamTotalGold']
    df['killDifference'] = df['blueTeamTotalKills'] - df['redTeamTotalKills']
    df['towerDifference'] = df['blueTeamTowersDestroyed'] - df['redTeamTowersDestroyed']
    df['dragonDifference'] = df['blueTeamDragonKills'] - df['redTeamDragonKills']
    
    all_features = SELECTED_FEATURES + [
        'goldDifference', 'killDifference', 'towerDifference', 'dragonDifference'
    ]
    
    return df[all_features]

def standardize_features(X, mean, std):
    X_array = X.values.astype(float)
    return (X_array - mean) / std

def predict(X_processed, model):
    prob = model.predict(X_processed, verbose=0)[0][0]
    pred = int(prob > 0.5)
    return prob, pred

def main():
    print(" League of Legends Win Predictor")
    
    model = load_model("lol_win_model.keras") 
    train_mean, train_std = load_normalization_stats()

    # Get user input
    user_df = get_user_input()

    processed_df = engineer_features(user_df)
    X_processed = standardize_features(processed_df, train_mean, train_std)

    # Predict
    prob, pred = predict(X_processed, model)

    print(f"\n Probability Blue Wins: {prob:.2f}")
    print("🔵 Predicted Winner:" if pred else "🔴 Predicted Winner:", "Blue Team" if pred else "Red Team")

if __name__ == "__main__":
    main()
