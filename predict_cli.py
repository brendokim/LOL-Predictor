import json
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

def load_metadata():
    with open("features.json", "r") as f:
        metadata = json.load(f)
    return metadata["features"], metadata["target"]

def get_user_input(features):
    print("\nEnter match stats for the following features for the first 15 minuites")
    values = {}
    for feature in features:
        while True:
            try:
                val = float(input(f"{feature}: "))
                values[feature] = val
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
    return pd.DataFrame([values])

def predict(df_input, model, scaler):
    X_scaled = scaler.transform(df_input)
    prob = model.predict(X_scaled)[0][0]
    pred = int(prob > 0.5)
    return prob, pred

def main():
    print(" League of Legends Win Predictor")


    model = load_model("lol_win_model.keras")
    scaler = joblib.load("scaler.pkl")
    features, target = load_metadata()

    # Get user input
    user_df = get_user_input(features)

    # Predict
    prob, pred = predict(user_df, model, scaler)

    print(f"\n Probability Blue Wins: {prob:.2f}")
    print("🔵 Predicted Winner:" if pred else "🔴 Predicted Winner:", "Blue Team" if pred else "Red Team")

if __name__ == "__main__":
    main()
