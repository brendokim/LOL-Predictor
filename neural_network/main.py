import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import json
import seaborn as sns
import matplotlib.pyplot as plt

csv_path = "match_data_v5.csv"   
has_header = True                

test_size = 0.2
epochs = 30
batch_size = 64
seed = 42

SELECTED_FEATURES = [
    'blueTeamTotalKills', 'blueTeamTotalGold', 'blueTeamTowersDestroyed',
    'blueTeamDragonKills', 'blueTeamHeraldKills', 'blueTeamFirstBlood',
    'redTeamTotalKills', 'redTeamTotalGold', 'redTeamTowersDestroyed',
    'redTeamDragonKills', 'redTeamHeraldKills'
]

def load_data(path):
    df = pd.read_csv(path, names=[
        'blueTeamControlWardsPlaced', 'blueTeamWardsPlaced', 'blueTeamTotalKills', 'blueTeamDragonKills', 'blueTeamHeraldKills',
        'blueTeamTowersDestroyed', 'blueTeamInhibitorsDestroyed', 'blueTeamTurretPlatesDestroyed', 'blueTeamFirstBlood', 'blueTeamMinionsKilled',
        'blueTeamJungleMinions', 'blueTeamTotalGold', 'blueTeamXp', 'blueTeamTotalDamageToChamps',
        'redTeamControlWardsPlaced', 'redTeamWardsPlaced', 'redTeamTotalKills', 'redTeamDragonKills', 'redTeamHeraldKills', 'redTeamTowersDestroyed',
        'redTeamInhibitorsDestroyed', 'redTeamTurretPlatesDestroyed', 'redTeamMinionsKilled', 'redTeamJungleMinions', 'redTeamTotalGold', 'redTeamXp',
        'redTeamTotalDamageToChamps', 'blueWin', 'extra'
    ], header=0)
    return df

def clean_data(df):
    if 'extra' in df.columns:
        df = df.drop('extra', axis=1)
    return df.dropna().drop_duplicates()

def get_features_and_labels(df):
    target_col = 'blueWin'
    df = df.copy()

    # Engineered features
    df['goldDifference'] = df['blueTeamTotalGold'] - df['redTeamTotalGold']
    df['killDifference'] = df['blueTeamTotalKills'] - df['redTeamTotalKills']
    df['towerDifference'] = df['blueTeamTowersDestroyed'] - df['redTeamTowersDestroyed']
    df['dragonDifference'] = df['blueTeamDragonKills'] - df['redTeamDragonKills']

    # Combine raw + engineered features
    features = SELECTED_FEATURES + [
        'goldDifference', 'killDifference', 'towerDifference', 'dragonDifference'
    ]

    X_df = df[features]
    y_df = df[target_col]

    # Standardize
    X = X_df.values.astype(float)
    X_mean = np.mean(X, axis=0, keepdims=True)
    X_std = np.std(X, axis=0, keepdims=True)
    X_std[X_std == 0] = 1
    X = (X - X_mean) / X_std

    y = y_df.values.astype(int)
    return X, y, features, X_mean, X_std

def create_model(input_dim: int):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_eval(model, X_train, y_train, X_test, y_test):
    cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_split=0.2, callbacks=[cb], verbose=1)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {loss:.4f}  |  Test Acc: {acc:.4f}")

    y_prob = model.predict(X_test, verbose=0)
    y_pred = (y_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    print(f"Best configuration achieves {accuracy:.1%} accuracy on test set")

    # Plot confusion matrix heatmap
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Red Win', 'Blue Win'],
                yticklabels=['Red Win', 'Blue Win'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Neural Network Confusion Matrix")
    plt.show()


def main():
    np.random.seed(seed)
    tf.random.set_seed(seed)

    df = load_data(csv_path)
    df = clean_data(df)
    X, y, used_features, X_mean, X_std = get_features_and_labels(df)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)

    model = create_model(Xtr.shape[1])
    train_and_eval(model, Xtr, ytr, Xte, yte)

    model.save("lol_win_model.keras")

    normalization_stats = {
        "mean": X_mean.flatten().tolist(),
        "std": X_std.flatten().tolist()
    }
    with open("normalization_stats.json", "w") as f:
        json.dump(normalization_stats, f, indent=2)

if __name__ == "__main__":
    main()
