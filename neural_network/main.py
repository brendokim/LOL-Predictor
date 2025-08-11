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


csv_path   = "match_data_v5.csv"   
has_header = False                


SELECTED_FEATURES = [
    'blueTeamTotalKills', 'blueTeamTotalGold', 'blueTeamTowersDestroyed',
    'blueTeamDragonKills', 'blueTeamHeraldKills', 'blueTeamFirstBlood',
    'redTeamTotalKills', 'redTeamTotalGold', 'redTeamTowersDestroyed',
    'redTeamDragonKills', 'redTeamHeraldKills'
]
TARGET_COL = 'blueWin'

test_size = 0.2
epochs    = 30
batch_size= 64
seed=42



ALL_COLUMNS = [
    'blueTeamControlWardsPlaced', 'blueTeamWardsPlaced', 'blueTeamTotalKills',
    'blueTeamDragonKills', 'blueTeamHeraldKills', 'blueTeamTowersDestroyed',
    'blueTeamInhibitorsDestroyed', 'blueTeamTurretPlatesDestroyed', 'blueTeamFirstBlood',
    'blueTeamMinionsKilled', 'blueTeamJungleMinions', 'blueTeamTotalGold',
    'blueTeamXp', 'blueTeamTotalDamageToChamps', 'redTeamControlWardsPlaced',
    'redTeamWardsPlaced', 'redTeamTotalKills', 'redTeamDragonKills',
    'redTeamHeraldKills', 'redTeamTowersDestroyed', 'redTeamInhibitorsDestroyed',
    'redTeamTurretPlatesDestroyed', 'redTeamMinionsKilled', 'redTeamJungleMinions',
    'redTeamTotalGold', 'redTeamXp', 'redTeamTotalDamageToChamps', 'blueWin', 'extra'
]

def load_data(path, has_header=False):
    if has_header:
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, names=ALL_COLUMNS, header=None)
    print(f"Loaded data: {df.shape}")
    return df

def clean_data(df):
    
    df = df.dropna()
    df = df.drop_duplicates()
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}'")
    print("Target distribution:\n", df[TARGET_COL].value_counts())
    return df


def select_features(df, selected):
   
    drop_cols = [TARGET_COL, 'extra']
    feature_pool = [c for c in df.columns if c not in drop_cols]

    if selected == "ALL":
        features = feature_pool
    else:
        missing = [c for c in selected if c not in df.columns]
        if missing:
            raise ValueError(f"Selected features not found in DataFrame")
        features = selected

    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET_COL]

    keep = ~X.isna().any(axis=1)
    if keep.sum() != len(X):
   
        X = X.loc[keep]
        y = y.loc[keep]


    return X, y, features


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler

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

    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def main():
    np.random.seed(seed)
    tf.random.set_seed(seed)

    df = load_data(csv_path, has_header=has_header)
    df = clean_data(df)
    X, y, used_features = select_features(df, SELECTED_FEATURES)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    Xtr_s, Xte_s, scaler = scale_features(Xtr, Xte)

    model = create_model(Xtr_s.shape[1])
    train_and_eval(model, Xtr_s, ytr.values, Xte_s, yte.values)

    
    model.save("lol_win_model.keras")
    joblib.dump(scaler, "scaler.pkl")
    with open("features.json", "w") as f:
        json.dump({"features": used_features, "target": TARGET_COL}, f, indent=2)

 

if __name__ == "__main__":
    main()
