import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd
from keras import layers, models, optimizers, Sequential
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.preprocessing import StandardScaler

def load_data(path):
    names = [
        'blueTeamControlWardsPlaced', 'blueTeamWardsPlaced', 'blueTeamTotalKills',
        'blueTeamDragonKills', 'blueTeamHeraldKills', 'blueTeamTowersDestroyed',
        'blueTeamInhibitorsDestroyed', 'blueTeamTurretPlatesDestroyed', 'blueTeamFirstBlood',
        'blueTeamMinionsKilled', 'blueTeamJungleMinions', 'blueTeamTotalGold',
        'blueTeamXp', 'blueTeamTotalDamageToChamps', 'redTeamControlWardsPlaced',
        'redTeamWardsPlaced', 'redTeamTotalKills', 'redTeamDragonKills',
        'redTeamHeraldKills', 'redTeamTowersDestroyed', 'redTeamInhibitorsDestroyed',
        'redTeamTurretPlatesDestroyed', 'redTeamMinionsKilled', 'redTeamJungleMinions',
        'redTeamTotalGold', 'redTeamXp', 'redTeamTotalDamageToChamps', 'blueWin'
    ]
    df = pd.read_csv(path, names=names, header=None)
    return df

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def clean_data(df):
    cols_to_drop=[]
    df=df.drop(columns=cols_to_drop, errors='ignore')
    df.dropna()
    return df


def get_features_and_labels(df):
    X=df.drop(columns=['blueWin'])
    y=df['blueWin']
    return X,y

def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def create_model(input_shape):
    model=keras.Sequential([
        layers.Dense(64,activation='relu', input_shape=(input_shape,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model,X_train,y_train,X_test,y_test, epoches=20, batch_size=32):
    history=model.fit(X_train, y_train, epochs=epoches, batch_size=batch_size, validation_split=0.2)
    loss,accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    return history
    