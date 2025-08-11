
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(
        path,
        names=[
            'blueTeamControlWardsPlaced', 'blueTeamWardsPlaced', 'blueTeamTotalKills',
            'blueTeamDragonKills', 'blueTeamHeraldKills', 'blueTeamTowersDestroyed',
            'blueTeamInhibitorsDestroyed', 'blueTeamTurretPlatesDestroyed', 'blueTeamFirstBlood',
            'blueTeamMinionsKilled', 'blueTeamJungleMinions', 'blueTeamTotalGold',
            'blueTeamXp', 'blueTeamTotalDamageToChamps', 'redTeamControlWardsPlaced',
            'redTeamWardsPlaced', 'redTeamTotalKills', 'redTeamDragonKills',
            'redTeamHeraldKills', 'redTeamTowersDestroyed', 'redTeamInhibitorsDestroyed',
            'redTeamTurretPlatesDestroyed', 'redTeamMinionsKilled', 'redTeamJungleMinions',
            'redTeamTotalGold', 'redTeamXp', 'redTeamTotalDamageToChamps',
            'blueWin', 'extra'
        ],
        header=0
    )
    return df

def clean_data(df):
    df = df.drop(columns=["extra"], errors="ignore")  # remove extra column
    df = df.dropna()  # drop missing values if any
    return df

def get_features_and_labels(df):
    X = df.drop(columns=["blueWin"])
    y = df["blueWin"]
    return X, y

def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)
