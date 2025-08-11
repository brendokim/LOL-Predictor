import numpy as np 
import pandas as pd

def load_data(path):
    df = pd.read_csv(path
                , names=['blueTeamControlWardsPlaced', 'blueTeamWardsPlaced', 'blueTeamTotalKills', 'blueTeamDragonKills', 'blueTeamHeraldKills',
                        'blueTeamTowersDestroyed', 'blueTeamInhibitorsDestroyed', 'blueTeamTurretPlatesDestroyed', 'blueTeamFirstBlood', 'blueTeamMinionsKilled',
                        'blueTeamJungleMinions', 'blueTeamTotalGold', 'blueTeamXp', 'blueTeamTotalDamageToChamps', 'redTeamControlWardsPlaced', 'redTeamWardsPlaced',
                        'redTeamTotalKills', 'redTeamDragonKills', 'redTeamHeraldKills', 'redTeamTowersDestroyed', 'redTeamInhibitorsDestroyed', 'redTeamTurretPlatesDestroyed',
                        'redTeamMinionsKilled', 'redTeamJungleMinions', 'redTeamTotalGold', 'redTeamXp', 'redTeamTotalDamageToChamps', 'blueWin', 'extra'], header=0)
    return df

def clean_data(df):
    if 'extra' in df.columns:
        df = df.drop('extra', axis=1)
    return df

def get_features_and_labels(df, features=None):
    target_col = 'blueWin'

    df = df.copy()
    
    df['goldDifference'] = df['blueTeamTotalGold'] - df['redTeamTotalGold']
    df['killDifference'] = df['blueTeamTotalKills'] - df['redTeamTotalKills']
    df['towerDifference'] = df['blueTeamTowersDestroyed'] - df['redTeamTowersDestroyed']
    df['xpDifference'] = df['blueTeamXp'] - df['redTeamXp']
    df['dragonDifference'] = df['blueTeamDragonKills'] - df['redTeamDragonKills']
    df['wardDifference'] = df['blueTeamWardsPlaced'] - df['redTeamWardsPlaced']

    if features is None:
        features = [col for col in df.columns if col != target_col]
    
    X_df = df[features]
    y_df = df[target_col]

    X = X_df.values.T
    y = y_df.values.reshape(1, -1)

    X_mean = np.mean(X, axis=1, keepdims=True)
    X_std = np.std(X, axis=1, keepdims=True)
    X_std[X_std == 0] = 1
    X = (X - X_mean) / X_std

    return X, y

def split_data(path, features=None, test_size=0.2):
    df = load_data(path)
    df = clean_data(df)
    X, y = get_features_and_labels(df, features)

    n_samples = X.shape[1]
    n_test = int(n_samples * test_size)

    np.random.seed(0)
    indices = np.random.permutation(n_samples)

    test_indices = indices[:n_test]
    X_test = X[:, test_indices]
    y_test = y[:, test_indices]

    train_indices = indices[n_test:]
    X_train = X[:, train_indices]
    y_train = y[:, train_indices]

    return X_test, y_test, X_train, y_train

def logistic_loss(x, y, th, th0):
    predictions = logistic_predict(x, th, th0)
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

def d_logistic_loss_th(x, y, th, th0):
    predictions = logistic_predict(x, th, th0)
    return -(y - predictions) * x

def d_mean_logistic_loss_th(x, y, th, th0):
    return np.mean(d_logistic_loss_th(x, y, th, th0), axis=1, keepdims=True)

def d_logistic_loss_th0(x, y, th, th0):
    predictions = logistic_predict(x, th, th0)
    return -(y - predictions)

def d_mean_logistic_loss_th0(x, y, th, th0):
    return np.mean(d_logistic_loss_th0(x, y, th, th0), axis=1, keepdims=True)

def d_logistic_obj_th(x, y, th, th0, lam):
    return d_mean_logistic_loss_th(x, y, th, th0) + 2 * lam * th

def d_logistic_obj_th0(x, y, th, th0, lam):
    return d_mean_logistic_loss_th0(x, y, th, th0)

def logistic_obj_grad(x, y, th, th0, lam):
    grad_th = d_logistic_obj_th(x, y, th, th0, lam)
    grad_th0 = d_logistic_obj_th0(x, y, th, th0, lam)
    return np.vstack([grad_th, grad_th0])

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def logistic_predict(x, th, th0):
    return sigmoid(np.dot(th.T, x) + th0)

def logistic_loss(x, y, th, th0):
    predictions = logistic_predict(x, th, th0)
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

def mean_logistic_loss(x, y, th, th0):
    return np.mean(logistic_loss(x, y, th, th0), axis=1, keepdims = True)

def logistic_obj(x, y, th, th0, lam):
    return mean_logistic_loss(x, y, th, th0) + lam * np.linalg.norm(th) ** 2

def sgd(X, y, J, dJ, w0, step_size_fn, max_iter):
    w = w0.copy()
    fs = []
    ws = []
    n = X.shape[1]

    for i in range(max_iter):
        j = np.random.randint(n)
        Xi = X[:, j:j+1]
        yi = y[:, j:j+1]

        cost = J(Xi, yi, w)
        fs.append(cost)

        gradient = dJ(Xi, yi, w)
        step = step_size_fn(i)
        w = w - step * gradient
        ws.append(w.copy())
    
    return w, fs, ws

def logistic_min(X, y, lam):
    def svm_min_step_size_fn(i):
        return 0.01/(i+1)**0.5

    d, n = X.shape
    X_extend = np.vstack([X, np.ones((1, n))])
    w_init = np.zeros((d+1, 1))

    def J(Xj, yj, th):
        return float(logistic_obj(Xj[:-1,:], yj, th[:-1,:], th[-1:,:], lam))

    def dJ(Xj, yj, th):
        return logistic_obj_grad(Xj[:-1,:], yj, th[:-1,:], th[-1:,:], lam)
    
    np.random.seed(0)
    w, fs, ws = sgd(X_extend, y, J, dJ, w_init, svm_min_step_size_fn, 1000)
    return w[:-1,:], w[-1:,:]

def eval_classifier(X_train, Y_train, X_test, Y_test, lam):
    th, th0 = logistic_min(X_train, Y_train, lam)
    predictions = logistic_predict(X_test, th, th0)
    y_pred_binary = (predictions > 0.5).astype(int)
    accuracy = np.mean(y_pred_binary == Y_test)
    return 1 - accuracy

def xval_learning_alg(X, y, lam, k):
    _, n = X.shape
    idx = list(range(n))
    np.random.seed(0)
    np.random.shuffle(idx)
    X, y = X[:,idx], y[:,idx]

    split_X = np.array_split(X, k, axis=1)
    split_y = np.array_split(y, k, axis=1)

    score_sum = 0
    for i in range(k):
        X_train = np.concatenate(split_X[:i] + split_X[i+1:], axis=1)
        y_train = np.concatenate(split_y[:i] + split_y[i+1:], axis=1)
        X_test = np.array(split_X[i])
        y_test = np.array(split_y[i])
        score_sum += eval_classifier(X_train, y_train, X_test, y_test, lam)
    return score_sum/k