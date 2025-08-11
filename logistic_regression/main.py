from data_processing import *
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

path = 'match_data_v5.csv'

def init_data(path):
    print("League of Legends Win Prediction Model")
    
    df = load_data(path)
    df = clean_data(df)

    print("Dataset Info")
    print(f"    Total samples: {len(df)}")
    print(f"    Total features: {len(df.columns) - 1}")

    feature_configs = {
        'all_features': None,
        'core_stats': [
            'blueTeamControlWardsPlaced', 'blueTeamWardsPlaced', 'blueTeamTotalKills', 'blueTeamDragonKills', 'blueTeamHeraldKills', 'blueTeamTowersDestroyed', 'blueTeamInhibitorsDestroyed', 'blueTeamTurretPlatesDestroyed', 'blueTeamFirstBlood', 'blueTeamTotalGold', 'blueTeamXp', 
            'redTeamControlWardsPlaced', 'redTeamWardsPlaced', 'redTeamTotalKills', 'redTeamDragonKills', 'redTeamHeraldKills', 'redTeamTowersDestroyed', 'redTeamInhibitorsDestroyed', 'redTeamTurretPlatesDestroyed', 'redTeamTotalGold', 'redTeamXp',
            ],
        'core_with_differences': [
            'blueTeamControlWardsPlaced', 'blueTeamWardsPlaced', 'blueTeamTotalKills', 'blueTeamDragonKills', 'blueTeamHeraldKills', 'blueTeamTowersDestroyed', 'blueTeamInhibitorsDestroyed', 'blueTeamTurretPlatesDestroyed', 'blueTeamFirstBlood', 'blueTeamTotalGold', 'blueTeamXp', 
            'redTeamControlWardsPlaced', 'redTeamWardsPlaced', 'redTeamTotalKills', 'redTeamDragonKills', 'redTeamHeraldKills', 'redTeamTowersDestroyed', 'redTeamInhibitorsDestroyed', 'redTeamTurretPlatesDestroyed', 'redTeamTotalGold', 'redTeamXp',
            'goldDifference', 'killDifference', 'towerDifference', 'xpDifference', 'dragonDifference', 'wardDifference'
            ]
        }

    lol_data = {}
    lol_labels = None

    for config, features in feature_configs.items():
        X, y = get_features_and_labels(df, features)
        lol_data[config] = X
        if lol_labels is None:
            lol_labels = y
        print(f"    {config}: {X.shape[0]} features")

    return lol_data, lol_labels, feature_configs

def analyze_data(path):
    lol_data, lol_labels, feature_configs = init_data(path)

    lambda_ranges = {
        'low': np.arange(0.0, 0.11, 0.01),
        'medium': np.arange(0.2, 2.1, 0.2),
        'high': np.arange(5, 51, 5)
    }

    lambdas = np.concatenate([lambda_ranges['low'],
                              lambda_ranges['medium'],
                              lambda_ranges['high']])
    
    results = {}
    
    for config, X in lol_data.items():
        print(f"Testing Feature Config: {config}")
        print(f"Features: {X.shape[0]}, Samples: {X.shape[1]}")

        config_results = []
        best_lambda = None
        best_rmse = float('inf')

        for lam in lambdas:
            rmse = xval_learning_alg(X, lol_labels, lam, 10)
            rmse_scalar = float(rmse.flatten()[0]) if hasattr(rmse, 'flatten') else float(rmse)
                
            config_results.append((lam, rmse_scalar))
                
            if rmse_scalar < best_rmse:
                best_rmse = rmse_scalar
                best_lambda = lam
                
            print(f"    Lambda={lam:6.3f}: Error Rate={rmse_scalar:.4f}")
                
        
        results[config] = {
            'best_lambda': best_lambda,
            'best_rmse': best_rmse,
            'all_results': config_results,
            'num_features': X.shape[0]
        }
    
        print(f"Best: Lambda={best_lambda}, Error Rate={best_rmse:.3f}")
    
    return results, feature_configs

def analyze_lambda_sensitivity(path, config_name='core_stats'):
    print(f"Lambda Sensitivity Analysis: {config_name}")

    lol_data, lol_labels, _ = init_data(path)
    X = lol_data[config_name]

    lambda_fine = np.concatenate([
        np.arange(0.0, 0.05, 0.005),
        np.arange(0.05, 0.5, 0.05),
        np.arange(0.5, 5.0, 0.5)
    ])
    
    detailed_results = []
    
    for lam in lambda_fine:
        try:
            rmse = xval_learning_alg(X, lol_labels, lam, 10)
            rmse_scalar = float(rmse.flatten()[0]) if hasattr(rmse, 'flatten') else float(rmse)
            detailed_results.append((lam, rmse_scalar))
            print(f"    Lambda={lam:7.3f}: Error Rate={rmse_scalar:.6f}")
            
        except Exception as e:
            print(f"    Lambda={lam:7.3f}: Error - {str(e)}")
            continue
    
    if detailed_results:
        best_lam, best_rmse = min(detailed_results, key=lambda x: x[1])
        print(f"Optimal: Lambda={best_lam:.3f}, Error Rate={best_rmse:.6f}")
        
        return detailed_results, best_lam, best_rmse
    
    return [], None, None

def evaluate_final_model(path, config_name, best_lambda, feature_configs):
    print("Final Model Evaluation")

    features = None if config_name == 'all_features' else feature_configs[config_name]

    X_test, y_test, X_train, y_train = split_data(
        path, 
        features=features,
        test_size=0.2
    )

    print(f"Training samples: {X_train.shape[1]}")
    print(f"Test samples: {X_test.shape[1]}")
    print(f"Features: {X_train.shape[0]}")

    th, th0 = logistic_min(X_train, y_train, best_lambda)

    train_loss = float(mean_logistic_loss(X_train, y_train, th, th0).item())
    test_loss = float(mean_logistic_loss(X_test, y_test, th, th0).item())

    print(f"Train Loss: {train_loss:.3f}")
    print(f"Test Loss:  {test_loss:.3f}")

    train_predictions = logistic_predict(X_train, th, th0)
    test_predictions = logistic_predict(X_test, th, th0)

    y_test_flat = np.array(y_test).flatten()
    test_pred_flat = np.array(test_predictions).flatten()

    test_pred_binary = (test_pred_flat > 0.5).astype(int)

    train_accuracy = np.mean((train_predictions > 0.5) == (y_train > 0.5))
    test_accuracy = np.mean((test_predictions > 0.5) == (y_test > 0.5))

    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy:  {test_accuracy:.4f}")

    print(classification_report(y_test_flat, test_pred_binary))
    cm = confusion_matrix(y_test_flat, test_pred_binary)
    print("Confusion Matrix:\n", cm)

    print(f"Best configuration achieves {test_accuracy:.1%} accuracy on test set")

    # Plot confusion matrix heatmap
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Red Win', 'Blue Win'],
                yticklabels=['Red Win', 'Blue Win'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Logistic Regression Confusion Matrix")
    plt.show()
    
def print_summary(results):
    print("Summary:")

    overall_best_rmse = float('inf')
    overall_best_config = None
    
    print(f"{'Configuration':<20} {'Features':<10} {'Best Lambda':<12} {'Best Error Rate':<15}")
    
    for config_name, result in results.items():
        best_lambda = result['best_lambda']
        best_rmse = result['best_rmse']
        num_features = result['num_features']
        
        print(f"{config_name:<20} {num_features:<10} {best_lambda:<12.3f} {best_rmse:<12.4f}")
        
        if best_rmse < overall_best_rmse:
            overall_best_rmse = best_rmse
            overall_best_config = config_name
    
    print("Best Overall Configuration:")
    if overall_best_config:
        best_result = results[overall_best_config]
        print(f"    Configuration: {overall_best_config}")
        print(f"    Features: {best_result['num_features']}")
        print(f"    Best Lambda: {best_result['best_lambda']:.3f}")
        print(f"    Best Error Rate: {best_result['best_rmse']:.4f}")
        
        estimated_accuracy = 1 - best_result['best_rmse']
        print(f"    Estimated Accuracy: {estimated_accuracy:.4f}")
    
    return overall_best_config

if __name__ == "__main__":
    print("Starting Comprehensive Analysis")
    results, feature_configs = analyze_data(path)

    best_config = print_summary(results)
        
    if best_config:
        detailed_results, optimal_lambda, optimal_rmse = analyze_lambda_sensitivity(path, best_config)
        final_results = evaluate_final_model(path, best_config, optimal_lambda, feature_configs)
