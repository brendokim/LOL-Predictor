from data_processing import load_data, clean_data, get_features_and_labels, split_data #change if adding/removing new functions
from sklearn.neighbors import KNeighborsClassifier #change based on what model you use
from sklearn.metrics import accuracy_score
from data_processing import load_data, clean_data, get_features_and_labels, split_data, create_model, train_model
from keras.models import Sequential

if __name__ == "__main__":
    def main():
        df = load_data("match_data_v5.csv")
        df = clean_data(df)
        X, y = get_features_and_labels(df)
        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
        model=create_model(X_train.shape[1])
        train_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
        
    main()