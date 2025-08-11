from data_processing import load_data, clean_data, get_features_and_labels, split_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

if __name__ == "__main__":
    df = load_data(r"C:\Users\Admin\Downloads\archive\match_data_v5.csv") #replace with actual path of the dataset
    df = clean_data(df)

    X, y = get_features_and_labels(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n")
    print("             Predicted")
    print("              Red Win   Blue Win")
    print(f"Actual Red Win    {cm[0][0]:<8}  {cm[0][1]}")
    print(f"       Blue Win   {cm[1][0]:<8}  {cm[1][1]}")
    print()

    # Metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print("Metrics:")
    print(f" Precision: {precision:.4f} (Blue wins predicted correctly)")
    print(f" Recall:    {recall:.4f} (Actual blue wins caught)")
    print(f" F1-Score:  {f1:.4f}")
    print()
    print(f"Best configuration achieves {accuracy * 100:.1f}% accuracy on test set")
