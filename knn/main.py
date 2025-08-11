from data_processing import load_data, clean_data, get_features_and_labels, split_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = load_data('match_data_v5.csv') #replace with actual path of the dataset
    df = clean_data(df)

    X, y = get_features_and_labels(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Confusion Matrix
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
    plt.title("KNN Confusion Matrix")
    plt.show()