import os
import subprocess

models = {
    "1": ("Linear Regression", "linear_regression/main.py"),
    "2": ("Logistic Regression", "logistic_regression/main.py"),
    "3": ("K-Nearest Neighbours", "knn/main.py"),
    "4": ("Neural Network", "neural_network/main.py"),
    "5": ("Neural Network Predictor", "neural_network/predict_cli.py")
}

def main():
    print("Select a model to run:")
    for key, (name, _) in models.items():
        print(f"{key}. {name}")

    choice = input("Enter your choice (1-5): ").strip()

    if choice in models:
        model_name, script_path = models[choice]
        print(f"\nRunning {model_name}...\n")
        subprocess.run(["python", script_path])
    else:
        print("Invalid choice, please enter a value from 1-5.")

if __name__ == "__main__":
    main()