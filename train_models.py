"""
train_models.py

This script trains machine learning models to classify orbital behavior
in the Circular Restricted Three-Body Problem (CR3BP).

The dataset is generated from numerical simulations and contains initial
conditions and physics-based features. The goal is to classify each orbit as:

0 = Stable / bounded
1 = Escape
2 = Collision
"""

import os
import joblib

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def load_data(path="data/orbits.csv"):
    """
    Loads the generated CR3BP dataset from a CSV file.

    Parameters:
        path (str): Path to the dataset CSV file.

    Returns:
        X (DataFrame): Feature columns used for prediction.
        y (Series): Target labels for orbit classification.
    """

    # Read the generated orbit dataset
    df = pd.read_csv(path)

    # Features are all columns except the label
    X = df.drop(columns=["label"])

    # Target is the orbit outcome label
    y = df["label"]

    return X, y


def train_models(X_train, y_train):
    """
    Trains multiple machine learning models on the training data.

    Parameters:
        X_train: Scaled training feature data.
        y_train: Training labels.

    Returns:
        trained_models (dict): Dictionary containing trained model objects.
    """

    # Define the models that will be compared
    models = {
        # Baseline linear model
        "logistic_regression": LogisticRegression(max_iter=1000),

        # Nonlinear tree-based model
        "random_forest": RandomForestClassifier(n_estimators=100),

        # Neural network classifier
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 64),
            max_iter=500,
            random_state=42
        )
    }

    trained_models = {}

    # Train each model and store it
    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        trained_models[name] = model

    return trained_models


def evaluate_models(models, X_test, y_test):
    """
    Evaluates each trained model using test data.

    The evaluation includes:
    - Precision
    - Recall
    - F1-score
    - Confusion matrix

    Parameters:
        models (dict): Dictionary of trained models.
        X_test: Scaled test feature data.
        y_test: True labels for the test set.
    """

    # Create results folder if it does not already exist
    os.makedirs("results", exist_ok=True)

    # Evaluate each model separately
    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        # Predict labels for unseen test data
        y_pred = model.predict(X_test)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))


def main():
    """
    Main workflow for training and evaluating models.

    Steps:
    1. Load generated dataset
    2. Split into training and testing sets
    3. Scale numerical features
    4. Train ML models
    5. Evaluate model performance
    6. Save trained models and scaler
    """

    print("Loading dataset...")
    X, y = load_data()

    print("Splitting data...")

    # Split data so the model is tested on examples it has never seen
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Scaling features...")

    # StandardScaler makes features have mean 0 and standard deviation 1
    # This helps Logistic Regression and MLP train more effectively
    scaler = StandardScaler()

    # Fit scaler only on training data to avoid data leakage
    X_train = scaler.fit_transform(X_train)

    # Apply the same scaling transformation to test data
    X_test = scaler.transform(X_test)

    print("Training models...")
    models = train_models(X_train, y_train)

    print("Evaluating models...")
    evaluate_models(models, X_test, y_test)

    print("Saving models...")

    # Create models folder if it does not exist
    os.makedirs("models", exist_ok=True)

    # Save each trained model as a .pkl file
    for name, model in models.items():
        joblib.dump(model, f"models/{name}.pkl")

    # Save the scaler so new user inputs can be transformed the same way
    joblib.dump(scaler, "models/scaler.pkl")

    print("\nAll done!")
    print("Trained models saved in the models/ folder.")


if __name__ == "__main__":
    main()