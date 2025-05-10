import numpy as np
import pandas as pd
from utils import load_data, load_models, standardize_with_stats, train_test_split

def train():
    # Load data
    X_linear, y_linear, X_logistic, y_logistic, linear_features, logistic_features, stats = load_data('./project/data/heart_attack_dataset.csv')
    
    # Define feature groups
    feature_groups = {
        "Demographic": ['Age', 'Gender', 'Ethnicity', 'Income', 'EducationLevel', 'Residence', 'EmploymentStatus', 'MaritalStatus'],
        "Lifestyle": ['Smoker', 'PhysicalActivity', 'AlcoholConsumption', 'Diet', 'StressLevel'],
        "Medical History": ['Diabetes', 'Hypertension', 'FamilyHistory', 'Medication', 'PreviousHeartAttack', 'StrokeHistory'],
        "Clinical Tests": ['Cholesterol', 'BloodPressure', 'HeartRate', 'BMI', 'MaxHeartRate', 'ST_Depression', 'NumberOfMajorVessels'],
        "Symptoms & Diagnostics": ['ChestPainType', 'ECGResults', 'ExerciseInducedAngina', 'Slope', 'Thalassemia']
    }
    input_features = feature_groups["Demographic"] + feature_groups["Lifestyle"] + feature_groups["Medical History"]
    target_features = feature_groups["Clinical Tests"] + feature_groups["Symptoms & Diagnostics"]

    # Prepare data for Linear Regression (Trajectory)
    print("\nPreparing data for Linear Regression...")
    # Use input features for X and target features for y
    X_linear_input = X_logistic[input_features].values
    y_linear_target = X_logistic[target_features].values

    # Train-test split for Linear Regression
    X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(
        X_linear_input, y_linear_target, test_size=0.2, seed=50
    )

    # Prepare data for Logistic Regression
    print("\nPreparing data for Logistic Regression...")
    # Use all features for logistic regression
    X_logistic_input = X_logistic[logistic_features].values
    y_logistic_target = y_logistic['Outcome'].values

    # Train-test split for Logistic Regression
    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
        X_logistic_input, y_logistic_target, test_size=0.2, seed=50
    )

    # Load models
    models = load_models()

    # Train Linear Regression (Trajectory)
    print("\nTraining Linear Regression (Trajectory)...")
    print(f"Input shape: {X_train_linear.shape}, Target shape: {y_train_linear.shape}")
    models['Trajectory'].fit(X_train_linear, y_train_linear)
    np.save('./project/data/linear_weights.npy', models['Trajectory'].weights)
    print(f"Linear Regression weights saved to 'data/linear_weights.npy' with shape: {models['Trajectory'].weights.shape}")

    # Train Logistic Regression
    print("\nTraining Logistic Regression (Heart Attack Risk)...")
    print(f"Input shape: {X_train_log.shape}, Target shape: {y_train_log.shape}")
    models['Logistic'].fit(X_train_log, y_train_log)
    np.save('./project/data/logistic_weights.npy', models['Logistic'].weights)
    print(f"Logistic Regression weights saved to 'data/logistic_weights.npy' with shape: {models['Logistic'].weights.shape}")

    print("\nTraining completed.")

if __name__ == "__main__":
    train()
