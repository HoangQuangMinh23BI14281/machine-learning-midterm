from utils import preprocess, evaluate
from models import logistic, ann
import pandas as pd
import numpy as np
from config import *

def main():
    # Load and preprocess data
    data_frame = pd.read_csv('data/heart_attack_dataset.csv')
    print("Dataset columns:", data_frame.columns.tolist())
    
    data_after_preprocess, stats = preprocess.full_preprocess(data_frame)
    print("Preprocessed dataset columns:", data_after_preprocess.columns.tolist())
    
    # Prepare features and target
    X = data_after_preprocess.drop(TARGET_COLUMN, axis=1).values
    y = data_after_preprocess[TARGET_COLUMN].values
    
    # Convert continuous target to binary (0 or 1)
    # Assuming values above 0.5 indicate heart attack
    y = (y > 0.5).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocess.train_test_split(
        X, y, test_size=TEST_SIZE, seed=RANDOM_SEED
    )
    
    # Logistic Regression
    print("\n=== Logistic Regression ===")
    log_model = logistic.LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
    log_model.fit(X_train, y_train)
    y_pred_log = log_model.predict(X_test)
    log_metrics = evaluate.evaluate_classification(y_test, y_pred_log)
    evaluate.print_metrics(log_metrics)
    
    # Neural Network
    print("\n=== Neural Network ===")
    input_size = X_train.shape[1]
    output_size = 1
    nn_model = ann.NeuralNetwork(
        input_size=input_size,
        output_size=output_size,
        **NEURAL_NETWORK_PARAMS
    )
    nn_model.fit(X_train, y_train.reshape(-1, 1), epochs=1000)
    y_pred_nn = nn_model.predict(X_test)
    nn_metrics = evaluate.evaluate_classification(y_test, y_pred_nn)
    evaluate.print_metrics(nn_metrics)

if __name__ == "__main__":
    main()
