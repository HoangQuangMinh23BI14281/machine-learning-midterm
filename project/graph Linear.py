import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_data, load_models, standardize_with_stats, inverse_standardize
from utils.evaluate import mae, mse, r2_score

def create_linear_regression_comparison_graphs():
    print("Loading data and models...")
    # Load data
    X_linear, y_linear, X_logistic, y_logistic, linear_features, logistic_features, stats = load_data('./project/data/heart_disease_data.csv')
    
    # Extract the target features we want to predict
    target_features = ['Cholesterol', 'BloodPressure', 'HeartRate', 'BMI']
    
    # Extract these features from the original dataset
    y_linear_df = pd.DataFrame()
    for feature in target_features:
        if feature in y_linear:
            y_linear_df[feature] = y_linear[feature]
        elif feature in X_linear.columns:
            y_linear_df[feature] = X_linear[feature]
        else:
            print(f"Warning: Feature {feature} not found in data")
    
    # Now split the data
    np.random.seed(50)  # Set seed for reproducibility
    test_size = 0.2
    indices = np.random.permutation(len(X_linear))
    test_count = int(test_size * len(X_linear))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    
    # Manual split
    X_train_linear = X_linear.iloc[train_indices]
    X_test_linear = X_linear.iloc[test_indices]
    y_train_linear = y_linear_df.iloc[train_indices]
    y_test_linear = y_linear_df.iloc[test_indices]
    
    # Load models
    models = load_models()
    
    # Load pre-trained weights
    try:
        models['Trajectory'].load_weights('./project/data/linear_weights.npy')
        print("Linear Regression weights loaded successfully.")
    except Exception as e:
        print(f"Failed to load Linear Regression weights: {str(e)}")
        return
    
    # Make predictions using the linear regression model
    print("Making predictions...")
    linear_predictions = models['Trajectory'].predict(X_test_linear[linear_features].values)
    
    # Ensure linear_predictions has the right shape
    if linear_predictions.ndim == 1:
        linear_predictions = linear_predictions.reshape(-1, 1)
    
    # Create a 2x2 grid of plots for the target features
    plt.figure(figsize=(15, 12))
    
    # For each target feature
    for i, feature in enumerate(target_features):
        if feature not in y_test_linear.columns:
            print(f"Warning: Feature {feature} not in test data")
            continue
            
        # Extract actual values for this feature
        actual_values = y_test_linear[feature].values
        
        # Extract predicted values for this feature
        pred_values = linear_predictions[:, i] if i < linear_predictions.shape[1] else []
        
        if len(pred_values) == 0:
            print(f"Warning: No predictions for feature {feature}")
            continue
        
        # Create DataFrames for inverse standardization
        actual_df = pd.DataFrame(actual_values, columns=[feature])
        pred_df = pd.DataFrame(pred_values, columns=[feature])
        
        # Inverse standardize to get original scale
        actual_original = inverse_standardize(actual_df, stats)
        pred_original = inverse_standardize(pred_df, stats)
        
        # Get the values as arrays
        actual_vals = actual_original[feature].values
        pred_vals = pred_original[feature].values
        
        # Calculate metrics
        feature_mae = mae(actual_vals, pred_vals)
        feature_mse = mse(actual_vals, pred_vals)
        feature_r2 = r2_score(actual_vals, pred_vals)
        
        # Create subplot
        plt.subplot(2, 2, i+1)
        
        # Create scatter plot of actual vs predicted values
        plt.scatter(actual_vals, pred_vals, alpha=0.5, color='blue', label='Data points')
        
        # Add perfect prediction line (y=x)
        min_val = min(min(actual_vals), min(pred_vals))
        max_val = max(max(actual_vals), max(pred_vals))
        plt.plot([min_val, max_val], [min_val, max_val], 'r-', label='Perfect prediction')
        
        # Add trend line
        z = np.polyfit(actual_vals, pred_vals, 1)
        p = np.poly1d(z)
        plt.plot(sorted(actual_vals), p(sorted(actual_vals)), "g--", 
                label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
        
        # Set title and labels
        plt.title(f'{feature}\nMAE: {feature_mae:.2f}, MSE: {feature_mse:.2f}, R²: {feature_r2:.2f}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Linear Regression: Actual vs. Predicted Values (Scatter Plot)', fontsize=16, y=1.02)
    
    plt.show()

if __name__ == "__main__":
    create_linear_regression_comparison_graphs()