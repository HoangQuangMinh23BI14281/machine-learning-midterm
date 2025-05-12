import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_data, load_models, standardize_with_stats
from utils.evaluate import accuracy, precision, recall, f1_score

# Custom implementation of confusion matrix
def custom_confusion_matrix(y_true, y_pred):
    # For binary classification
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP], [FN, TP]])

# Custom ROC curve implementation
def custom_roc_curve(y_true, y_scores):
    # Sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_scores, kind="mergesort")[::-1]
    y_score = y_scores[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    # Get distinct values to use as thresholds
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    # Calculate TPR and FPR for each threshold
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    
    # Add a point at the origin
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    
    return fpr, tpr

# Calculate Area Under Curve (AUC) using trapezoidal rule
def custom_auc(fpr, tpr):
    width = np.diff(fpr)
    height = (tpr[:-1] + tpr[1:]) / 2
    return np.sum(width * height)

def create_logistic_regression_graphs():
    try:
        print("Loading data and models...")
        X_linear, y_linear, X_logistic, y_logistic, linear_features, logistic_features, stats = load_data('./project/data/heart_disease_data.csv')
        
        models = load_models()
        
        # Try to load weights
        try:
            models['Logistic'].load_weights('./project/data/logistic_weights.npy')
            print("Logistic Regression weights loaded successfully.")
        except Exception as e:
            print(f"Failed to load weights: {str(e)}")
            print("Continuing with initialized model...")
        
        # Manual split for test data
        np.random.seed(50)
        test_size = 0.2
        indices = np.random.permutation(len(X_logistic))
        test_count = int(test_size * len(X_logistic))
        test_indices = indices[:test_count]
        
        X_test_logistic = X_logistic.iloc[test_indices]
        
        # Get test labels - convert to numpy array right away to avoid indexing issues
        if isinstance(y_logistic, dict):
            # If it's a dictionary, convert to numpy array
            actual_outcome = np.array(y_logistic['Outcome'][test_indices])
        else:
            # If it's a DataFrame/Series, convert to numpy array
            actual_outcome = np.array(y_logistic.iloc[test_indices]['Outcome'])
        
        # Make sure actual_outcome is a numpy array
        actual_outcome = np.asarray(actual_outcome)
        
        # Print debug info
        print(f"Type of actual_outcome: {type(actual_outcome)}")
        print(f"Shape of actual_outcome: {actual_outcome.shape}")
        print(f"First few values: {actual_outcome[:5]}")
        
        # Standardize test data
        X_test_std = standardize_with_stats(X_test_logistic, stats)
        
        # Make predictions
        print("Making predictions...")
        y_prob = models['Logistic'].predict_proba(X_test_std[logistic_features].values)
        y_pred = (y_prob >= 0.5).astype(int)
        
        # Calculate metrics
        acc = accuracy(actual_outcome, y_pred)
        prec = precision(actual_outcome, y_pred)
        rec = recall(actual_outcome, y_pred)
        f1 = f1_score(actual_outcome, y_pred)
        
        print(f"Metrics - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        cm = custom_confusion_matrix(actual_outcome, y_pred)
        axes[0, 0].imshow(cm, cmap='Blues')
        axes[0, 0].set_title(f'Confusion Matrix\nAccuracy: {acc:.4f}')
        
        # Add text labels
        for i in range(2):
            for j in range(2):
                axes[0, 0].text(j, i, str(cm[i, j]), 
                               ha='center', va='center',
                               color='white' if cm[i, j] > cm.max()/2 else 'black')
        
        axes[0, 0].set_xticks([0, 1])
        axes[0, 0].set_yticks([0, 1])
        axes[0, 0].set_xticklabels(['Negative (0)', 'Positive (1)'])
        axes[0, 0].set_yticklabels(['Negative (0)', 'Positive (1)'])
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. ROC Curve
        fpr, tpr = custom_roc_curve(actual_outcome, y_prob)
        roc_auc = custom_auc(fpr, tpr)
        
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                      label=f'ROC (AUC = {roc_auc:.2f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', lw=2)
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend(loc='lower right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Sigmoid Function Visualization
        x_range = np.linspace(-5, 5, 100)
        y_sigmoid = 1 / (1 + np.exp(-x_range))
        
        axes[1, 0].plot(x_range, y_sigmoid, 'b-', lw=2)
        axes[1, 0].set_title('Sigmoid (Logistic) Function')
        axes[1, 0].set_xlabel('Input (z)')
        axes[1, 0].set_ylabel('Probability P(y=1)')
        axes[1, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
        axes[1, 0].axvline(x=0, color='g', linestyle='--', alpha=0.3)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([-0.05, 1.05])
        
        # 4. Actual vs Predicted (simple version without sorting)
        # Just use the index directly
        x_indices = np.arange(len(actual_outcome))
        
        # Plot actual values
        axes[1, 1].scatter(x_indices, actual_outcome, color='blue', label='Actual', 
                         alpha=0.7, s=50)
        
        # Plot predicted probabilities
        axes[1, 1].scatter(x_indices, y_prob, color='red', label='Predicted Prob', 
                         alpha=0.5, s=50)
        
        # Add threshold line
        axes[1, 1].axhline(y=0.5, color='green', linestyle='--', 
                         label='Decision Threshold')
        
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Class / Probability')
        axes[1, 1].set_title('Actual vs. Predicted Probabilities')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.suptitle('Logistic Regression Evaluation', fontsize=16, y=0.98)
        fig.subplots_adjust(top=0.92)
        
        # Display plots without saving
        print("Displaying graphs (close window to continue)...")
        plt.show()
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_logistic_regression_graphs()