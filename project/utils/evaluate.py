import numpy as np

def mae(y_true, y_pred):
    """
    Compute Mean Absolute Error (MAE).
    y_true: True values (numpy array)
    y_pred: Predicted values (numpy array)
    """
    return np.mean(np.abs(y_true - y_pred))

def mse(y_true, y_pred):
    """
    Compute Mean Squared Error (MSE).
    y_true: True values (numpy array)
    y_pred: Predicted values (numpy array)
    """
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    """
    Compute R² Score (Coefficient of Determination).
    y_true: True values (numpy array)
    y_pred: Predicted values (numpy array)
    """
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    if ss_tot == 0:
        return 0  # Avoid division by zero
    return 1 - (ss_res / ss_tot)

def accuracy(y_true, y_pred):
    """
    Compute Accuracy for binary classification.
    y_true: True binary labels (0 or 1)
    y_pred: Predicted binary labels (0 or 1)
    """
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    """
    Compute Precision for binary classification.
    y_true: True binary labels (0 or 1)
    y_pred: Predicted binary labels (0 or 1)
    """
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    predicted_positives = np.sum(y_pred == 1)
    if predicted_positives == 0:
        return 0
    return true_positives / predicted_positives

def recall(y_true, y_pred):
    """
    Compute Recall for binary classification.
    y_true: True binary labels (0 or 1)
    y_pred: Predicted binary labels (0 or 1)
    """
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    actual_positives = np.sum(y_true == 1)
    if actual_positives == 0:
        return 0
    return true_positives / actual_positives

def f1_score(y_true, y_pred):
    """
    Compute F1-Score for binary classification.
    y_true: True binary labels (0 or 1)
    y_pred: Predicted binary labels (0 or 1)
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    if prec + rec == 0:
        return 0
    return 2 * (prec * rec) / (prec + rec)

def log_loss(y_true, y_pred_proba):
    """
    Compute Log Loss (Binary Cross-Entropy).
    y_true: True binary labels (0 or 1)
    y_pred_proba: Predicted probabilities for class 1
    """
    epsilon = 1e-15  # To avoid log(0)
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))