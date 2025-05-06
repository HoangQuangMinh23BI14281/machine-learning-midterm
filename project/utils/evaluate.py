import numpy as np
import matplotlib.pyplot as plt

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
