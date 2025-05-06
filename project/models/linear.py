import numpy as np

class LinearRegression:
    def __init__(self, lambda_l1=1.0, lambda_l2=1.0, learning_rate=0.01, max_iter=1000):
        self.weights = None
        self.var = None
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def cost_function(self, X, y, weight):
        m = len(y)
        prediction = X.dot(weight)
        cost = (1 / (2 * m)) * np.sum(np.square(prediction - y))
        l1_term = (self.lambda_l1 / m) * np.sum(np.abs(weight[1:]))
        l2_term = (self.lambda_l2 / (2 * m)) * np.sum(np.square(weight[1:]))
        return cost + l1_term + l2_term

    def gradient(self, X, y, weight):
        m = len(y)
        prediction = X.dot(weight)
        grad = (1 / m) * (X.T.dot(prediction - y))
        l1_grad = (self.lambda_l1 / m) * np.concatenate(([0], np.sign(weight[1:])))
        l2_grad = (self.lambda_l2 / m) * np.concatenate(([0], weight[1:]))
        return grad + l1_grad + l2_grad

    def fit(self, X, y):
        X_bias = np.c_[np.ones(X.shape[0]), X]
        n_samples, n_features = X_bias.shape
        self.weights = np.zeros((n_features, y.shape[1]))  # Multi-output regression

        for _ in range(self.max_iter):
            grad = self.gradient(X_bias, y, self.weights)
            self.weights -= self.learning_rate * grad

        # Compute variance of residuals for prediction intervals
        y_pred = self.predict(X)
        residuals = y - y_pred
        self.var = np.var(residuals, axis=0)

    def predict_trajectory(self, X_initial, years=100, change_params=None, change_years=None):
        """
        Predict the 100-year trajectory of all features.
        X_initial: Initial feature vector (1 sample, all features)
        change_params: Dict of {feature_idx: new_value} to change at specific years
        change_years: List of years when changes occur
        """
        trajectory = np.zeros((years, X_initial.shape[1]))
        X_current = X_initial.copy()
        
        # Assume a simple linear trend: predict future values as a perturbation
        for t in range(years):
            # Predict next year's values
            X_next = self.predict(X_current)
            trajectory[t] = X_next
            
            # Apply changes if specified
            if change_params and change_years and t in change_years:
                for feature_idx, new_value in change_params.items():
                    X_next[0, feature_idx] = new_value
            
            X_current = X_next
        
        return trajectory

    def predict_interval(self, X, confidence=0.95):
        y_pred = self.predict(X)
        z = 1.96  # For 95% confidence
        se = np.sqrt(self.var)
        lower = y_pred - z * se
        upper = y_pred + z * se
        return lower, upper