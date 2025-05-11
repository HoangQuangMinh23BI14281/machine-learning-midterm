import numpy as np

class LinearRegression:
    def __init__(self, lambda_l1, lambda_l2, learning_rate, max_iter):
        self.weights = None
        self.var = None
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.cost_history = []

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
        l1_grad = np.zeros_like(weight)
        l1_grad[1:] = (self.lambda_l1 / m) * np.sign(weight[1:])
        l2_grad = np.zeros_like(weight)
        l2_grad[1:] = (self.lambda_l2 / m) * weight[1:]
        return grad + l1_grad + l2_grad

    def fit(self, X, y):
        if X.shape[1] != 19:
            raise ValueError(f"Expected 19 features, got {X.shape[1]}")

        
        X_bias = np.c_[np.ones(X.shape[0]), X]  # (n_samples, 20)
        n_samples, n_features = X_bias.shape

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Initialize weights with small random values instead of zeros
        self.weights = np.random.randn(n_features, y.shape[1]) * 0.01

        # Print target statistics
        prev_cost = float('inf')
        patience = 50  # Number of iterations to wait for improvement
        patience_counter = 0
        best_weights = self.weights.copy()
        best_cost = float('inf')

        for i in range(self.max_iter):
            grad = self.gradient(X_bias, y, self.weights)
            self.weights -= self.learning_rate * grad
            
            # Calculate cost
            cost = self.cost_function(X_bias, y, self.weights)
            self.cost_history.append(cost)
            
            # Early stopping with patience
            if cost < best_cost:
                best_cost = cost
                best_weights = self.weights.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                self.weights = best_weights
                break
        y_pred = self.predict(X)
        residuals = y - y_pred
        self.var = np.var(residuals, axis=0)

    def load_weights(self, filepath):
        weights = np.load(filepath, allow_pickle=True)
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights, dtype=float)

        if weights.shape != (20, 12):
            raise ValueError(f"Expected weights shape (20, 12), but got {weights.shape}")

        self.weights = weights

    def predict(self, X):
        if X.shape[1] != 19:
            raise ValueError(f"Expected 19 input features, got {X.shape[1]}")
        if self.weights is None:
            raise ValueError("Model weights are not initialized. Please call fit() or load_weights() first.")
        X_bias = np.c_[np.ones(X.shape[0]), X]
        return X_bias.dot(self.weights)

