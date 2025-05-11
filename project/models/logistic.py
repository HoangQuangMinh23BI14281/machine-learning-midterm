import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate, max_iter, lambda_l1, lambda_l2):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.weights = None
        self.cost_history = []  # Added to track cost history

    def sigmoid(self, z):
        # Prevent overflow by clipping z
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def cost_function(self, X, y, weight):
        m = X.shape[0]
        prediction = self.sigmoid(np.dot(X, weight))  
        cost = (-1 / m) * np.sum(y * np.log(prediction + 1e-9) + (1 - y) * np.log(1 - prediction + 1e-9))
        # Add L1 and L2 regularization terms
        l1_term = (self.lambda_l1 / m) * np.sum(np.abs(weight[1:]))
        l2_term = (self.lambda_l2 / (2 * m)) * np.sum(np.square(weight[1:]))
        return cost + l1_term + l2_term

    def gradient(self, X, y, weight):
        m = X.shape[0]
        # Compute predictions
        prediction = self.sigmoid(np.dot(X, weight)) 
        error = prediction - y
        # Compute gradient
        grad = (1 / m) * np.dot(X.T, error)  
        # Add L1 and L2 regularization gradients
        l1_grad = (self.lambda_l1 / m) * np.concatenate(([0], np.sign(weight[1:])))
        l2_grad = (self.lambda_l2 / m) * np.concatenate(([0], weight[1:]))
        return grad + l1_grad + l2_grad

    def fit(self, X, y):
        # Add bias term (column of ones)
        X_bias = np.c_[np.ones(X.shape[0]), X]  # Shape: (n_samples, n_features + 1)
        
        # Initialize weights if not already set
        if self.weights is None:
            # Initialize with small random values instead of zeros for better convergence
            self.weights = np.random.randn(X_bias.shape[1]) * 0.01
        
        # Variables for early stopping
        best_cost = float('inf')
        best_weights = self.weights.copy()
        patience = 50  # Number of iterations to wait for improvement
        patience_counter = 0
        
        # Gradient descent with early stopping
        for i in range(self.max_iter):
            grad = self.gradient(X_bias, y, self.weights)
            self.weights -= self.learning_rate * grad
            
            # Calculate and store cost
            cost = self.cost_function(X_bias, y, self.weights)
            self.cost_history.append(cost)
            
            # Early stopping logic
            if cost < best_cost:
                best_cost = cost
                best_weights = self.weights.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                            
            # Check if we should stop early
            if patience_counter >= patience:
                print(f"Early stopping at iteration {i}")
                self.weights = best_weights
                break
                
        # Ensure we use the best weights found
        self.weights = best_weights
        

    def load_weights(self, filepath):
        # Load weights from a file
        self.weights = np.load(filepath)

    def predict_proba(self, X):
        if self.weights is None:
            raise ValueError("Model weights are not initialized. Please call fit() or load_weights() first.")
        X_bias = np.c_[np.ones(X.shape[0]), X]
        return self.sigmoid(np.dot(X_bias, self.weights))  

    def predict(self, X):
        # Predict binary class labels (0 or 1)
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)