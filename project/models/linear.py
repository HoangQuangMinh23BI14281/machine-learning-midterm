import numpy as np 
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_reg=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weight = None
        self.lambda_reg = lambda_reg

    def cost_fuction(self, X, y, weight):
        m = len(y)
        prediction = X.dot(weight)
        cost = (1 / (2 * m)) * np.sum((prediction - y) ** 2) + (self.lambda_reg / (2 * m)) * np.sum(weight[1:] ** 2)
        return cost
    
    def gradient_descent(self, X, y, weight):
        m = len(y)
        prediction = X.dot(weight)
        gradients = (1/m) * X.T.dot(prediction - y)
        gradients[1:] += (self.lambda_reg/m) * weight[1:]
        weight = weight - self.learning_rate * gradients
        return weight
    
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.weight = np.random.rand(X_b.shape[1])
        self.weight = self.gradient_descent(X_b, y, self.weight)

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.weight)