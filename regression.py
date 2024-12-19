import numpy as np
import joblib
import matplotlib.pyplot as plt

class CustomRegression:
    def __init__(self, alpha=0.01, iterations=1000):
        self.alpha = alpha
        self.iterations = iterations
        self.theta = None

    def cost_function(self, X, y):
        m = len(y)
        predictions = X.dot(self.theta)
        cost = (1 / (2 * m)) * np.sum((y - predictions) ** 2)
        return cost

    def gradient_descent(self, X, y):
        m, n = X.shape
        self.theta = np.zeros((n + 1, 1))
        cost_history = np.zeros(self.iterations)
        X_b = np.c_[np.ones((m, 1)), X]

        if y.ndim == 1:
            y = np.array(y).reshape(-1, 1)

        for i in range(self.iterations):
            predictions = X_b.dot(self.theta)
            errors = predictions - y
            gradient = (1 / m) * X_b.T.dot(errors)
            self.theta -= self.alpha * gradient
            cost_history[i] = self.cost_function(X_b, y)

        return self.theta, cost_history

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)
    
    def score(self, X, y):
        y = np.array(y).reshape(-1, 1)
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

    def evaluate(self, X, y):
        r2 = self.score(X, y)
        y = np.array(y).reshape(-1, 1)
        mse = np.mean((self.predict(X) - y) ** 2)
        print(f"Erreur quadratique moyenne (MSE) : {mse}")
        print(f"R² : {r2}")

    def learning_curve(self, cost_history):
        plt.plot(range(self.iterations), cost_history)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Learning Curve')
        plt.show()

    def save_model(self, filename):
        joblib.dump(self.theta, filename)
    
    def fit(self, X, y):
        self.gradient_descent(X, y)
        return self
    
    
class Utils:
    @staticmethod 
    def grid_search(X_train, y_train, X_test, y_test, param_grid):
        best_score = -np.inf
        best_params = None
        best_model = None
        
        for alpha in param_grid['alpha']:
            for iterations in param_grid['iterations']:
                model = CustomRegression(alpha=alpha, iterations=iterations)
                model.fit(X_train, y_train)
                
                score = model.score(X_test, y_test)
                
                if score > best_score:
                    best_score = score
                    best_params = {'alpha': alpha, 'iterations': iterations}
                    best_model = model
        
        print("\nMeilleurs paramètres trouvés :")
        print(best_params)
        print(f"Meilleur score R² : {best_score}")
        
        return best_model, best_params

    @staticmethod
    def polynomial_features(X, degree):
        X_poly = X.copy()
        for d in range(2, degree + 1):
            X_poly = np.c_[X_poly, X[:, 0]**d]
        return X_poly