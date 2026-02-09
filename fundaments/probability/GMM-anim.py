import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse

class GaussianMixtureEM:
    def __init__(self, n_components=3, max_iter=50, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state
        self.history = []  
        
    def fit(self, X):
        n_samples, n_features = X.shape
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])
        self.weights = np.ones(self.n_components) / self.n_components
        
        for iteration in range(self.max_iter):
            # E-step
            responsibilities = np.array([
                self.weights[k] * multivariate_normal.pdf(X, self.means[k], self.covariances[k])
                for k in range(self.n_components)
            ])
            responsibilities /= responsibilities.sum(axis=0)
            
            # M-step
            Nk = responsibilities.sum(axis=1)
            self.weights = Nk / n_samples
            self.means = np.array([
                np.sum(responsibilities[k].reshape(-1, 1) * X, axis=0) / Nk[k]
                for k in range(self.n_components)
            ])
            for k in range(self.n_components):
                diff = X - self.means[k]
                self.covariances[k] = np.dot(responsibilities[k] * diff.T, diff) / Nk[k]
                self.covariances[k] += 1e-6 * np.eye(n_features)  # Regularization
            
            self.history.append({
                'means': self.means.copy(),
                'covariances': self.covariances.copy(),
                'weights': self.weights.copy(),
                'iteration': iteration
            })

    def plot_iteration(self, iteration):
        """Plot the estimated Gaussians at a specific iteration"""
        params = self.history[iteration]
        plt.figure(figsize=(10, 6))
        
        # Plot data points
        plt.scatter(X[:, 0], X[:, 1], alpha=0.3, c='gray', label='Data')
        
        # Plot estimated Gaussians
        for k in range(self.n_components):
            # Draw ellipse (2 standard deviations)
            eigvals, eigvecs = np.linalg.eigh(params['covariances'][k])
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            width, height = 4 * np.sqrt(eigvals)  # 2 std dev = 95% confidence
            ellipse = Ellipse(params['means'][k], width, height, angle=angle,
                             edgecolor='red', facecolor='none', linewidth=2)
            plt.gca().add_patch(ellipse)
            
            # Mark mean
            plt.scatter(params['means'][k, 0], params['means'][k, 1], 
                       marker='x', s=100, c='red', linewidths=2)
        
        plt.title(f'EM Iteration {iteration + 1}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

# ggenerate sample data
np.random.seed(17)
X = np.vstack([
    np.random.multivariate_normal([1, 1], [[1, 0.5], [0.5, 1]], 100),
    np.random.multivariate_normal([5, 5], [[1, -0.5], [-0.5, 1]], 100),
    np.random.multivariate_normal([9, 1], [[1, 0], [0, 1]], 100)
])

# fit the model and store history
gmm = GaussianMixtureEM(n_components=3, max_iter=50, random_state=42)
gmm.fit(X)

fig, ax = plt.subplots(figsize=(10, 6))
def update(i):
    ax.clear()
    params = gmm.history[i]
    ax.scatter(X[:, 0], X[:, 1], alpha=0.3, c='gray')
    for k in range(gmm.n_components):
        eigvals, eigvecs = np.linalg.eigh(params['covariances'][k])
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 4 * np.sqrt(eigvals)
        ellipse = Ellipse(params['means'][k], width, height, angle=angle,
                         edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(ellipse)
        ax.scatter(params['means'][k, 0], params['means'][k, 1], 
                   marker='x', s=100, c='red')
    ax.set_title(f'EM Iteration {i + 1}')
    ax.grid(True)
    ax.axis('equal')

ani = FuncAnimation(fig, update, frames=len(gmm.history), interval=500)
plt.show()
# ani.save('em_progress.gif', writer='pillow')  # Save as GIF