
from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize_scalar # this function helps to the chapter 4 of AFO
import jax
import jax.numpy as jnp

class DescentMethod(ABC):
    @abstractmethod
    def init(self, f, grad_f, x0):
        """Initialize the descent method"""
        pass
    
    @abstractmethod
    def step(self, f, grad_f, x):
        """Perform one optimization step"""
        pass

def iterated_descent(method: DescentMethod, f, grad_f, x0, k_max=1000, tol=1e-8):
    """
    Perform iterative descent optimization 
    """
    method.init(f, grad_f, x0)
    x = x0.copy()
    history = [x.copy()]
    prev_loss = f(x)
    
    for i in range(k_max):
        x_new = method.step(f, grad_f, x)
        current_loss = f(x_new)
        
        # Check for convergence based on parameter change and loss change
        param_change = jnp.linalg.norm(x_new - x)
        loss_change = abs(current_loss - prev_loss)
        
        if param_change < tol and loss_change < tol:
            print(f"Converged after {i+1} iterations")
            break
            
        x = x_new
        prev_loss = current_loss
        history.append(x.copy())
        
        # Print progress every 100 iterations
        if (i + 1) % 100 == 0:
            print(f"Iteration {i+1}: Loss = {current_loss:.6e}")
    
    return x, jnp.array(history)

class GradientDescent(DescentMethod):
    def __init__(self, alpha=0.01):
        '''Initialize GD. alpha: step size/learning rate'''
        self.alpha = alpha
        
    def init(self, f, grad_f, x0):
        """No special initialization needed for basic gradient descent"""
        return self
    
    def step(self, f, grad_f, x):
        return x - self.alpha * grad_f(x)

class AdaptiveGradientDescent(DescentMethod):
    """Gradient descent with adaptive step size and momentum"""
    def __init__(self, alpha=0.01, beta=0.9, adaptive=True):
        self.alpha = alpha
        self.beta = beta  # momentum coefficient
        self.adaptive = adaptive
        self.velocity = None
        self.prev_loss = None
        
    def init(self, f, grad_f, x0):
        self.velocity = jnp.zeros_like(x0)
        self.prev_loss = f(x0)
        return self
    
    def step(self, f, grad_f, x):
        grad = grad_f(x)
        
        # Momentum update
        self.velocity = self.beta * self.velocity + (1 - self.beta) * grad
        
        # Adaptive step size
        if self.adaptive and self.prev_loss is not None:
            current_loss = f(x)
            if current_loss > self.prev_loss:
                self.alpha *= 0.9  # Reduce step size if loss increased
            else:
                self.alpha *= 1.01  # Slightly increase if loss decreased
            self.prev_loss = current_loss
        
        return x - self.alpha * self.velocity

class NewtonMethod(DescentMethod):
    """Newton's method using Hessian matrix"""
    def __init__(self, damping=1e-6):
        self.damping = damping  # Ridge regularization for numerical stability
        
    def init(self, f, grad_f, x0):
        return self
    
    def step(self, f, grad_f, x):
        grad = grad_f(x)
        hess = jax.hessian(f)(x)
        
        # Add damping for numerical stability
        hess_reg = hess + self.damping * jnp.eye(len(x))
        
        # Newton step: x_new = x - H^(-1) * grad
        try:
            newton_step = jnp.linalg.solve(hess_reg, grad)
            return x - newton_step
        except:
            # Fallback to gradient descent if Hessian is singular
            return x - 0.01 * grad

class LevenbergMarquardt(DescentMethod):
    """Levenberg-Marquardt algorithm for nonlinear least squares"""
    def __init__(self, lambda_init=1e-3, lambda_up=10, lambda_down=0.1):
        self.lambda_param = lambda_init
        self.lambda_up = lambda_up
        self.lambda_down = lambda_down
        self.prev_loss = None
        
    def init(self, f, grad_f, x0):
        self.prev_loss = f(x0)
        return self
    
    def step(self, f, grad_f, x):
        grad = grad_f(x)
        hess = jax.hessian(f)(x)
        
        # LM update: (H + λI)^(-1) * grad
        lm_matrix = hess + self.lambda_param * jnp.eye(len(x))
        
        try:
            lm_step = jnp.linalg.solve(lm_matrix, grad)
            x_new = x - lm_step
            
            current_loss = f(x_new)
            
            # Adjust lambda based on improvement
            if current_loss < self.prev_loss:
                self.lambda_param *= self.lambda_down  # Decrease lambda (move toward Newton)
                self.prev_loss = current_loss
                return x_new
            else:
                self.lambda_param *= self.lambda_up    # Increase lambda (move toward GD)
                return x  # Don't update if no improvement
                
        except:
            # Fallback to gradient descent
            return x - 0.01 * grad

class QuasiNewtonBFGS(DescentMethod):
    """BFGS Quasi-Newton method"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.H = None  # Inverse Hessian approximation
        self.prev_x = None
        self.prev_grad = None
        
    def init(self, f, grad_f, x0):
        self.H = jnp.eye(len(x0))  # Initialize as identity
        self.prev_x = x0
        self.prev_grad = grad_f(x0)
        return self
    
    def step(self, f, grad_f, x):
        grad = grad_f(x)
        
        if self.prev_x is not None:
            # BFGS update
            s = x - self.prev_x  # step
            y = grad - self.prev_grad  # gradient change
            
            # Avoid division by zero
            sy = jnp.dot(s, y)
            if sy > 1e-10:
                # BFGS formula
                rho = 1.0 / sy
                H_y = jnp.dot(self.H, y)
                self.H = self.H - rho * jnp.outer(H_y, H_y) / jnp.dot(y, H_y) + rho * jnp.outer(s, s)
        
        # Update for next iteration
        self.prev_x = x
        self.prev_grad = grad
        
        # BFGS step
        search_direction = -jnp.dot(self.H, grad)
        return x + self.alpha * search_direction

class AdamOptimizer(DescentMethod):
    """Adam optimizer with adaptive learning rates"""
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step
        
    def init(self, f, grad_f, x0):
        self.m = jnp.zeros_like(x0)
        self.v = jnp.zeros_like(x0)
        self.t = 0
        return self
    
    def step(self, f, grad_f, x):
        self.t += 1
        grad = grad_f(x)
        
        # Update biased first and second moment estimates
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        
        # Compute bias-corrected moment estimates
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Adam update
        return x - self.alpha * m_hat / (jnp.sqrt(v_hat) + self.epsilon)

def bracket_minimum(f, x=0, s=1e-2, k=2.0):
    '''    
    Bracket the minimum of a function.
    Returns a tuple (a, c) that contains the minimum.
    '''
    a, ya = x, f(x)
    b, yb = a + s, f(a + s)
    if yb > ya: # downhill
        a, b = b, a
        ya, yb = yb, ya
        s = -s

    while True: # expand the bracket
        c, yc = b + s, f(b + s)
        if yc > yb:
            return min(a,c), max(a,c)
        a, ya, b, yb = b, yb, c, yc
        s *= k

def line_search(f, x, d):
    '''
    Directly optimize the step factor to minimize 
    the objective function
    f: is the function, x is the initial point and d the directional derivative
    '''
    objective = lambda alpha: f(x + alpha*d)
    a, b = bracket_minimum(objective)
    res = minimize_scalar(objective, bounds=(a,b), method='bounded')
    alpha = res.x # res is the optimizal step size, here we acces a stored value
    return x + alpha*d, alpha # also return alpha min


class GradientDescent(DescentMethod):
    def __init__(self, alpha=0.01):
        '''Initialize GD. alpha: step size/learning rate'''
        self.alpha = alpha
        
    def init(self, f, grad_f, x0):
        """No special initialization needed for basic gradient descent"""
        return self
    
    def step(self, f, grad_f, x):
        return x - self.alpha * grad_f(x)

class Momentum(DescentMethod):
    def __init__(self, alpha, beta, v):
        '''alpha: step factor, beta: momentum decay, v:momentum'''
        self.alpha = alpha
        self.beta = beta
        self.v = None
    
    def init(self, f, grad_f, x0):
        self.v = np.zeros(len(x0))
        return self

    def step(self, f, grad_f, x):
        '''Perform one momentum descent step'''
        self.v = self.beta * self.v + self.alpha * grad_f(x) # update momentum
        return x - self.v # update position
    
def run_optim(method, f, grad_f, x0, k_max, method_name):
    '''A function to prove algorithms'''
    result, history = iterated_descent(method, f, grad_f, x0, k_max)
    print(f'{method_name}')
    print(f'    Starting point: {x0}')
    print(f'    Final point {result}')
    print(f'    Function value: {f(result):.4f}')
    print(f'    Distance from true minimum (1, 1): {np.linalg.norm(result - np.array([1,1])):.3f}')
    return result, history


class ModelOptimizer:
    '''Handle any kinetic model'''
    def __init__(self, model, target_data, loss_type='mse', param_names=None):
        """Initialize optimizer for a specific model. """
        self.model = model
        self.target = target_data
        self.loss_type = loss_type
        self.param_names = param_names or [f'param_{i}' for i in range(5)]
        
        self._loss_fn = self._create_loss_function()  # as advice from https://docs.jax.dev/en/latest/index.html, precompile JAX functions for efficiency
        self._grad_fn = jax.grad(self._loss_fn)
        self._hess_fn = jax.hessian(self._loss_fn)
        
        print(f"ModelOptimizer initialized with {loss_type.upper()} loss")
    
    def _create_loss_function(self):
        """Create loss function based on specified type"""
        if self.loss_type == 'mse':
            def loss_fn(params):
                predicted = self.model(params)
                return jnp.mean((predicted - self.target)**2)
        elif self.loss_type == 'mae':
            def loss_fn(params):
                predicted = self.model(params)
                return jnp.mean(jnp.abs(predicted - self.target))
        elif self.loss_type == 'rmse':
            def loss_fn(params):
                predicted = self.model(params)
                return jnp.sqrt(jnp.mean((predicted - self.target)**2))
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss_fn
    
    def loss(self, params):
        return self._loss_fn(params)
    
    def grad(self, params):
        return self._grad_fn(params)
    
    def hessian(self, params):
        return self._hess_fn(params)
    
    def predict(self, params):
        return self.model(params)
    
    def optimize(self, method, x0, max_iter=1000, method_name=None):
        """
        Run optimization using specified method.
        
        Parameters:
        - method: DescentMethod instance <- 
        - x0: Initial parameters
        - max_iter: Maximum iterations
        - method_name: Name for reporting
        
        Returns:
        - result: Optimized parameters
        - history: Parameter history during optimization
        """
        if method_name is None:
            method_name = f"{method.__class__.__name__}"
        
        result, history = iterated_descent(method, self.loss, self.grad, x0, max_iter)
        
        print(f'\n{method_name} Results:')
        print(f'  Starting point: {x0}')
        print(f'  Final point: {result}')
        print(f'  Final {self.loss_type.upper()}: {self.loss(result):.6e}')
        print(f'  Iterations: {len(history)}')
        
        print(f'  Optimized Parameters:')
        for name, value in zip(self.param_names, result):
            print(f'    {name}: {value:.4e}')
        
        return result, history
    
    def compare_methods(self, methods_dict, x0, max_iter=1000):
        """Compare multiple optimization methods"""
        results = {}
        
        print("=" * 80)
        print("COMPARING OPTIMIZATION METHODS")
        print("=" * 80)
        
        for name, method in methods_dict.items():
            try:
                result, history = self.optimize(method, x0.copy(), max_iter, name)
                results[name] = {
                    'result': result,
                    'history': history,
                    'final_loss': self.loss(result),
                    'iterations': len(history)
                }
            except Exception as e:
                print(f"Error with {name}: {e}")
                results[name] = None
        
        # Summary comparison
        print(f"\n{'Method':<25} {'Final Loss':<15} {'Iterations':<12} {'Success':<10}")
        print("-" * 65)
        for name, res in results.items():
            if res is not None:
                print(f"{name:<25} {res['final_loss']:<15.6e} {res['iterations']:<12} {'Yes':<10}")
            else:
                print(f"{name:<25} {'Failed':<15} {'N/A':<12} {'No':<10}")
        
        return results