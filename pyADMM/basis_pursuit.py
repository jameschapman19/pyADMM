import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from abc import abstractmethod
from numba import jit
from time import time


class _ADMM:
    def __init__(self, rho: float, alpha: float, quiet: bool = True, max_iter=1000, abstol=1e-4, reltol=1e-2):
        self.rho = rho
        self.alpha = alpha
        self.quiet = quiet
        self.max_iter = max_iter
        self.abstol = abstol
        self.reltol = reltol

    @abstractmethod
    def objective(self, *args):
        pass


class BasisPursuit(_ADMM):
    """
    Solve basis pursuit via ADMM

    Solves the following problem via ADMM:

    minimize     ||x||_1
    subject to   Ax = b

    PORTED FROM https://web.stanford.edu/~boyd/admm.html

    More information can be found in the paper linked at:
    http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
    """

    def __init__(self, rho: float, alpha: float, quiet: bool = True, max_iter=1000, abstol=1e-4, reltol=1e-2):
        """

        :param rho: augmented lagrangian parameter
        :param.alpha: typical values betwen 1.0 and 1.8
        """
        super().__init__(rho, alpha, quiet, max_iter, abstol, reltol)

    def fit(self, A, b):
        # precompute static variables for x-update (projection on to Ax=b)
        AAt = A @ A.T
        P = np.eye(A.shape[1]) - A.T @ sparse.linalg.spsolve(AAt, A)[0]
        q = A.T @ spsolve(AAt, b)[0]
        x, history = _fit(A, b, P, q, self.rho, self.alpha, self.abstol, self.reltol, self.max_iter)
        self.history = {
            'objval': history[0],
            'r_norm': history[1],
            's_norm': history[2],
            'eps_pri': history[3],
            'eps_dual': history[4]
        }
        return x


@jit(nopython=True, cache=True)
def _fit(A, b, P, q, rho, alpha, abstol, reltol, max_iter):
    (m, n) = A.shape
    x = np.zeros((n, 1))
    z = np.zeros((n, 1))
    u = np.zeros((n, 1))
    AAt = A @ A.T
    history = np.zeros((5, max_iter))

    for k in range(max_iter):
        # x-update
        x = P @ (z - u) + q

        # z-update with relaxation
        zold = z
        x_hat = alpha * x + (1 - alpha) * zold
        z = _shrinkage(x_hat + u, 1 / rho)

        u = u + (x_hat - z)

        # diagnostics, reporting, termination checks
        history[0, k] = _objective(x)
        history[1, k] = np.linalg.norm(x - z)
        history[2, k] = np.linalg.norm(-rho * (z - zold))
        history[3, k] = np.sqrt(n) * abstol + reltol * max(np.linalg.norm(x), np.linalg.norm(-z))
        history[4, k] = np.sqrt(n) * abstol + reltol * np.linalg.norm(rho * u)
        if history[1][k] < history[3][k] and history[2][k] < history[4][k]:
            break
    return x, history


@jit(nopython=True, cache=True)
def _objective(x):
    """

    :param A:
    :param b:
    :param x:
    """
    return np.linalg.norm(x, ord=1)


@jit(nopython=True, cache=True)
def _shrinkage(a, kappa):
    """

    :param a:
    :param kappa:
    """
    return np.maximum(0,a-kappa)-np.maximum(0,-a-kappa)


def main():
    t0 = time()
    n = 30
    m = 10
    A = np.random.rand(m, n)

    x = sparse.rand(n, 1, density=0.1)
    b = A * x

    xtrue = x

    bp = BasisPursuit(1, 1)
    x = bp.fit(A, b)
    t0 = time()
    for _ in range(100):
        x = bp.fit(A, b)
    print(time() - t0)
    K = len(-bp.history['objval'])

    fig, axs = plt.subplots(3, 1, sharex=True)

    axs[0].plot(bp.history['objval'])
    axs[0].set_ylabel('f(x^k) + g(z^k)')

    axs[1].plot(bp.history['r_norm'])
    axs[1].plot(bp.history['eps_pri'])
    axs[1].set_yscale('log')
    axs[1].set_ylabel('||r||_2')

    axs[2].plot(bp.history['s_norm'])
    axs[2].plot(bp.history['eps_dual'])
    axs[2].set_yscale('log')
    axs[2].set_ylabel('||s||_2')
    axs[2].set_xlabel('iter (k)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
