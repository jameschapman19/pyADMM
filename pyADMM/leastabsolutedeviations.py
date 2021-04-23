import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from numpy.linalg import lstsq

import basis_pursuit


class LeastAbsoluteDeviations(basis_pursuit._ADMM):
    """
    lasso  Solve lasso problem via ADMM

    Solves the following problem via ADMM:

    minimize 1/2*|| Ax - b ||_2^2 + \lambda || x ||_1

    PORTED FROM https://web.stanford.edu/~boyd/admm.html

    The solution is returned in the vector x.

    history is a structure that contains the objective value, the primal and
    dual residual norms, and the tolerances for the primal and dual residual
    norms at each iteration.

    More information can be found in the paper linked at:
    http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
    """

    def __init__(self, rho: float, alpha: float, quiet: bool = True, max_iter=1000,
                 abstol=1e-4,
                 reltol=1e-2):
        """

        :param rho: augmented lagrangian parameter
        :param.alpha: typical values betwen 1.0 and 1.8
        """
        super().__init__(rho, alpha, quiet, max_iter, abstol, reltol)

    def fit(self, A, b):
        x, history = _fit(A, b, self.rho, self.alpha, self.abstol, self.reltol, self.max_iter)
        self.history = {
            'objval': history[0],
            'r_norm': history[1],
            's_norm': history[2],
            'eps_pri': history[3],
            'eps_dual': history[4]
        }
        return x


@jit(nopython=True, cache=True)
def _fit(A, b, rho, alpha, abstol, reltol, max_iter):
    n, p = A.shape
    x = np.zeros((p, 1))
    z = np.zeros((n, 1))
    w = np.zeros((n, 1))

    history = np.zeros((5, max_iter))

    Atb = A.T @ b

    invA = np.linalg.pinv(A.T @ A + rho * np.eye(p))

    for k in range(max_iter):
        # x - update
        q = A.T@(b+z-w)
        x = invA @ q

        # z-update with relaxation
        z_old = z
        Ax_hat = alpha * A @ x + (1 - alpha) * (z_old + b)
        z = _shrinkage(Ax_hat - b + w, 1 / rho)

        # u - update
        w = w + (Ax_hat - z - b)

        # diagnostics, reporting, termination checks
        history[0, k] = _objective(z)
        history[1, k] = np.linalg.norm(A @ x - z - b)
        history[2, k] = np.linalg.norm(-rho * A.T @ (z - z_old))
        history[3, k] = np.sqrt(n) * abstol + reltol * np.max(np.array([np.linalg.norm(A @ x), np.linalg.norm(-z), np.linalg.norm(b)]))
        history[4, k] = np.sqrt(p) * abstol + reltol * np.linalg.norm(rho * A.T @ w)
        if history[1][k] < history[3][k] and history[2][k] < history[4][k]:
            break
    return x, history


@jit(nopython=True, cache=True)
def _objective(z):
    return np.linalg.norm(z, ord=1)


@jit(nopython=True, cache=True)
def _shrinkage(a, kappa):
    """

    :param a:
    :param kappa:
    """
    return np.sign(a) * np.maximum(np.abs(a) - kappa, 0.)


def main():
    n = 1000
    p = 100

    A = np.random.randn(n, p)
    x = 10 * np.random.randn(p, 1)
    b = A @ x
    idx = np.random.choice(n, size=int(n / 50),replace=False)
    b[idx] = b[idx] + 1e2 * np.random.randn(len(idx),1)

    lasso = LeastAbsoluteDeviations(1, 1, max_iter=100)
    x = lasso.fit(A, b)

    fig, axs = plt.subplots(3, 1, sharex=True)

    axs[0].plot(lasso.history['objval'])
    axs[0].set_ylabel('f(x^k) + g(z^k)')

    axs[1].plot(lasso.history['r_norm'])
    axs[1].plot(lasso.history['eps_pri'])
    axs[1].set_yscale('log')
    axs[1].set_ylabel('||r||_2')

    axs[2].plot(lasso.history['s_norm'])
    axs[2].plot(lasso.history['eps_dual'])
    axs[2].set_yscale('log')
    axs[2].set_ylabel('||s||_2')
    axs[2].set_xlabel('iter (k)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
