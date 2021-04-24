from time import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import lstsq
from numpy.linalg import solve
from numba import jit
import basis_pursuit


class TotalVariation(basis_pursuit._ADMM):
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

    def __init__(self, lam: float, rho: float, alpha: float, quiet: bool = True, max_iter=1000, abstol=1e-4,
                 reltol=1e-2):
        """

        :param rho: augmented lagrangian parameter
        :param.alpha: typical values betwen 1.0 and 1.8
        """
        self.lam = lam
        super().__init__(rho, alpha, quiet, max_iter, abstol, reltol)

    def fit(self, b):
        x, history = _fit(b, self.lam, self.rho, self.alpha, self.abstol, self.reltol, self.max_iter)
        self.history = {
            'objval': history[0],
            'r_norm': history[1],
            's_norm': history[2],
            'eps_pri': history[3],
            'eps_dual': history[4]
        }
        return x


@jit(cache=True)
def _fit(b, lam, rho, alpha, abstol, reltol, max_iter):
    n = b.shape[0]

    e = np.ones(n)
    D = np.diag(e)-np.diag(e,1)[:n,:n]
    x = np.zeros((n, 1))
    z = np.zeros((n, 1))
    w = np.zeros((n, 1))

    I = np.eye(n)
    DtD = D.T @ D

    history = np.zeros((5, max_iter))

    for k in range(max_iter):
        # x - update
        x = solve(I + rho * DtD, b + rho * D.T @ (z - w))

        # z-update with relaxation
        z_old = z.copy()
        Ax_hat = alpha * D @ x + (1 - alpha) * z_old
        z = _shrinkage(Ax_hat + w, lam / rho)

        # u - update
        w = w + (Ax_hat - z)

        # diagnostics, reporting, termination checks
        history[0, k] = _objective(b, lam, x, z)
        history[1, k] = np.linalg.norm(D @ x - z)
        history[2, k] = np.linalg.norm(-rho * D.T @ (z - z_old))
        history[3, k] = np.sqrt(n) * abstol + reltol * np.max(np.array([np.linalg.norm(D @ x), np.linalg.norm(-z)]))
        history[4, k] = np.sqrt(n) * abstol + reltol * np.linalg.norm(rho * D.T @ w)
        if history[1][k] < history[3][k] and history[2][k] < history[4][k]:
            break
    return x, history


@jit(nopython=True, cache=True)
def _objective(b, lam, x, z):
    return 1 / 2 * np.sum((x - b) ** 2) + lam * np.linalg.norm(z, ord=1)


@jit(nopython=True, cache=True)
def _shrinkage(a, kappa):
    """

    :param a:
    :param kappa:
    """
    return np.maximum(0, a - kappa) - np.maximum(0, -a - kappa)


def main():
    n = 100
    x = np.ones((n, 1))
    for j in range(3):
        idx = np.random.randint(n, size=1)
        k = np.random.randint(10, size=1)
        x[int(idx / 2):int(idx)] = k * x[int(idx / 2):int(idx)]

    x_true=x
    b = x + np.random.randn(n, 1)

    lam = 5

    lasso = TotalVariation(lam, 1, 1, max_iter=100)
    x = lasso.fit(b)
    t0 = time()
    x = lasso.fit(b)
    print(time() - t0)

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
