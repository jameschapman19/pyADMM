import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from numpy.linalg import lstsq
from scipy.sparse import spdiags
from numpy.linalg import cholesky
import basis_pursuit
from numpy.linalg import solve
from scipy import sparse


class Huber(basis_pursuit._ADMM):
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

    def __init__(self, rho: float, alpha: float, transition_point=1, quiet: bool = True, max_iter=1000,
                 abstol=1e-4,
                 reltol=1e-2):
        """

        :param rho: augmented lagrangian parameter
        :param.alpha: typical values betwen 1.0 and 1.8
        """
        self.transition_point = transition_point
        super().__init__(rho, alpha, quiet, max_iter, abstol, reltol)

    def fit(self, A, b):
        x, history = _fit(A, b, self.transition_point, self.rho, self.alpha, self.abstol, self.reltol, self.max_iter)
        self.history = {
            'objval': history[0],
            'r_norm': history[1],
            's_norm': history[2],
            'eps_pri': history[3],
            'eps_dual': history[4]
        }
        return x


@jit(nopython=True, cache=True)
def _fit(A, b, transition_point, rho, alpha, abstol, reltol, max_iter):
    n, p = A.shape
    x = np.zeros((p, 1))
    z = np.zeros((n, 1))
    w = np.zeros((n, 1))

    history = np.zeros((5, max_iter))

    Atb = A.T @ b

    L, U = _factor(A, rho)

    for k in range(max_iter):
        # x - update
        q = Atb + A.T @ (z - w)
        x = solve(U, solve(L, q))

        # z-update with relaxation
        z_old = z
        Ax_hat = alpha * A @ x + (1 - alpha) * (z_old + b)
        tmp = Ax_hat - b + w
        z = rho / (1 + rho) * tmp + 1 / (1 + rho) * _shrinkage(tmp, 1 + 1 / rho)

        # u - update
        w = w + (Ax_hat - z - b)

        # diagnostics, reporting, termination checks
        history[0, k] = _objective(z, transition_point)
        history[1, k] = np.linalg.norm(A @ x - z - b)
        history[2, k] = np.linalg.norm(-rho * A.T @ (z - z_old))
        history[3, k] = np.sqrt(n) * abstol + reltol * np.max(
            np.array([np.linalg.norm(A @ x), np.linalg.norm(-z), np.linalg.norm(b)]))
        history[4, k] = np.sqrt(n) * abstol + reltol * np.linalg.norm(rho * w)
        if history[1][k] < history[3][k] and history[2][k] < history[4][k]:
            break
    return x, history


@jit(nopython=True, cache=True)
def _objective(z, transition_point):
    l1 = np.sum((np.abs(z) < transition_point) * np.abs(z))
    l2 = np.sum((np.abs(z) > transition_point) * np.linalg.norm(z) ** 2)
    return l1 + l2


@jit(nopython=True, cache=True)
def _shrinkage(a, kappa):
    """

    :param a:
    :param kappa:
    """
    return np.sign(a) * np.maximum(np.abs(a) - kappa, 0.)


@jit(nopython=True, cache=True)
def _factor(A, rho):
    """

    :param A:
    :param kappa:
    """
    n, p = A.shape
    if n >= p:
        L = cholesky(A.T.dot(A) + rho * np.eye(p))
    else:
        L = cholesky(np.eye(n) + 1 / rho * (A @ A.T))
    # L = sparse.csc_matrix(L)
    # U = sparse.csc_matrix(L.T)
    return np.asarray(L), np.asarray(L.T)


def main():
    n = 5000
    p = 200
    x = np.random.randn(p, 1)
    A = np.random.randn(n, p)
    A = A @ spdiags(1 / np.sqrt(np.sum(A ** 2, axis=0)), 0, p, p)
    b = A @ x + np.sqrt(0.01) * np.random.randn(n, 1)
    b = b + 10 * sparse.rand(n, 1, 200 / n)

    x_true = x

    lasso = Huber(1, 1, max_iter=100)
    x = lasso.fit(A, np.asarray(b))

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
