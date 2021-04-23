import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import lstsq
from scipy.sparse import random as sprandn
from scipy.sparse import spdiags
from numba import jit
import basis_pursuit
from numpy.linalg import cholesky


class GroupLasso(basis_pursuit._ADMM):
    """
    group_lasso  Solve group lasso problem via ADMM

    solves the following problem via ADMM:

    minimize 1/2*|| Ax - b ||_2^2 + \lambda sum(norm(x_i))

    PORTED FROM https://web.stanford.edu/~boyd/admm.html

    The input p is a K-element vector giving the block sizes n_i, so that x_i
    is in R^{n_i}.

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

    def fit(self, A, b, p):
        x, history = _fit(A, b, p, self.lam, self.rho, self.alpha, self.abstol, self.reltol, self.max_iter)
        self.history = {
            'objval': history[0],
            'r_norm': history[1],
            's_norm': history[2],
            'eps_pri': history[3],
            'eps_dual': history[4]
        }
        return x


@jit(nopython=True)
def _fit(A, b, partition, lam, rho, alpha, abstol, reltol, max_iter):
    n, p = A.shape
    Atb = A.T @ b
    x = np.zeros((p, 1))
    z = np.zeros((p, 1))
    w = np.zeros((p, 1))

    cum_part = np.cumsum(partition)

    history = np.zeros((5, max_iter))

    invA = np.linalg.pinv(A.T @ A + rho * np.eye(p))

    for k in range(max_iter):

        q = Atb + rho * (z - w)
        x = invA@q

        # z-update with relaxation
        z_old = z
        x_hat = alpha * x + (1 - alpha) * z_old
        for i, start_ind in enumerate(cum_part[:-1]):
            z[start_ind:cum_part[i + 1]] = _shrinkage(x_hat[start_ind:cum_part[i + 1]] + w[start_ind:cum_part[i + 1]], lam / rho)

        # u - update
        w = w + (x - z)

        # diagnostics, reporting, termination checks
        history[0, k] = _objective(A, b, lam, cum_part, x, z)
        history[1, k] = np.linalg.norm(x - z)
        history[2, k] = np.linalg.norm(-rho * (z - z_old))
        history[3, k] = np.sqrt(n) * abstol + reltol * np.max(np.array([np.linalg.norm(x), np.linalg.norm(-z)]))
        history[4, k] = np.sqrt(n) * abstol + reltol * np.linalg.norm(rho * w)
        if history[1][k] < history[3][k] and history[2][k] < history[4][k]:
            break
    return x, history


@jit(nopython=True)
def _objective(A, b, lam, cum_part, x, z):
    obj = 0
    for i, start_ind in enumerate(cum_part[:-1]):
        obj = obj + np.linalg.norm(z[start_ind:cum_part[i + 1]])
    p = (1 / 2 * np.sum((A @ x - b) ** 2) + lam * obj)
    return p


@jit(nopython=True)
def _shrinkage(a, kappa):
    """

    :param a:
    :param kappa:
    """
    out = a * np.maximum(np.linalg.norm(a) - kappa, 0.)/np.linalg.norm(a)
    return out


def main():
    n = 1500
    K = 200
    partition = np.random.randint(int(50), size=(K, 1))
    p = int(sum(partition))
    sparsity = 100 / p

    x = np.zeros((p, 1))
    cum_part = np.cumsum(partition)
    cum_part = np.concatenate(([0], cum_part))
    for i, start_ind in enumerate(cum_part[:-1]):
        x[start_ind:cum_part[i + 1]] = 0
        if np.random.randn() < sparsity:
            x[start_ind:cum_part[i + 1]] = np.random.randn(int(partition[i]), 1)

    A = np.random.randn(n, p)
    A = A @ spdiags(1 / np.sqrt(np.sum(A ** 2, axis=0)), 0, p, p)
    b = A @ x + np.sqrt(0.001) * np.random.randn(n, 1)

    lams = []
    for i, start_ind in enumerate(cum_part[:-1]):
        lams.append(np.linalg.norm(A[:, start_ind:cum_part[i + 1]].T @ b))

    lambda_max = max(lams)
    lam = 0.1 * lambda_max

    x_true = x

    lasso = GroupLasso(lam, 1, 1, max_iter=100)
    x = lasso.fit(A, b, partition)

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
