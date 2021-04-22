import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import lstsq
from scipy.sparse import random as sprandn
from scipy.sparse import spdiags

import basis_pursuit


class GroupLasso(basis_pursuit._ADMM):
    """
    group_lasso  Solve group lasso problem via ADMM

    solves the following problem via ADMM:

    minimize 1/2*|| Ax - b ||_2^2 + \lambda sum(norm(x_i))

    The input p is a K-element vector giving the block sizes n_i, so that x_i
    is in R^{n_i}.

    The solution is returned in the vector x.

    history is a structure that contains the objective value, the primal and
    dual residual norms, and the tolerances for the primal and dual residual
    norms at each iteration.

    More information can be found in the paper linked at:
    http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
    """

    def __init__(self, lam: float,p, rho: float, alpha: float, quiet: bool = True, max_iter=1000, abstol=1e-4,
                 reltol=1e-2):
        """

        :param rho: augmented lagrangian parameter
        :param.alpha: typical values betwen 1.0 and 1.8
        """
        self.lam = lam
        self.p=p
        super().__init__(rho, alpha, quiet, max_iter, abstol, reltol)

    def fit(self, A, b):
        m, n = A.shape
        Atb = A.T @ b
        x = np.zeros((n, 1))
        z = np.zeros((n, 1))
        w = np.zeros((n, 1))

        cum_part = np.cumsum(self.p)

        # if not self.quiet:
        #    f'3s\t10s\t10s\t10s\t10s\t10s\n', 'iter', ...
        #     'r np.linalg.norm', 'eps pri', 's np.linalg.norm', 'eps dual', 'objective')

        history = {
            'objval': [],
            'r_norm': [],
            's_norm': [],
            'eps_pri': [],
            'eps_dual': []
        }

        M=np.linalg.inv(A.T@A+np.eye(A.shape[1]))

        for k in range(self.max_iter):
            # x - update
            x=M@(Atb+self.rho*(z-w))

            # z-update with relaxation
            z_old=z
            x_hat = self.alpha * x + (1 - self.alpha) * z_old
            for i,start_ind in enumerate(cum_part):
                z[start_ind:cum_part[i+1]] = self.shrinkage(x_hat[start_ind:cum_part[i+1]] + w[start_ind:cum_part[i+1]], self.lam /self.rho)

            # u - update
            w = w + (x - z)

            # diagnostics, reporting, termination checks
            history['objval'].append(self.objective(A, b, self.lam, cum_part, x, z))

            history['r_norm'].append(np.linalg.norm(x - z))
            history['s_norm'].append(np.linalg.norm(-self.rho * (z - z_old)))

            history['eps_pri'].append(
                np.sqrt(n) * self.abstol + self.reltol * max(np.linalg.norm(x), np.linalg.norm(-z)))
            history['eps_dual'].append(np.sqrt(n) * self.abstol + self.reltol * np.linalg.norm(self.rho * w))

            if history['r_norm'][-1] < history['eps_pri'][-1] and history['s_norm'][-1] < history['eps_dual'][-1]:
                break

        self.history = history
        return z

    def objective(self, A, b, lam,cum_part, x, z):
        obj = 0
        for i,start_ind in enumerate(cum_part):
            obj = obj + np.linalg.norm(z[start_ind:cum_part[i+1]])
        p = (1 / 2 * sum((A * x - b)**2) + self.lam * obj)
        return p

    def shrinkage(self, a, kappa):
        """

        :param a:
        :param kappa:
        """
        out = ((1 - kappa / np.linalg.norm(a))>0) @ a
        return out


def main():
    n = 150
    p = 500
    sparsity = 0.05
    x0 = sprandn(p, 1, sparsity)
    A = np.random.rand(n, p)
    A = A @ spdiags(1 / np.sqrt(np.sum(A ** 2, axis=0)), 0, p, p)
    b = A @ x0

    lam = 0.1
    rho = 0.05
    partition = np.random.randint(50, size=(5, 1))
    lasso = GroupLasso(lam, 1, 1, max_iter=100)
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
    print()


if __name__ == "__main__":
    main()