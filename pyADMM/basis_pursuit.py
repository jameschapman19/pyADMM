import numpy as np
from scipy.sparse import random as sprandn
import matplotlib.pyplot as plt
from abc import abstractmethod

class _ADMM:
    def __init__(self, rho: float, alpha: float, quiet: bool = True, max_iter=1000, abstol=1e-4, reltol=1e-2):
        self.rho = rho
        self.alpha = alpha
        self.quiet = quiet
        self.max_iter = max_iter
        self.abstol = abstol
        self.reltol = reltol
        pass

    @abstractmethod
    def objective(self, *args):
        pass


class BasisPursuit(_ADMM):
    """
    Solve basis pursuit via ADMM

    Solves the following problem via ADMM:

    minimize     ||x||_1
    subject to   Ax = b

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
        (m, n) = A.shape
        x = np.zeros((n, 1))
        z = np.zeros((n, 1))
        u = np.zeros((n, 1))

        # if not self.quiet:
        #    f'%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
        #     'r np.linalg.norm', 'eps pri', 's np.linalg.norm', 'eps dual', 'objective')

        # precompute static variables for x-update (projection on to Ax=b)
        AAt = A @ A.T
        P = np.eye(n) - A.T @ np.linalg.lstsq(AAt,A)[0]
        q = A.T @ np.linalg.lstsq(AAt, b)[0]

        history = {
            'objval': [],
            'r_norm': [],
            's_norm': [],
            'eps_pri': [],
            'eps_dual': []
        }

        for k in range(self.max_iter):
            # x-update
            x = P @ (z - u) + q

            # z-update with relaxation
            zold = z
            x_hat = self.alpha * x + (1 - self.alpha) * zold
            z = self.shrinkage(x_hat + u, 1 / self.rho)

            u = u + (x_hat - z)

            # diagnostics, reporting, termination checks
            history['objval'].append(self.objective(A, b, x))

            history['r_norm'].append(np.linalg.norm(x - z))
            history['s_norm'].append(np.linalg.norm(-self.rho * (z - zold)))

            history['eps_pri'].append(
                np.sqrt(n) * self.abstol + self.reltol * max(np.linalg.norm(x), np.linalg.norm(-z)))
            history['eps_dual'].append(np.sqrt(n) * self.abstol + self.reltol * np.linalg.norm(self.rho * u))

            if history['r_norm'][-1] < history['eps_pri'][-1] and history['s_norm'][-1] < history['eps_dual'][-1]:
                break
        self.history = history
        return z

        # if not self.quiet:
        #    fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
        #        history.r_norm(k), history.eps_pri(k), ...
        #        history.s_norm(k), history.eps_dual(k), history.objval(k))

        # if not self.quiet:
        # toc(t_start)

    def objective(self, A, b, x):
        """

        :param A:
        :param b:
        :param x:
        """
        return np.linalg.norm(x, ord=1)

    def shrinkage(self, a, kappa):
        """

        :param a:
        :param kappa:
        """
        diff = abs(a) - kappa
        diff[diff < 0] = 0
        out = np.sign(a) * diff
        return out

def main():
    n = 30
    m = 10
    A = np.random.rand(m, n)

    x = sprandn(n, 1, density=0.1)
    b = A * x

    xtrue = x

    bp=BasisPursuit(1,1)
    x=bp.fit(A,b)

    K = len(bp.history['objval'])

    fig,axs=plt.subplots(3,1,sharex=True)

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
    print()

if __name__ == "__main__":
    main()