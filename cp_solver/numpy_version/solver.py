import numpy as np
from numpy.linalg import solve, inv, pinv, lstsq


class NumpyCpSolver:
    """
    Solve QP problem:
    minimize    (1/2)x'Px + q'x
    subject to  Hx + b \in K,
    where x in R^n, b in R^m.
    """

    def __init__(self, P: np.array, q: np.array, alpha=1.0, beta=1.0):
        """
        Initialize the QP solver.

        P: Matrix P in the objective
        q: Coefficients in the objective
        alpha, beta: conditioning parameters
        """
        self.n = P.shape[0]
        self.P = P
        self.q = q
        self.alpha = alpha
        self.beta = beta

        # Initialize the constraint
        self.m = 0
        self.H = np.zeros((0, self.n))
        self.b = np.zeros(0)

        self.constraint = []

    def add_constraint(
        self,
        H: np.array,
        b: np.array,
        a_min: float = None,
        a_max: float = None,
        type="soc",
    ):
        """
        Add a second-order cone constraint.

        H: Constraint matrix
        b: Constraint vector
        """
        if type in ["soc", "interval", "relu"]:
            self.constraint.append(
                {
                    "z_start": self.m,
                    "z_dim": b.shape[0],
                    "type": type,
                    "a_min": a_min,
                    "a_max": a_max,
                }
            )
            assert H.shape[1] == self.n and b.shape[0] == H.shape[0], print(
                f"shape mismatch: {H.shape}, {b.shape}"
            )
            self.m += b.shape[0]
            self.H = np.concatenate([self.H, H], axis=0)
            self.b = np.concatenate([self.b, b], axis=0)
        else:
            raise ValueError(f"Unsupported constraint type: {type}")

    def project(self, z: np.array):
        """
        Project the variable z onto the feasible set.

        z: z=Hx+b in the QP problem
        """
        for c in self.constraint:
            start_idx = c["z_start"]
            end_idx = start_idx + c["z_dim"]
            if c["type"] == "soc":
                z[start_idx:end_idx] = project_onto_soc(z[start_idx:end_idx])
            elif c["type"] == "interval":
                z[start_idx:end_idx] = np.clip(
                    z[start_idx:end_idx],
                    a_min=c["a_min"],
                    a_max=c["a_max"],
                )
            elif c["type"] == "relu":
                z[start_idx:end_idx] = np.maximum(0, z[start_idx:end_idx])
            else:
                raise ValueError(f"Unsupported constraint type: {c['type']}")
        return z

    def get_AB(self):
        """
        Given P, q, H, b, and the conditioning parameters, compute matrices A and B used in the iterations.
        A=[F beta * F; alpha*(I-2F) I-2*alpha*beta * F]
        B=[beta*mu; -2*alpha**beta*mu]
        """
        I = np.eye(self.m)
        F = I - self.H @ pinv(self.P + self.H.T @ self.H) @ self.H.T
        mu = bmv(F, bmv(self.H, pinv(self.P) @ self.q) - self.b)
        A = np.concatenate(
            [
                np.concatenate([F, self.beta * F], axis=1),
                np.concatenate(
                    [
                        self.alpha * (I - 2 * F),
                        I - 2 * self.alpha * self.beta * F,
                    ],
                    axis=1,
                ),
            ],
            axis=0,
        )  # num_envs * 2m * 2m
        B = np.concatenate(
            [self.beta * mu, -2 * self.alpha * self.beta * mu]
        )  # num_envs * 2m
        return A, B

    def forward(self, warm_lambda_z=None, iters=100, return_residuals=False):
        """
        Solve the QP problem using Chambolle-Pock method.

        warm_lambda_z: Initial guess for the variable (lambda, z)
        iters: Number of iterations
        return_residuals: Whether to return the residuals
        """
        if warm_lambda_z is None:
            lambda_z = np.zeros((2 * self.m))
        else:
            lambda_z = warm_lambda_z

        A, B = self.get_AB()
        for _ in range(iters):
            lambda_z = bmv(A, lambda_z) + B
            lambda_z[self.m :] = self.project(lambda_z[self.m :])

        # get primal solution
        x = pinv(self.H) @ (lambda_z[self.m :] - self.b)

        if return_residuals:
            primal_residual, dual_residual = self.compute_residuals(
                x, lambda_z, lambda_z[: self.m]
            )
            return x, lambda_z, primal_residual, dual_residual
        else:
            return x, lambda_z, None, None

    def forward_each_iter(self, warm_lambda_z=None, iters=100, return_residuals=False):
        """
        Solve the QP problem using Chambolle-Pock method.

        warm_lambda_z: Initial guess for the variable (lambda, z)
        iters: Number of iterations
        return_residuals: Whether to return the residuals
        """
        if warm_lambda_z is None:
            lambda_z = np.zeros((2 * self.m))
        else:
            lambda_z = warm_lambda_z
        A, B = self.get_AB()

        x_history = np.zeros((iters, self.n))
        lambda_z_history = np.zeros((iters, 2 * self.m))
        primal_residual_history = np.zeros(iters)
        dual_residual_history = np.zeros(iters)
        for k in range(iters):
            lambda_z = bmv(A, lambda_z) + B
            lambda_z[self.m :] = self.project(lambda_z[self.m :])

            # get primal solution
            x = bmv(pinv(self.H), lambda_z[self.m :] - self.b)

            primal_residual, dual_residual = self.compute_residuals(
                x, lambda_z[self.m :], lambda_z[: self.m]
            )
            x_history[k] = x
            lambda_z_history[k] = lambda_z
            primal_residual_history[k] = primal_residual
            dual_residual_history[k] = dual_residual

        return (
            x_history,
            lambda_z_history,
            primal_residual_history,
            dual_residual_history,
        )

    def compute_residuals(self, x, z, lam):
        # Compute primal residual: Hx + b - z
        primal_residual = bmv(self.H, x) + self.b - z

        # Compute dual residual: Px + q - H'u
        dual_residual = bmv(self.P, x) + self.q - bmv(self.H.T, lam)

        return np.linalg.norm(primal_residual), np.linalg.norm(dual_residual)


def bmv(A, b):
    """Compute matrix multiply vector in batch mode."""
    return np.matmul(A, b)


def project_onto_soc(z_part: np.array):
    """
    Project the variable z onto the second-order cone.
    z_part: The part of z corresponding to the second-order cone
    """
    if z_part[2] < 0:
        return np.zeros(3)
    norm_xy = np.linalg.norm(z_part[:2]) + 1e-6

    if z_part[2] >= norm_xy:
        return z_part
    ratio = np.abs(z_part[2]) / norm_xy
    z_part[:2] *= ratio
    z_part[2] = norm_xy * ratio
    return z_part
