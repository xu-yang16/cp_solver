import numpy as np

from cp_solver.numpy_version.solver import NumpyCpSolver


# build a QP problem:
# minimize    (1/2)x'Px + q'x
#    subject to  Hx + b \in K,
#    where x in R^n, b in R^m.
P = np.array([[4.0, 1.0], [1.0, 2.0]])
q = np.array([1.0, 1.0])

my_solver = NumpyCpSolver(P=P, q=q)
my_solver.add_constraint(
    H=np.array([[1.0, 2.0]]), b=np.array([0.0]), a_min=-0.5, a_max=0.5, type="interval"
)
my_solver.add_constraint(H=np.array([[-1.0, -1.0]]), b=np.array([0.0]), type="relu")
x, *_ = my_solver.forward(iters=20)

print(f"x from our numpy solver: {x[0]:.3f}, {x[1]:.3f}")

import osqp
from scipy import sparse

# Define problem data
P = sparse.csc_matrix([[4, 1], [1, 2]])
q = np.array([1, 1])
A = sparse.csc_matrix([[1, 2], [-1, -1]])
l = np.array([-0.5, 0.0])
u = np.array([0.5, np.inf])

# Create an OSQP object
prob = osqp.OSQP()

# Setup workspace and change alpha parameter
prob.setup(P, q, A, l, u, alpha=1.0, verbose=False)

# Solve problem
res = prob.solve()
print(f"x from OSQP: {res.x[0]:.3f}, {res.x[1]:.3f}")
