# The Chambolle-Pock Solver

![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

The Chambolle-Pock solver is a numerical optimization package for solving problems in the form
```math
\text{minimize}\quad        0.5 x' P x + q' x\\

\text{subject to}\quad      Hx + b \in \mathcal{K}, 
```
where $\mathcal{K}$ is a convex set.

We have implemented the following constraints:
1. Second-order cone (SOC) constraint:
```math
    \{ (x, t) | ||x||_2 <= t \}
```
2. Interval constraint:
```math
    \{ x | a_{\min} <= x <= a_{\max} \}
```
3. ReLU constraint:
```math
    \{ x | x >= 0 \}
```

The solver has two versions: 
* NumPy version: This version is implemented in Python using NumPy. It is suitable for a single convex QP problem.
* Torch version: This version is implemented in PyTorch, which is tailored for massive parallel computation on GPUs.


## Getting Started
```python
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
my_solver.add_constraint(
    H=np.array([[1.0, 1.0]]), b=np.array([0.0]), a_min=-1, a_max=1, type="relu"
)
x, *_ = my_solver.forward(iters=20)
```