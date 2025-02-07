import torch
from cp_solver.torch_version.solver import TorchCpSolver


# build a QP problem:
# minimize    (1/2)x'Px + q'x
# subject to  Hx + b \in K,
# where x in R^n, b in R^m.
device = torch.device("cpu")
num_envs = 100
P = torch.tensor([[4.0, 1.0], [1.0, 2.0]], device=device).repeat(
    num_envs, 1, 1
)  # num_envs * 2 * 2
q = torch.tensor([1.0, 1.0], device=device).repeat(num_envs, 1)  # num_envs * 2

my_solver = TorchCpSolver(P=P, q=q, device=device)
my_solver.add_constraint(
    H=torch.tensor([[1.0, 2.0]]).repeat(num_envs, 1, 1),  # num_envs * 1 * 2
    b=torch.zeros(num_envs, 1),  # num_envs * 1
    a_min=-0.5,
    a_max=0.5,
    type="interval",
)
my_solver.add_constraint(
    H=torch.tensor([[1.0, 1.0]]).repeat(num_envs, 1, 1),  # num_envs * 1 * 2
    b=torch.zeros(num_envs, 1),  # num_envs * 1
    a_min=-1,
    a_max=1,
    type="interval",
)
x, *_ = my_solver.forward(iters=20)
x = x.squeeze(-1)

# qpth
from qpth.qp import QPFunction, QPSolvers

# Define QP function
qf = QPFunction(
    verbose=-1,
    check_Q_spd=False,
    eps=1e-10,
    solver=QPSolvers.PDIPM_BATCHED,
    maxIter=100,
)
# solve QP
# min 1/2 x'Px + q'x
# s.t. Ax - b = 0
#      Gx - h <= 0

e = torch.autograd.Variable(torch.Tensor())
G = torch.tensor(
    [[1.0, 2.0], [-1.0, -2.0], [1.0, 1.0], [-1.0, -1.0]], device=device
).repeat(
    num_envs, 1, 1
)  # num_envs * 4 * 2
h = torch.tensor([0.5, 0.5, 1, 1], device=device).repeat(num_envs, 1)  # num_envs * 4
qpth_x = qf(P.double(), q.double(), G.double(), h.double(), e, e)

# check the result
error = x - qpth_x
print(f"error: {error}")
print(f"norm error: {torch.norm(error)}")
