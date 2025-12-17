"""
Solve the generalized Nash equilibrium problem described in [1, Fig. 6], originally proposed in [2, Section 5] for n=20 agents.

[1] F. Fabiani and A. Bemporad, “An active learning method for solving competitive multi-agent decision-making and control problems,” 2024, http://arxiv.org/abs/2212.12561. 

[2] F. Salehisadaghiani, W. Shi, and L. Pavel, “An ADMM approach to the problem of distributed Nash equilibrium seeking.” CoRR, 2017.

(C) 2025 A. Bemporad
"""
import numpy as np
import jax
import jax.numpy as jnp
import time
from functools import partial
from jax_gnep import GNEP

t0 = time.time()

N = 20  # number of agents
sizes = [1]*N  # n agents of dimension 1
nvar = np.sum(sizes)

@jax.jit
def cost(x, i):
    # Cost function minimized by agent #i, i=0,...,n-1
    ci = N*(1.+i/2.)
    return ci*x[i]-x[i]*(60.*N-jnp.sum(x))

f = [partial(cost, i=i) for i in range(N)]

lb = 7. * np.ones(nvar)
ub = 100. * np.ones(nvar)

gnep = GNEP(sizes, f=f, lb=lb, ub=ub)

x0 = jnp.zeros(nvar)
print("Solving GNEP with N =", N, "agents ... ", end="")
x_star, lam_star, residual, opt = gnep.solve(x0)
print("done.")

print("=== GNE solution ===")
print(f"x = {np.array2string(x_star, precision=8)}")
for i in range(gnep.N):
    print(f"lambda[{i}] = {np.array2string(lam_star[i], precision=8)}")

print(f"KKT residual norm = {float(jnp.linalg.norm(residual)): 10.7g}")
print(f"LM iterations     = {int(opt.state.iter_num): 3d}")
print(f"Elapsed time: {time.time() - t0: .2f} seconds")
