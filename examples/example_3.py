"""
(C) 2025 A. Bemporad
"""
import numpy as np
import jax
import jax.numpy as jnp
import time
from jax_gnep import GNEP

t0 = time.time()

N = 2  # number of agents
sizes = [10,10]  # 2 agents of dimension 10
nvar = np.sum(sizes)

@jax.jit
def f1(x):
    return jnp.sum((x[:sizes[0]]+x[sizes[0]:])**2)+jnp.sum(x**2)

@jax.jit
def f2(x):
    return jnp.sum((x[:sizes[0]]+x[sizes[0]:]-10.)**2)+jnp.sum(x**2)

f = [f1, f2]  # agents' objectives

# Linear inequality constraints A*x<=b
A = np.vstack((np.ones((1, nvar)), -np.ones((1, nvar))))
b = np.array([15., 15.])
@jax.jit
def g(x):
    gx = A @ x - b
    return gx
    
lb = -3. * np.ones(nvar)
ub = 3. * np.ones(nvar)

gnep = GNEP(sizes, f=f, g=g, ng=2, lb=lb, ub=ub)

x0 = jnp.zeros(nvar)
x_star, lam_star, residual, opt = gnep.solve(x0)

print("=== GNE solution ===")
print(f"x = {np.array2string(x_star, precision=8)}")
for i in range(gnep.N):
    print(f"lambda[{i}] = {np.array2string(lam_star[i], precision=8)}")

print(f"KKT residual norm = {float(jnp.linalg.norm(residual)): 10.7g}")
print(f"LM iterations     = {int(opt.state.iter_num): 3d}")
print(f"Elapsed time: {time.time() - t0: .2f} seconds")
