import numpy as np
import jax
import jax.numpy as jnp
from jax_gnep import GNEP
from functools import partial

# Solve a Variational GNE problem with N agents:
np.random.seed(3)

N = 3  # number of agents
sizes = [2]*N # sizes of each agent
nvar = sum(sizes)

# Agents' objective functions:
f = []
Q = []
c = []
for i in range(len(sizes)):
    Qi = np.random.randn(nvar,nvar)
    Qi=Qi@Qi.T + 1.e-3*np.eye(nvar)
    ci = 10.*np.random.randn(nvar)
    f.append(jax.jit(partial(lambda x,Qi,ci: 0.5*x@Qi@x + ci@x, Qi=Qi, ci=ci)))
    Q.append(Qi)
    c.append(ci)

# Shared constraints:
ncon = 2*3  # number of inequality constraints
A = .5*np.random.randn(int(ncon/2),sum(sizes))
A = np.vstack((A, -A))
b = np.random.rand(ncon)
@jax.jit
def g(x): 
    return A@x-b

Aeq = np.ones(nvar).reshape(1,-1)
beq = np.array([1.0])

nvar = sum(sizes)
lb=-np.ones(nvar) # lower bounds
ub=np.ones(nvar) # upper bounds

# create GNEP object and solve for vGNE
gnep = GNEP(sizes, f=f, g=g, ng=ncon, lb=lb, ub=ub, Aeq=Aeq, beq=beq, variational=True)

x0 = jnp.zeros(nvar)
x_star_vgne, lam_star_vgne, residual_vgne, opt_vgne = gnep.solve(x0)

print("=== vGNE solution ===")
print(f"x = {np.array2string(x_star_vgne, precision=4)}")
for i in range(gnep.N):
    print(f"lambda[{i}] = {np.array2string(lam_star_vgne[i], precision=2)}")

print(f"KKT residual norm = {float(jnp.linalg.norm(residual_vgne)): 10.7g}")
print(f"LM iterations     = {int(opt_vgne.state.iter_num): 3d}")

# check best responses of all agents at the computed GNE
for i in range(gnep.N):
    x_br_vgne, fbr_opt_vgne, iters_vgne = gnep.best_response(i, x_star_vgne, rho=1.e8)
    print(f"Agent {i}'s BR at the GNE: ", end="")
    print(f"|x_br-x_star| = {jnp.linalg.norm(x_br_vgne-x_star_vgne): 10.2g}", end="")
    #print(f", fbr_opt = {fbr_opt_vgne: 10.7g}")
    print(f" [{iters_vgne: 2d} L-BFGS-B iter(s)]")
