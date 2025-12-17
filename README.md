# jax-gnep — A KKT-based Algorithm for Computing Generalized Nash Equilibria

This repository includes a numerical solver to solve nonlinear **Generalized Nash Equilibrium Problems (GNEPs)** based on **JAX**. The decision variables and Lagrange multipliers satisfying the KKT conditions jointly for all agents are determined by solving a nonlinear least-squares problem via a **Levenberg–Marquardt** method. If a zero residual is found, this corresponds to a potential generalized Nash equilibrium, a property that can be tested by evaluating the individual best responses. 

---
## Installation

~~~python
pip install jax-gnep
~~~

## Overview

We consider a game with $N$ agents. Each agent $i$ solves the following problem

$$
x_i^\star \in \arg\min_{x_i \in \mathbb{R}^{n_i}} f_i(x)
$$

subject to shared and local constraints

$$
g(x) \le 0, \qquad Ax = b, \qquad \ell \le x \le u.
$$

where:

- $f_i$'s are the agents' objectives, specified as JAX functions;
- $x = (x_1^\top \dots x_N^\top)^\top \in \mathbb{R}^n$;
- $g : \mathbb{R}^n \to \mathbb{R}^m$ encodes shared constraints (JAX function);
- $A, b$ define shared equality constraints;
- $\ell, u$ are local box constraints.

A **generalized Nash equilibrium** $x^\star$ is a vector where no agent can reduce their cost given the others' strategies and feasibility constraints, i.e.,

$$f_i(x^\star_{i}, x^\star_{-i})\leq f_i(x_i, x^\star_{-i})$$ 

for all feasible $x=(x_i,x_{-i}^\star)$, or equivalently, in terms of **best responses**: 

$$
x_i^\star \in \arg\min_{\ell_{i}\leq x_{i}\leq u_{i} \in \mathbb{R}^{n_{i}}} f_i(x)
$$

$$
\textrm{s.t.} \qquad g(x) \le 0, \qquad Ax = b, \qquad x_{-i}=x_{-i}^\star.
$$


---

## KKT Conditions

For each agent $i$, the necessary KKT conditions are:

**1. Stationarity**

$$ \nabla_{x_i} f_i(x) + \nabla_{x_i} g(x)^\top \lambda_i + A_i^\top \mu_i - v_i + y_i = 0 $$

**2. Primal Feasibility**

$$
g(x) \le 0, \qquad Ax = b, \qquad \ell \le x \le u
$$

**3. Dual Feasibility**

$$
\lambda_i \ge 0, \qquad v_i\geq 0, \qquad y_i\geq 0
$$

**4. Complementary Slackness**

$$
\lambda_{i,j} \, g_j(x) = 0
$$

$$
v_{i,k} \, (x_{i,k} - \ell_{i,k}) = 0
$$

$$
y_{i,k} \, (u_{i,k} - x_{i,k}) = 0
$$

In `jax_gnep`, primal feasibility (with respect to inequalities), dual feasibility, and complementary slackness conditions, which can be summarized as complementarity pairs $0\leq a\perp b\geq 0$, are enforced by using the nonlinear complementarity problem (NCP) Fischer–Burmeister function [1]

$$
\phi(a, b) = \sqrt{a^2 + b^2} - a - b
$$

which has the property

$$
\phi(a,b) = 0 \;\Longleftrightarrow\; a \ge 0,\; b \ge 0,\; ab = 0.
$$

Therefore, the above KKT conditions can be rewritten as the nonlinear system of equalities

$$R(z)=0$$

where $z = (x, \{\lambda_i\}, \{\mu_i\}, \{v_i\}, \{y_i\})$.  To find a solution, we solve the nonlinear least-squares problem

$$
   \min_z \frac{1}{2}\|R(z)\|^2
$$

using the ``LevenbergMarquardt`` function in `jaxopt` and exploiting JAX's autodiff to evaluate Jacobians.

After solving the nonlinear least-squares problem, if the residual $R(z^\star)=0$, we can check if it indeeds is a GNE by computing the best responses of each agent

$$ \min_{\ell_i\leq x_i\leq u_i} f_i(x_i, x^\star_{-i}) $$

$$ \textrm{s.t.} \qquad g_i(x), \qquad Ax=b$$

In `jax_gnep`, the best response of agent $i$ is computed by solving the following box-constrained nonlinear
programming problem with `L-BFGS-B`:

$$ \min_{x_i} f_i(x_i, x_{-i}) + \rho \left(\sum_j \max(g_i(x), 0)^2 + \|A x - b\|_2^2\right) $$
            
$$ \textrm{s.t.} \qquad \ell_i \leq x_i \leq u_i$$

with $x_{-i}=x^\star_{-i}$, where $\rho\gg 1$ is a large penalty on the violation of shared constraints.

---

## References

> [1] Alexander Fischer. *A special Newton-type optimization method.* **Optimization**, 24(3–4):269–284, 1992.

## Example

We want to solve a simple GNEP with 3 agents, $x_1\in\mathbb{R}^2$, $x_2\in\mathbb{R}$, $x_3\in\mathbb{R}$, defined as follows:

```python
import numpy as np
import jax
import jax.numpy as jnp
from jax_gnep import GNEP

sizes = [2, 1, 1]      # [n1, n2, n3]

# Agent 1 objective:
@jax.jit
def f1(x):
    return jnp.sum((x[0:2] - jnp.array([1.0, -0.5]))**2)

# Agent 2 objective:
@jax.jit
def f2(x):
    return (x[2] + 0.3)**2

# Agent 3 objective:
@jax.jit
def f3(x):
    return (x[3] - 0.5*(x[0] + x[2]))**2

# Shared constraint:
def g(x): 
    return jnp.array([x[3] + x[0] + x[2] - 2.0])

lb=np.zeros(4) # lower bounds
ub=np.ones(4) # upper bounds

gnep = GNEP(sizes, f=[f1,f2,f3], g=g, ng=1, lb=lb, ub=ub)
```

We call `solve()` to solve the problem defined above:

```python
x_star, lam_star, residual, opt = gnep.solve()
```

which gives the following solution:
```
x* = [ 1.00000000e+00 -1.05289340e-14 -2.23603233e-14  5.00000000e-01]
```
We can check if the KKT conditions are satisfied by looking at the residual $||R(x)||_2$:
```python
print(np.linalg.norm(residual))

8.265311429442589e-14
```
We can check if indeed $x^\star$ is an equilibrium by evaluating the agents' best responses:

```python
for i in range(gnep.N):
    x_br, fbr_opt, iters = gnep.best_response(i, x_star)
    print(x_br)
```

```
[ 1.00000000e+00  0.00000000e+00 -2.23603233e-14  5.00000000e-01]
[ 1.0000000e+00 -1.0528934e-14  0.0000000e+00  5.0000000e-01]
[ 1.00000000e+00 -1.05289340e-14 -2.23603233e-14  5.00000000e-01]
```

To add equality constraints, use the following:

```python
Aeq = np.array([[1,1,1,1]])
beq = np.array([2.0])

gnep = GNEP(sizes, f=[f1,f2,f3], g=g, ng=1, lb=lb, ub=ub, Aeq=Aeq, beq=beq)
```

You can also specify an initial guess $x_0$ to the GNEP solver as follows:
```python
x_star, lam_star, residual, opt = gnep.solve(x0)
```


### Citation

```
@misc{jax_gnep,
    author={A. Bemporad},
    title={{jax-gnep -- A {KKT}-based Algorithm for Computing Generalized {Nash} Equilibria}},
    howpublished = {\url{https://github.com/bemporad/jax-gnep}},
    year=2025
}
```

### License

Apache 2.0

(C) 2025 A. Bemporad