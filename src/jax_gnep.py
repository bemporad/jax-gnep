"""
JAX-GNEP - Solving Generalized Nash Equilibrium Problems (GNEPs) via a Levenberg-Marquardt approach in JAX.

(C) 2025 Alberto Bemporad
"""

import numpy as np
import jax
import jax.numpy as jnp
from jaxopt import LevenbergMarquardt, ScipyBoundedMinimize

jax.config.update("jax_enable_x64", True)

class GNEP():
    def __init__(self, sizes, f, g=None, ng=None, lb=None, ub=None, Aeq=None, beq=None, variational=False):
        """
        Generalized Nash Equilibrium Problem (GNEP) solver via Levenberg-Marquardt method.
        
        We consider a GNEP with N agents, where agent i solves:
        
            min_{x_i} f_i(x)
            s.t. g(x) <= 0          (shared inequality constraints)
                 Aeq x = beq        (shared equality constraints)
                 lb <= x <= ub      (box constraints on x_i)
                 i= 1,...,N         (N = number of agents)
        
        The residuals of the KKT optimality conditions of all agents are minimized jointly as a 
        nonlinear least-squares problem, solved via a Levenberg-Marquardt method. Strict complementarity
        is enforced via the Fischer–Burmeister NCP function. Variational GNEs are also supported by simply imposing equal Lagrange multipliers.
                
        Parameters:
        -----------
        sizes : list of int
            List containing the number of variables for each agent.
        f : list of callables
            List of objective functions for each agent. Each function f[i](x) takes the full variable vector x as input.    
        g : callable, optional
            Shared inequality constraint function g(x) <= 0, common to all agents.
        ng : int, optional
            Number of shared inequality constraints. Required if g is provided.
        lb : array-like, optional
            Lower bounds for the variables. If None, no lower bounds are applied.
        ub : array-like, optional
            Upper bounds for the variables. If None, no upper bounds are applied.
        Aeq : array-like, optional
            Equality constraint matrix. If None, no equality constraints are applied.
        beq : array-like, optional
            Equality constraint vector. If None, no equality constraints are applied.
        variational : bool, optional
            If True, solve for a variational GNE by imposing equal Lagrange multipliers.
            
        (C) 2025 Alberto Bemporad
        """
        
        self.sizes = sizes
        self.N=len(sizes) # number of agents
        self.nvar=sum(sizes) # number of variables
        self.i2 = np.cumsum(sizes) # x_i = x(i1[i]:i2[i])
        self.i1 = np.hstack((0,self.i2[:-1]))
        if len(f) != self.N:
            raise ValueError(f"List of functions f must contain {self.N} elements, you provided {len(f)}.")
        self.f = f
        self.g = g # shared constraints        
        self.ng = int(ng) if ng is not None else 0 # number of shared constraints, taken into account by all agents
        if self.ng>0 and g is None:
            raise ValueError("If ng>0, g must be provided.")

        if lb is None:
            lb = -np.inf * np.ones(self.nvar)
        if ub is None:
            ub =  np.inf * np.ones(self.nvar)
    
        # Make bounds JAX arrays
        self.lb = jnp.asarray(lb)
        self.ub = jnp.asarray(ub)

        # Use *integer indices* of bounded variables per agent
        self.lb_idx = []
        self.ub_idx = []
        self.nlb    = []
        self.nub    = []
        self.is_lower_bounded = []
        self.is_upper_bounded = []
        self.is_bounded = []

        for i in range(self.N):
            sl = slice(self.i1[i], self.i2[i])
            lb_mask = np.isfinite(lb[sl])
            ub_mask = np.isfinite(ub[sl])
            lb_idx_i = np.nonzero(lb_mask)[0]
            ub_idx_i = np.nonzero(ub_mask)[0]
            self.lb_idx.append(lb_idx_i)
            self.ub_idx.append(ub_idx_i)
            self.nlb.append(len(lb_idx_i))
            self.nub.append(len(ub_idx_i))
            self.is_lower_bounded.append(self.nlb[i]>0)
            self.is_upper_bounded.append(self.nub[i]>0)
            self.is_bounded.append(self.is_lower_bounded[i] or self.is_upper_bounded[i])
        
        if Aeq is not None:
            if beq is None:
                raise ValueError("If Aeq is provided, beq must also be provided.")
            if Aeq.shape[1] != self.nvar:
                raise ValueError(f"Aeq must have {self.nvar} columns.")
            if Aeq.shape[0] != beq.shape[0]:
                raise ValueError("Aeq and beq must have compatible dimensions.")
            self.Aeq = jnp.asarray(Aeq)
            self.beq = jnp.asarray(beq)
            self.neq = Aeq.shape[0]
        else:
            self.Aeq = None
            self.beq = None
            self.neq = 0
        self.has_eq = self.neq > 0
        self.has_constraints = any(self.is_bounded) or (self.ng>0) or self.has_eq

        if variational:
            if self.ng ==0 and self.neq ==0:
                print("\033[1;31mVariational GNE requested but no shared constraints are defined.\033[0m")
                variational = False
        self.variational = variational

        n_shared = self.ng + self.neq 
        self.nlam = [int(self.nlb[i] + self.nub[i] + n_shared) for i in range(self.N)]  # Number of multipliers per agent           
        
        if not variational:
            self.nlam_sum = sum(self.nlam) # total number of multipliers
            i2_lam = np.cumsum(self.nlam)
            i1_lam = np.hstack((0, i2_lam[:-1]))
            self.ii_lam = [np.arange(i1_lam[i], i2_lam[i], dtype=int) for i in range(self.N)] # indices of multipliers for each agent            
        else:
            # all agents have the same multipliers for shared constraints
            self.ii_lam = []
            j = n_shared
            for i in range(self.N):
                self.ii_lam.append(np.hstack((np.arange(self.ng, dtype=int), # shared inequality-multipliers
                    np.arange(j, j + self.nlb[i] + self.nub[i], dtype=int), # agent-specific box multipliers
                    np.arange(self.ng, self.ng + self.neq, dtype=int)))) # shared equality-multipliers
                j += self.nlb[i] + self.nub[i]
            self.nlam_sum = n_shared + sum([self.nlb[i] + self.nub[i] for i in range(self.N)])
            
        # Gradients of agents' objectives
        self.df = [
            jax.jit(
                jax.grad(
                    lambda xi, x, i=i: self.f[i](
                        x.at[self.i1[i]:self.i2[i]].set(xi)
                    ),
                    argnums=0,
                )
            )
            for i in range(self.N)
        ]

        if self.ng>0:
            self.g  = jax.jit(self.g)
            self.dg = jax.jit(jax.jacobian(self.g))
        
    def _kkt_residual_impl(self, z):
        # pure JAX implementation, used inside JIT
        x   = z[:self.nvar]
        if self.has_constraints:
            lam = z[self.nvar:]

        res = []
        
        ng = self.ng
        if ng > 0:
            gx  = self.g(x)            # (ng,)
            dgx = self.dg(x)           # (ng, nvar)

        # primal feasibility for shared constraints
        neq = self.neq
        if ng > 0:
            res.append(jnp.maximum(gx, 0.0))
        if neq > 0:
            res.append(self.Aeq @ x - self.beq)

        is_bounded = self.is_bounded
        is_lower_bounded = self.is_lower_bounded
        is_upper_bounded = self.is_upper_bounded
        neq = self.neq
        
        for i in range(self.N):
            i1 = int(self.i1[i])
            i2 = int(self.i2[i])
            
            if is_bounded[i]:
                zero = jnp.zeros(self.sizes[i])
            if is_bounded[i] or ng>0 or neq>0: # we have inequality constraints
                nlam_i = self.nlam[i]
                lam_i = lam[self.ii_lam[i]]

            # 1st KKT condition
            res_1st = self.df[i](x[i1:i2], x)
            if ng>0:
                res_1st += dgx[:, i1:i2].T @ lam_i[:ng]
            if is_lower_bounded[i]:
                lb_idx_i = self.lb_idx[i]
                # Add -sum(e_i * lam_lb_i), e_i = unit vector                
                res_1st -= zero.at[lb_idx_i].set(lam_i[ng:ng + self.nlb[i]]) 
            if is_upper_bounded[i]:
                ub_idx_i = self.ub_idx[i]
                # Add sum(e_i * lam_ub_i)
                res_1st += zero.at[ub_idx_i].set(lam_i[ng + self.nlb[i]:ng + self.nlb[i] + self.nub[i]])
            if neq > 0:
                res_1st += self.Aeq[:, i1:i2].T @ lam_i[-neq:]
            res.append(res_1st)

            x_i = x[i1:i2]

            if is_bounded[i] or ng>0:
                # inequality constraints
                if ng>0:
                    g_parts = [gx]
                else:
                    g_parts = []
                if is_lower_bounded[i]:
                    g_parts.append(-x_i[lb_idx_i] + self.lb[i1:i2][lb_idx_i])
                if is_upper_bounded[i]:
                    g_parts.append( x_i[ub_idx_i] - self.ub[i1:i2][ub_idx_i])
                gix = jnp.concatenate(g_parts)

                # complimentary slackness
                # Use Fischer–Burmeister NCP function: min phi(a,b) = sqrt(a^2 + b^2) - (a + b)
                res.append(jnp.sqrt(lam_i[:nlam_i-neq]**2 + gix**2) - lam_i[:nlam_i-neq] + gix)
                #res.append(jnp.minimum(lam_i[:nlam_i-neq], -gix))
                #res.append(lam_i[:nlam_i-neq]*gix)
                
                # dual feasibility
                res.append(jnp.minimum(lam_i[:nlam_i-neq], 0.0)) 

        return jnp.concatenate(res)

    def _build_solver(self, maxiter, tol):
        # JIT the residual used by LM
        self._kkt_residual_jit = jax.jit(self._kkt_residual_impl)

        lm_solver = LevenbergMarquardt(
            self._kkt_residual_jit,
            maxiter=maxiter,
            tol=tol,
            verbose=0,
        )

        # Wrap run() in a JITted function that closes over lm_solver and self
        def run_fn(z0):
            return lm_solver.run(z0)
        
        self._lm_run = jax.jit(run_fn)
        
    # Public residual (can be called outside JIT, e.g. for diagnostics)
    def kkt_residual(self, z):
        return self._kkt_residual_jit(z)

    def solve(self, x0=None, maxiter=200, tol=1e-12):
        """ Solve the GNEP starting from initial guess x0.
        
        Parameters:
        -----------
        x0 : array-like or None
            Initial guess for the Nash equilibrium x.
        maxiter : int, optional
            Maximum number of Levenberg-Marquardt iterations.
        tol : float, optional
            Tolerance for convergence.
            
        Returns:
        --------
        x_star : ndarray
            Computed GNE solution (if one is found).
        lam_star : list of ndarrays
            List of Lagrange multipliers for each agent at the GNE solution.
        residual : ndarray
            KKT residual at the solution x_star.
        opt : OptimizeResult
            Full optimization result returned by the Levenberg-Marquardt solver.
        """
        
        # Pre-create & JIT the solver
        self._build_solver(maxiter, tol)

        if x0 is None:
            x0 = jnp.zeros(self.nvar)
        else:
            x0 = jnp.asarray(x0)            
        
        if self.has_constraints:
            lam0 = 0.1 * jnp.ones(self.nlam_sum)
            z0   = jnp.hstack((x0, lam0))
        else:           
            z0   = x0

        # First call pays compilation, subsequent calls are much faster
        opt    = self._lm_run(z0)
        z_star = opt.params
        x = z_star[:self.nvar]
        lam = []
        if self.has_constraints:
            lam_star = z_star[self.nvar:]
            for i in range(self.N):
                lam.append(np.asarray(lam_star[self.ii_lam[i]]))
        #res = self._kkt_residual_jit(z_star)
        res = opt.state.residual

        return np.asarray(x), lam, np.asarray(res), opt

    def best_response(self, i, x, rho=1e5, maxiter=200, tol=1e-8):
        """
        Compute best response for agent i via SciPy L-BFGS-B:

            min_{x_i} f_i(x_i, x_{-i}) + rho * (sum_j max(g_i(x), 0)^2 + ||Aeq x - beq||^2)
            s.t. lb_i <= x_i <= ub_i

        Parameters:
        -----------
        i : int
            Index of the agent for which to compute the best response.
        x : array-like
            Current joint strategy of all agents.
        rho : float, optional
            Penalty parameter for constraint violations.
        maxiter : int, optional
            Maximum number of L-BFGS-B iterations.
        tol : float, optional
            Tolerance used in L-BFGS-B optimization.
            
        Returns:
        x_i     : best response of agent i
        res     : SciPy optimize result
        """

        i1 = self.i1[i]
        i2 = self.i2[i]
        x = jnp.asarray(x)
        
        @jax.jit
        def fun(xi):
            # reconstruct full x with x_i replaced            
            x_i = x.at[i1:i2].set(xi)
            f = jnp.array(self.f[i](x_i)).reshape(-1)
            if self.ng > 0:
                f += rho*jnp.sum(jnp.maximum(self.g(x_i), 0.0)**2)
            if self.neq > 0:
                f += rho*jnp.sum((self.Aeq @ x_i - self.beq)**2)
            return f[0]

        li = self.lb[i1:i2]
        ui = self.ub[i1:i2]
        
        options = {'iprint': -1, 'maxls': 20, 'gtol': tol, 'eps': tol,
               'ftol': tol, 'maxfun': maxiter, 'maxcor': 10}
    
        solver = ScipyBoundedMinimize(
                fun=fun, tol=tol, method="L-BFGS-B", maxiter=maxiter, options=options)
        xi, state = solver.run(x[i1:i2], bounds=(li, ui))
        x_new = np.asarray(x.at[i1:i2].set(xi))
        fi_opt = self.f[i](x_new)
        iters = state.iter_num

        return x_new, fi_opt, iters

