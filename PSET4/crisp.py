import numpy as np
import time
from qpsolvers import solve_qp
from scipy import sparse
from scipy.sparse import csr_matrix, eye, hstack, bmat, vstack

class SQPSolver:
    def __init__(self,
                 cost_fn, grad_cost_fn, hess_cost_fn,
                 eq_cons_fn, jac_eq_fn,
                 ineq_cons_fn, jac_in_fn,
                 x0,
                 mu_eq_init,
                 mu_in_init,
                 delta_init=1.0,
                 delta_max=10.0,
                 eta_low=0.25,
                 eta_high=0.75,
                 tol=1e-6,
                 problem_param=None,
                 max_iter=50):
        """
        cost_fn(x) -> scalar
        grad_cost_fn(x) -> (n,)
        hess_cost_fn(x) -> (n,n)

        eq_cons_fn(x) -> (m_eq,)
        jac_eq_fn(x)   -> (m_eq, n)

        ineq_cons_fn(x) -> (m_in,)
        jac_in_fn(x)    -> (m_in, n)
        """
        self.cost_fn      = cost_fn
        self.grad_cost_fn = grad_cost_fn
        self.hess_cost_fn = hess_cost_fn

        self.eq_cons_fn   = eq_cons_fn
        self.jac_eq_fn    = jac_eq_fn
        self.ineq_cons_fn = ineq_cons_fn
        self.jac_in_fn    = jac_in_fn
        self.problem_param = problem_param

        self.x = x0.copy()
        self.n = x0.size

        # penalty + trust‐region params
        self.mu_eq      = np.array(mu_eq_init)   # shape (m_eq,)
        self.mu_in      = np.array(mu_in_init)   # shape (m_in,)
        self.delta      = delta_init
        self.delta_max  = delta_max
        self.eta_low    = eta_low
        self.eta_high   = eta_high
        self.tol        = tol
        self.max_iter   = max_iter

        # history for debug / plot
        self.hist = {
            'cost': [], 'phi': [], 'delta': [],
            'eq_violation': [], 'ineq_violation': [], 'solve_time': []
        }

    def _merit(self, cost, c_eq, c_in):
        """
        phi = cost
            + sum_i mu_eq[i] * |c_eq[i]|
            + sum_j mu_in[j] * max(0, -c_in[j])
        """
        eq_term   = np.sum(self.mu_eq * np.abs(c_eq))
        in_term   = np.sum(self.mu_in * np.maximum(0., -c_in))
        return cost + eq_term + in_term
    
    def solve(self):
        mu_current = 1
        for k in range(self.max_iter):
            xk = self.x

            # eval cost + grads + constraints
            fk     = self.cost_fn(xk)
            gk     = self.grad_cost_fn(xk)
            Hk     = self.hess_cost_fn(xk)

            ceq    = self.eq_cons_fn(xk)
            Jeq    = self.jac_eq_fn(xk)
            cin    = self.ineq_cons_fn(xk)
            Jin    = self.jac_in_fn(xk)

            phi_k  = self._merit(fk, ceq, cin)
            
            m_eq, _ = Jeq.shape
            m_in, _ = Jin.shape

            # build QP subproblem:
            #   min 1/2 p^T Hk p + gk^T p + mu*(|Jeq p + ceq|_1 + max(0, -(Jin p + cin))_1)
            # linearize constraints: Jeq p + ceq = 0,   Jin p + cin >= 0
            # introduce slack for L1 penalization or fold into objective
            
            # identity & zero blocks as sparse
            LICQ_constraint_num = 0
            # LICQ_constraint_num = self.problem_param["nv"] * 2 * self.problem_param["num_steps"]

            m_eq_slack = m_eq - LICQ_constraint_num
            m_in_slack = m_in
            Z_top_eq = csr_matrix((LICQ_constraint_num, m_eq_slack))
            I_bot_eq = eye(m_eq_slack, format='csr')
            I_eq = vstack([Z_top_eq, I_bot_eq], format='csr') # (m_eq, m_eq_slack)
            I_in = eye(m_in_slack, format='csr')    # (m_in, m_in_slack)
            Z_eq_in = csr_matrix((m_eq, m_in_slack))# zeros (m_eq, m_in_slack)
            Z_in_eq = csr_matrix((m_in, m_eq_slack))# zeros (m_in, m_eq_slack)

            # equality constraints: Jeq p + w - v = -ceq
            #   A_eq [p; w; v; t] = b_eq
            A_eq = hstack([
            Jeq,        # (m_eq, n)
                I_eq,          # (m_eq, m_eq_slack)  slack w
            -I_eq,          # (m_eq, m_eq_slack)  slack v
                Z_eq_in        # (m_eq, m_in_slack)  no t here
            ], format='csr')
            b_eq = -ceq        # still a dense vector of length m_eq

            # inequality constraints: Jin p + t >= -cin  (t >= 0)
            #   A_in [p; w; v; t] >= b_in
            A_in = hstack([
                Jin,        # (m_in, n)
                Z_in_eq,       # (m_in, m_eq_slack)
                Z_in_eq,       # (m_in, m_eq_slack)
                I_in           # (m_in, m_in_slack)  slack t
            ], format='csr')
            b_in = -cin       # dense vector of length m_in

            # QP vars = [p (n,) ; s (m_eq+m_in,)] where s are slacks for ineq
            m_tot  = 2 * m_eq_slack + m_in_slack
            
            zero_n_m = csr_matrix((self.n, m_tot))      # shape (n, m_tot)
            zero_m_n = csr_matrix((m_tot, self.n))      # shape (m_tot, n)
            zero_m_m = csr_matrix((m_tot, m_tot))      # shape (m_tot, m_tot)

            P = bmat(
                [[Hk,        zero_n_m],
                [zero_m_n,  zero_m_m]],
                format='csr'   
            )
            
            q = np.hstack([ gk, self.mu_eq[LICQ_constraint_num:], self.mu_eq[LICQ_constraint_num:], self.mu_in])
            # trust‐region: ||p||_∞ ≤ delta  → bound constraints on p
            lb = np.hstack([ -self.delta * np.ones(self.n),
                              np.zeros(m_tot) ])
            ub = np.hstack([  self.delta * np.ones(self.n),
                               np.inf * np.ones(m_tot) ])
            # solve QP
            t0 = time.time()
            sol = solve_qp(P, q, G=-A_in, h=-b_in, A=A_eq, b=b_eq, lb=lb, ub=ub, solver='piqp')
            solve_dt = time.time() - t0
            if sol is None:
                raise RuntimeError("QP subproblem failed at iter %d" % k)
            p = sol[:self.n]

            # candidate
            x_new = xk + p
            cost_new = self.cost_fn(x_new)
            ceq_new = self.eq_cons_fn(x_new)
            cin_new = self.ineq_cons_fn(x_new)
            phi_new = self._merit(cost_new, ceq_new, cin_new)

            # actual / predicted reduction
            ared = phi_k - phi_new
            pred = -p.dot(gk) - 0.5*p.dot(Hk.dot(p)) +  np.sum(self.mu_eq * (np.abs(ceq) - np.abs(ceq + Jeq.dot(p)))) + np.sum(self.mu_in * (np.maximum(0,-cin) - np.maximum(0, -(cin + Jin.dot(p))))) 

            rho = ared / (pred + 1e-12)
    
            if rho < 0 or ared < 1e-9:
                self.delta *= 0.25
                print('REJ   ', end='')
            elif rho < self.eta_low:
                self.delta *= 0.25
                self.x = x_new
                print('TR-   ', end='')
            elif rho > self.eta_high and np.max(np.abs(p)) > 0.95 * self.delta:
                self.delta = min(2 * self.delta, self.delta_max)
                self.x = x_new
                print('TR+   ', end='')
            else:
                self.x = x_new
                print('TR    ', end='')

            # print predicted, actual, cost, merit, eq_vio, in_vio
            print(f"predicted = {pred:.6f} , "
                f"actual = {ared:.6f} , "
                f"cost = {fk:.6f} , "
                f"merit = {phi_k:.6f} , "
                f"eq_vio = {np.max(np.abs(ceq)):.6f} , "
                f"in_vio = {np.max(np.maximum(0., -cin)):.6f} , "
                f"QP_time = {solve_dt:.6f}s")

            # # update penalty if needed
            # if np.max(np.abs(ceq_new)) > 0.1*self.tol or np.max(-cin_new) > 0.1*self.tol:
            #     self.mu *= 10

            # record history
            self.hist['solve_time'].append(solve_dt)
            self.hist['cost'].append(fk)
            self.hist['phi'].append(phi_k)
            self.hist['delta'].append(self.delta)
            self.hist['eq_violation'].append(np.max(np.abs(ceq)))
            self.hist['ineq_violation'].append(np.max(np.maximum(0., -cin)))

            # check convergence
            if np.linalg.norm(p) < self.tol \
               and np.max(np.abs(ceq_new)) < self.tol \
               and np.max(np.maximum(0., -cin_new)) < self.tol:
                print(f"Solved at iter {k}")
                return self.x, self.hist
            
            if np.abs(ared) < self.tol:
                print(f"Increase lower than tolerance at iter {k}")
            else:
                continue
            
            if mu_current < 1e3:
                self.mu_eq *= 10
                self.mu_in *= 10
                mu_current *= 10
            else:
                print(f"Penalty too large at iter {k}")
                return self.x, self.hist
            
        print(f"Max iterations reached")    

        return self.x, self.hist


