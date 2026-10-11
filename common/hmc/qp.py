"""OSQP wrapper with a *fixed dense* sparsity pattern.

Whole-body QPs change every control step but keep the same structure. OSQP can
re-solve very fast if the pattern does not change, so we declare P (upper
triangle) and A as fully dense once, then only push new numeric values.

    min ½ xᵀ P x + qᵀ x   s.t.  l ≤ A x ≤ u
"""
from __future__ import annotations

import numpy as np
import osqp
import scipy.sparse as sp


class DenseQP:
    def __init__(self, n: int, m: int, eps: float = 1e-5, max_iter: int = 4000):
        self.n, self.m = n, m
        # P: upper triangle, column-major
        rows, cols = np.triu_indices(n)
        order = np.lexsort((rows, cols))            # sort by column then row
        self._pr, self._pc = rows[order], cols[order]
        indptr = np.concatenate([[0], np.cumsum(np.arange(1, n + 1))])
        P0 = sp.csc_matrix((np.ones(len(self._pr)), self._pr, indptr), shape=(n, n))
        # A: dense, column-major
        A0 = sp.csc_matrix((np.ones(n * m), np.tile(np.arange(m), n),
                            np.arange(0, n * m + 1, m)), shape=(m, n))
        self.solver = osqp.OSQP()
        self.solver.setup(P=P0, q=np.zeros(n), A=A0, l=-np.ones(m), u=np.ones(m),
                          verbose=False, eps_abs=eps, eps_rel=eps, max_iter=max_iter,
                          warm_starting=True, polishing=False, adaptive_rho=True)
        self.last_x = None
        self.ok = False
        self.iters = 0

    def solve(self, P: np.ndarray, q: np.ndarray, A: np.ndarray, l: np.ndarray, u: np.ndarray):
        Px = P[self._pr, self._pc]
        Ax = A.flatten(order="F")
        self.solver.update(Px=Px, Ax=Ax, q=q, l=l, u=u)
        if self.last_x is not None:
            self.solver.warm_start(x=self.last_x)
        res = self.solver.solve()
        status = str(res.info.status).lower()
        self.ok = ("solved" in status) and ("inaccurate" not in status) and np.all(np.isfinite(res.x))
        self.iters = res.info.iter
        self.status = status
        if self.ok:
            self.last_x = res.x.copy()
        return res.x if self.ok else None
