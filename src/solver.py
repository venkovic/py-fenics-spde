import numpy as np
from scipy.sparse import csr_matrix
from pyamg.aggregation import smoothed_aggregation_solver

def pcg(A, b, M, x=None, eps=1e-7, itmax=1000):
  it = 0
  if x is None:
    x = np.zeros(A.shape[0])
  r = b - A.dot(x)
  rTr = r.dot(r)
  z = M(r) 
  rTz = r.dot(z)
  p = np.copy(z)
  iterated_res_norm = [np.sqrt(rTr)]
  bnorm = np.linalg.norm(b)
  tol = eps * bnorm
  while (it < itmax) & (iterated_res_norm[-1] > tol):
    Ap = A.dot(p)
    d = Ap.dot(p)
    alpha = rTz / d
    beta = 1. / rTz
    x += alpha * p
    r -= alpha * Ap
    rTr = r.dot(r)  
    z = M(r)
    rTz = r.dot(z)
    beta *= rTz
    p = beta * p + z
    iterated_res_norm += [np.sqrt(rTr)]
    it += 1
  return x, iterated_res_norm, it

def set_amg_precond(A):
  ml = smoothed_aggregation_solver(csr_matrix(A))
  M = ml.aspreconditioner(cycle='V')
  return M