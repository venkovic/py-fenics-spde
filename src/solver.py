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


def eigdefpcg(A, b, M, W, spdim, x=None, eps=1e-7, itmax=1000):
  """
  Performs RR-LO-TR-Def-PCG (Venkovic et al., 2020), here referred to as eigDef-PCG.  

  Works as a combination of eigPCG and Def-PCG. The linear solve is deflated as in
  Def-PCG, and approximate least dominant right eigenvectors of M^{-1}A are
  computed throughout the solve in a similar way as in eigPCG. This algorithm is
  an alternative to the incremental eigPCG algorithm when solving for a sequence
  of systems A xs = bs with constant SPD A and M, and different right-hand sides
  bs. This algorithm should be the method of choice when solving a sequence of
  linear systems of the form As xs = bs with correlated SPD matrices A1, A2, ...
  Examples are shown below for each type of problem.

  spdim: Maximum dimension of eigen search space.
  """ 
  if x is None:
    x = np.zeros(A.shape[0])
  r = np.zeros_like(x)
  Ap = np.zeros_like(x)
  res_norm = np.zeros_like(x)
  p = np.zeros_like(x)
  z = np.zeros_like(x)
  #
  WtA = A.dot(W).T
  WtAW = WtA.dot(W)
  WtW = W.T.dot(W)
  #
  r = b - A.dot(x)
  #
  mu = W.T.dot(r)
  mu = np.linalg.solve(WtAW, mu)
  x += W.dot(mu)
  #
  n = x.shape[0]
  nvec = W.shape[1]
  nev = nvec
  V = np.zeros((n, spdim))
  VtAV = np.zeros((spdim, spdim))
  Y = np.zeros((spdim, 2 * nvec))
  first_restart = True
  #
  it = 0
  r = b - A.dot(x)
  rTr = r.dot(r)
  z = M(r)
  rTz = r.dot(z)
  mu = np.linalg.solve(WtAW, WtA.dot(z))
  p = z - W.dot(mu)
  res_norm[it] = np.sqrt(rTr)
  #
  VtAV[:nvec, :nvec] = WtAW
  V[:, :nvec] = W
  just_restarted = False
  #
  ivec = nvec
  V[:, ivec] = z / np.sqrt(rTz)
  #
  bnorm = np.linalg.norm(b)
  tol = eps * bnorm
  #
  while (it < itmax) & (res_norm[it] > tol):
    Ap = A.dot(p)
    d = np.dot(p, Ap)
    alpha = rTz / d
    beta = 1. / rTz
    x += alpha * p
    r -= alpha * Ap
    r -= W.dot(np.linalg.solve(WtW, W.T.dot(r)))
    rTr = np.dot(r, r)
    z = M(r)
    rTz = np.dot(r, z)
    beta *= rTz
    mu = np.linalg.solve(WtAW, WtA.dot(z))
    p = beta * p + z - W.dot(mu)
    it += 1
    res_norm[it] = np.sqrt(rTr)
    #
    VtAV[ivec, ivec] += 1. / alpha
    #
    if ivec == spdim - 1:
      print('restarting!')
      if first_restart:
        VtAV[:nvec, nvec:spdim] = WtA.dot(V[:, nvec:spdim])
        first_restart = False
      Tm = (VtAV + VtAV.T) / 2 # spdim-by-spdim
      _, vecs = np.linalg.eigh(Tm)
      Y[:, :nvec] = vecs[:, :nvec]
      _, vecs = np.linalg.eigh(Tm[:spdim - 1, :spdim - 1])
      Y[:spdim - 1, nvec:] = vecs[:, :nvec]
      nev = np.linalg.matrix_rank(Y)
      di_nev = np.diag_indices(nev)
      vecs, _, _ = np.linalg.svd(Y)
      Q = vecs[:, :nev] # spdim-by-nev
      H = Q.T.dot(Tm.dot(Q)) # nev-by-nev
      vals, Z = np.linalg.eigh(H)
      V[:, :nev] = V.dot(Q.dot(Z)) # n-by-nev
      #
      ivec = nev
      V[:, ivec] = z / np.sqrt(rTz)
      VtAV[:, :] = 0
      VtAV[di_nev] = vals
      VtAV[ivec, ivec] = beta / alpha
      just_restarted = True
    #
    else:
      just_restarted = False
      ivec += 1
      V[:, ivec] = z / np.sqrt(rTz)
      VtAV[ivec - 1, ivec] = - np.sqrt(beta) / alpha
      VtAV[ivec, ivec] = beta / alpha
  #
  if not just_restarted:
    if ivec > nvec - 1:
      ivec -= 1
      if first_restart:
        VtAV[:nvec, nvec:ivec + 1] = WtA.dot(V[:, nvec:ivec + 1])
      Tm = (VtAV[:ivec + 1, :ivec + 1] + VtAV[:ivec + 1, :ivec + 1].T) / 2 # spdim-by-spdim
      Y[:, :] = 0
      _, vecs = np.linalg.eigh(Tm)
      Y[:ivec + 1, :nvec] = vecs[:, :nvec] # ivec-by-nvec
      _, vecs = np.linalg.eigh(Tm[:ivec, :ivec])
      Y[:ivec, nvec:] = vecs[:, :nvec] # (ivec-1)-by-nvec
      nev = np.linalg.matrix_rank(Y[:ivec + 1, :]) # nvec <= nev <= 2*nvec
      vecs, _, _ = np.linalg.svd(Y[:ivec + 1, :])
      Q = vecs[:, :nev] # ivec-by-nev
      H = Q.T.dot(Tm.dot(Q)) # nev-by-nev
      vals, Z = np.linalg.eigh(H)
      V[:, :nev] = V[:, :ivec + 1].dot(Q.dot(Z)) # n-by-nev
    else:
      print('Warning: Less CG iterations than the number of',
            'eigenvectors wanted. Only Lanczos vectors may be returned.')
  #
  return x, res_norm[:it], V[:, :nvec], it



def eigpcg(A, b, M, nvec, spdim, x=None, eps=1e-7, itmax=1000):
  """
  Performs eigPCG (Stathopoulos & Orginos, 2010).  

  Used at the beginning of a solving procedure of linear systems A xs = bs with
  constant SPD matrix A and SPD preconditioner M, and different right-hand sides
  bs. eigPCG may be run once (or incrementally) to generate approximate least
  dominant right eigenvectors of M^{-1}A. These approximate eigenvectors are then
  used to generate a deflated initial guess with the Init-PCG algorithm.
  Incremental eigPCG should be used when the solve of the first system ends before
  accurate eigenvector approximations can be obtained by eigPCG, which then limits
  the potential speed-up obtained for the subsequent Init-PCG solves. See Examples
  for typical use and implementation of the Incremental eigPCG algorithm
  (Stathopoulos & Orginos, 2010).

  spdim: Maximum dimension of eigen search space.
  """ 

  if x is None:
    x = np.zeros(A.shape[0])
  r = np.zeros_like(x)
  Ap = np.zeros_like(x)
  res_norm = np.zeros_like(x)
  p = np.zeros_like(x)
  z = np.zeros_like(x)
  #
  n = x.shape[0]
  V = np.zeros((n, spdim))
  VtAV = np.zeros((spdim, spdim))
  Y = np.zeros((spdim, 2 * nvec))
  tvec = np.zeros_like(x)
  just_restarted = False
  #
  it = 0
  r = b - A.dot(x)
  rTr = r.dot(r)
  z = M(r)
  rTz = r.dot(z)
  p = np.copy(z)
  res_norm[it] = np.sqrt(rTr)
  #
  ivec = 0
  V[:, ivec] = z / np.sqrt(rTz)
  just_restarted = False
  #
  bnorm = np.linalg.norm(b)
  tol = eps * bnorm
  #
  while (it < itmax) & (res_norm[it] > tol):
    Ap = A.dot(p)
    d = np.dot(p, Ap)
    alpha = rTz / d
    beta = 1. / rTz
    x += alpha * p
    r -= alpha * Ap
    rTr = np.dot(r, r)
    z = M(r)
    if just_restarted:
      hlpr = np.sqrt(rTz)
    rTz = np.dot(r, z)
    beta *= rTz
    if ivec == spdim - 1:
      tvec -= beta * Ap
    p = beta * p + z
    it += 1
    res_norm[it] = np.sqrt(rTr)
    #
    VtAV[ivec, ivec] += 1. / alpha
    if just_restarted:
      tvec += Ap
      nev = ivec
      VtAV[:nev, ivec] = V[:, :nev].T.dot(tvec / hlpr)
      just_restarted = False
    #
    if ivec == spdim - 1:
      print('restarting!')
      VtAV = V.T.dot(A.dot(V))
      Tm = (VtAV + VtAV.T) / 2 # spdim-by-spdim    
      _, vecs = np.linalg.eigh(Tm)
      Y[:, :nvec] = vecs[:, :nvec]
      _, vecs = np.linalg.eigh(Tm[:spdim - 1, :spdim - 1])
      Y[:spdim - 1, nvec:] = vecs[:, :nvec]
      nev = np.linalg.matrix_rank(Y)
      di_nev = np.diag_indices(nev)
      vecs, _, _ = np.linalg.svd(Y)
      Q = vecs[:, :nev] # spdim-by-nev
      H = Q.T.dot(Tm.dot(Q)) # nev-by-nev
      vals, Z = np.linalg.eigh(H)
      V[:, :nev] = V.dot(Q.dot(Z)) # n-by-nev
      #
      ivec = nev
      V[:, ivec] = z / np.sqrt(rTz)
      VtAV[:, :] = 0
      VtAV[di_nev] = vals
      VtAV[ivec, ivec] = beta / alpha
      tvec = - beta * Ap
      just_restarted = True
    #
    else:
      ivec += 1
      V[:, ivec] = z / np.sqrt(rTz)
      VtAV[ivec - 1, ivec] = - np.sqrt(beta) / alpha
      VtAV[ivec, ivec] = beta / alpha
  #
  if not just_restarted:
    if ivec > nvec - 1:
      ivec -= 1
      Tm = (VtAV[:ivec + 1, :ivec + 1] + VtAV[:ivec + 1, :ivec + 1].T) / 2 # spdim-by-spdim
      Y[:, :] = 0
      _, vecs = np.linalg.eigh(Tm)
      Y[:ivec + 1, :nvec] = vecs[:, :nvec] # ivec-by-nvec      
      _, vecs = np.linalg.eigh(Tm[:ivec, :ivec])
      Y[:ivec, nvec:] = vecs[:, :nvec] # (ivec-1)-by-nvec
      nev = np.linalg.matrix_rank(Y[:ivec + 1, :]) # nvec <= nev <= 2*nvec
      vecs, _, _ = np.linalg.svd(Y[:ivec + 1, :])
      Q = vecs[:, :nev] # ivec-by-nev
      H = Q.T.dot(Tm.dot(Q)) # nev-by-nev
      vals, Z = np.linalg.eigh(H)
      V[:, :nev] = V[:, :ivec + 1].dot(Q.dot(Z)) # n-by-nev
    else:
      print('Warning: Less CG iterations than the number of',
            'eigenvectors wanted. Only Lanczos vectors may be returned.')
  #
  return x, res_norm[:it], V[:, :nvec], it
