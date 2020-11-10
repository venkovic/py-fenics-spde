import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
import sys 
from spde_fem import get_system, get_median_A
from solver import pcg, eigdefpcg, set_amg_precond, eigpcg

solve_path = 'data/solve/'

def solve_systems(discretized_spde, nEl, nsmp, nvec=200, nvec_recycle=50, spdim=400, 
  nd=2, type_domain='ublock', smp_type='hybrid_md'):
  """
  Iterative solves of linear systems
  """

  median_coeff = discretized_spde.get_median()
  median_A = get_median_A(median_coeff, discretized_spde.mesh, nd, type_domain)
  median_M = set_amg_precond(median_A)
  #
  iters = {'0': np.zeros(nsmp, dtype=int),
           'd': np.zeros(nsmp, dtype=int), 
           't': np.zeros(nsmp, dtype=int),}
  #
  itmax = 1_000
  #
  # NOT A REAL SEED...
  np.random.seed(123457)
  # Be carefull, set_amg_precond() also uses the random number generator.
  #
  for ismp in range(nsmp):
    if discretized_spde.symmetry['type'] == 'deterministic_ratio_random_theta':
      res = 'k_omega = %d, ismp = %d, ' % (discretized_spde.symmetry['k'], ismp)
    #
    elif discretized_spde.symmetry['type'] == 'deterministic_ratio_theta':
      res = 'theta = %g, ismp = %d, ' % (discretized_spde.symmetry['theta'], ismp)
    #
    else:
      res = 'ismp = %d, ' % ismp

    #
    coeff = discretized_spde.sample()
    A, b = get_system(coeff, discretized_spde.mesh, nd, type_domain)
    #
    x, iterated_res_norm, it = pcg(A, b, median_M, itmax=itmax)
    iters['0'][ismp] = it
    res += 'iters for M0: %d, ' % it
    #
    if ismp == 0:
      if False:
        _, W = sparse.linalg.eigsh(A, k=nvec, sigma=0)
      else:
        _, _, W, _  = eigpcg(A, b, median_M, nvec, spdim, itmax=itmax)
    #
    print(np.linalg.matrix_rank(W))
    x, iterated_res_norm, V, it = eigdefpcg(A, b, median_M, W, spdim, itmax=itmax)
    #
    W[:, np.random.permutation(nvec)[:nvec_recycle]] = V[:, :nvec_recycle]
    #
    iters['d'][ismp] = it
    res += 'iters for Def-M0: %d, ' % it
    #
    M = set_amg_precond(A)
    x, iterated_res_norm, it = pcg(A, b, M, itmax=itmax)
    iters['t'][ismp] = it
    res += 'iters for Mt: %d.' % it
    #
    print(res)
    sys.stdout.flush()
  #
  print(iters['0'].mean(), iters['d'].mean(), iters['t'].mean())
  rel_change = (iters['0'].mean() - iters['d'].mean()) / iters['0'].mean()
  print('Part of iterations saved by deflation: %g' % rel_change)
  sys.stdout.flush()

  if discretized_spde.symmetry['type'] == 'deterministic_ratio_random_theta':
    k = discretized_spde.symmetry['k']
    ratio = discretized_spde.symmetry['ratio']
    if 'hybrid' in smp_type:
      smp_type += str(discretized_spde.nmcmc)
    np.save('%s%s_k_omega%d_ratio%g.%dDoFs.pcg-amg-median.iters' % (solve_path, smp_type, k, ratio, nEl), iters['0'])
    np.save('%s%s_k_omega%d_ratio%g.%dDoFs.pcg-amg-current.iters' % (solve_path, smp_type, k, ratio, nEl), iters['t'])
  #
  elif discretized_spde.symmetry['type'] == 'deterministic_ratio_theta':
    ratio = discretized_spde.symmetry['ratio']
    theta = discretized_spde.symmetry['theta']
    if 'hybrid' in smp_type:
      smp_type += str(discretized_spde.nmcmc)
    np.save('%s%s_ratio%g_theta%g.%dDoFs.pcg-amg-median.iters' % (solve_path, smp_type, ratio, theta, nEl), iters['0'])
    np.save('%s%s_ratio%g_theta%g.%dDoFs.pcg-amg-current.iters' % (solve_path, smp_type, ratio, theta, nEl), iters['t'])




def investigate_amg_setup():
  pass