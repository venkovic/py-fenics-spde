import numpy as np
import sys 
from spde_fem import get_system, get_median_A
from solver import pcg, set_amg_precond

solve_path = 'data/solve/'

def solve_systems(discretized_spde, nsmp, nd=2, type_domain='ublock'):
  """
  Iterative solves of linear systems
  """

  median_coeff = discretized_spde.get_median()
  median_A = get_median_A(median_coeff, discretized_spde.mesh, nd, type_domain)
  median_M = set_amg_precond(median_A)
  #
  iters = {'0': np.zeros(nsmp, dtype=int),
           't': np.zeros(nsmp, dtype=int),}
  #
  itmax = 10_000
  #
  # NOT A REAL SEED...
  np.random.seed(12345)
  # Be carefull, set_amg_precond() also uses the random number generator.
  #
  for ismp in range(nsmp):
    res = 'k_omega = %d, ismp = %d, ' % (discretized_spde.symmetry['k'], ismp)
    #
    coeff = discretized_spde.sample()
    A, b = get_system(coeff, discretized_spde.mesh, nd, type_domain)
    #
    x, iterated_res_norm, it = pcg(A, b, median_M, itmax=itmax)
    iters['0'][ismp] = it
    res += 'iters for M0: %d, ' % it
    #
    M = set_amg_precond(A)
    x, iterated_res_norm, it = pcg(A, b, M, itmax=itmax)
    iters['t'][ismp] = it
    res += 'iters for Mt: %d.' % it
    print(res)
    sys.stdout.flush()
  #
  print(iters['0'].mean(), iters['t'].mean())
  sys.stdout.flush()

  if discretized_spde.symmetry['type'] == 'deterministic_ratio_random_theta':
    np.save('%s%s_k_omega%d.pcg-amg-median.iters' % (solve_path, smp_type, discretized_spde.symmetry['k']), iters['0'])
    np.save('%s%s_k_omega%d.pcg-amg-current.iters' % (solve_path, smp_type, discretized_spde.symmetry['k']), iters['t'])


def investigate_amg_setup():
  pass