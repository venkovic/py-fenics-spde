from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
iproc = comm.Get_rank()

import sys
sys.path += ['../../src/']


do_preproc_kl = True
do_preproc_plot_coeffs = False
do_solve = True
do_postproc = False

nd = 2
nEl, type_domain = 16_000, 'ublock'
sig2, L = 1., .2
model = "SExp" # ('Exp', 'SExp')
delta2 = 1 - .995

nsmp = 100

#symmetry = {'type': 'isotropic',}
#symmetry = {'type': 'deterministic_ratio', 'ratio': .01,}
#symmetry = {'type': 'deterministic_ratio_theta', 'ratio': .01, 'theta': 3 * np.pi,}
#symmetry = {'type': 'deterministic_ratio_random_theta', 'ratio': .01, 'omega': 1 / (10 * np.pi),}
#symmetry = {'type': 'random_u1_u2_theta', 'theta': (0, 3 * np.pi),}
#symmetry = {'type': 'deterministic_ratio_random_u1_u2_theta', 'theta': (0, 3 * np.pi), 'ratio':.01}

k_omegas = (1, 2, 5, 10,)
symmetries = []
for k_omega in k_omegas:
  symmetries += [{'type': 'deterministic_ratio_random_theta', 'ratio': .01, 'omega': 1 / (k_omega * np.pi), 'k': k_omega}]


from example01_preproc import get_spde
if do_preproc_plot_coeffs:
  from example01_preproc import plot_coeff
#
if do_solve:
  from example01_solve import solve_systems
#
if do_postproc:
  from example01_postproc import plot_results


for isym, symmetry in enumerate(symmetries):
  #
  if isym % nproc == iproc:
    discretized_spde = get_spde(nEl, type_domain, sig2, L, model, symmetry=symmetry, delta2=delta2)
    #
    if do_preproc_plot_coeffs:
      plot_coeff(discretized_spde)  
    #
    if do_solve:  
      solve_systems(discretized_spde, nsmp, nd=nd, type_domain=type_domain)  
    #
    elif do_postproc:
      plot_results()