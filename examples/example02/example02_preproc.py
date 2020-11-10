import os, sys, copy
import numpy as np

from spde_kl import get_KL
from spde_sampling import sampler
from spde_anisotropy import anisotropic_spde
from fem_util import to_axis

home_dir = os.getenv('HOME')
img_path = 'img/'

path_to_kls = home_dir + '/Dropbox/Git/py-fenics-spde/kl-data/'
path_to_meshes = home_dir + '/Dropbox/Git/py-fenics-spde/meshes/'


def get_spde(nEl, type_domain, sig2, L, model, nmcmc, nd=2, symmetry = {'type': 'isotropic',},
 delta2=.005):
  """
  Returns sampler.

  """

  #
  ndom = 1
  w, W, mesh = get_KL(nd, nEl, ndom, sig2, L, model, delta2, type_domain=type_domain, 
    path_to_meshes=path_to_meshes, path=path_to_kls)
  #
  np.random.seed(4525899)
  fe_sampler = sampler(w, W, mesh, smp_type='hybrid_md', nmcmc=nmcmc)
  np.random.seed(4525899)
  fe_sampler2 = sampler(w, W, mesh, smp_type='hybrid_md', nmcmc=nmcmc)
  #
  discretized_spde = anisotropic_spde(fe_sampler, symmetry, fe_sampler2=fe_sampler2)
  #
  return discretized_spde


def plot_coeff(discretized_spde, nsmp=4, smp_type='hybrid_md'):
  """
  Returns sampler.

  """

  import pylab as pl
  pl.rcParams['text.usetex'] = True
  params={'text.latex.preamble':[r'\usepackage{amssymb}',r'\usepackage{amsmath}']}
  pl.rcParams.update(params)
  #
  #
  np.random.seed(12345)
  #
  if discretized_spde.symmetry['type'] in ('deterministic_ratio_theta',
                                           'deterministic_ratio_random_theta',
                                           'random_u1_u2_theta',
                                           'deterministic_ratio_random_u1_u2_theta',):
    #
    fig, axes = pl.subplots(3, nsmp, figsize=(1.7 * nsmp, 5))
    #
    indexes = ('11', '22', '12',)
    #
    for ismp in range(nsmp):
      #
      kappa = discretized_spde.sample(return_type='dict')
      #
      for i, inds in enumerate(indexes):
        #
        if inds == '12':
          im = to_axis(kappa[inds], discretized_spde.mesh, axes[i, ismp], plot='symlognorm', cmap='RdBu', ticks=False)
        else:
          im = to_axis(kappa[inds], discretized_spde.mesh, axes[i, ismp], plot='lognorm', cmap='Blues', ticks=False)
      #
      string_nsmp = f'{10 ** ismp:,}'
      axes[0, ismp].set_title(r'$s=%s$' % string_nsmp)
    #
    for i, ax in enumerate(axes[:, 0]):
      ax.set_ylabel(r'$c_{%s}(\underline{x})$' % indexes[i])
    #
    cb_ax = fig.add_axes([0.91, 0.11, 0.02, 0.77])
    cbar = fig.colorbar(im, cax=cb_ax)
    #
    for ax in axes.flatten():
      ax.set_aspect('equal')
    pl.subplots_adjust(hspace=.1, wspace=.01)
    #
    k = discretized_spde.symmetry['k']
    ratio = discretized_spde.symmetry['ratio']
    #
    if 'hybrid' in smp_type:
      smp_type += str(discretized_spde.nmcmc)
    pl.savefig('%s%s_k_omega%d_ratio%g.png' % (img_path, smp_type, k, ratio), bbox_inches='tight', dpi=300)