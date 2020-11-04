import os, sys
import numpy as np 

import fenics as fe
try:
  fe.parameters['linear_algebra_backend'] = 'Eigen'
  LA_backend_is_Eigen = True
except:
  LA_backend_is_Eigen = False

from fem_util import get_mesh_fname


def get_kl_fname(nd, nEl, sig2, L, model, delta, type_domain='block'):
  return "%gDKL_%s_%s_L%g_sig2%g_nEl%g" %(nd, type_domain, model, L, sig2, nEl)


def save_KL(w, W, mesh, nd, nEl, sig2, L, model, delta, type_domain='block', path='./', path_to_meshes=None):
  kl_fname = get_kl_fname(nd, nEl, sig2, L, model, delta, type_domain)
  np.save(path + kl_fname + ".eigvals", w)
  np.save(path + kl_fname + ".eigvecs", W)
  #
  mesh_fname = get_mesh_fname(nd, nEl, type_domain)
  if path_to_meshes is None:
    mesh_file = fe.File('%s%s.xml' % (path, mesh_fname))
  else:
    mesh_file = fe.File('%s%s.xml' % (path_to_meshes, mesh_fname))
  mesh_file << mesh


def load_KL(nd, nEl, sig2, L, model, delta, type_domain='block', path='./', path_to_meshes=None):
  kl_fname = get_kl_fname(nd, nEl, sig2, L, model, delta, type_domain)
  w = np.load(path + kl_fname + '.eigvals.npy')
  print('Loaded %s.' % (path + kl_fname + '.eigvals.npy'))
  W = np.load(path + kl_fname + '.eigvecs.npy')
  print('Loaded %s.' % (path + kl_fname + '.eigvecs.npy'))
  #
  mesh_fname = get_mesh_fname(nd, nEl, type_domain)
  if path_to_meshes is None:
    mesh = fe.Mesh(path + mesh_fname + '.xml')
    print('Loaded %s.' % (path + mesh_fname + '.xml'))
  else:
    mesh = fe.Mesh(path_to_meshes + mesh_fname + '.xml')
    print('Loaded %s.' % (path_to_meshes + mesh_fname + '.xml'))
  #
  return w, W, mesh