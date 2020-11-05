import os, sys
import numpy as np 
import scipy.sparse as sp
from scipy.stats import norm

import fenics as fe
try:
  fe.parameters['linear_algebra_backend'] = 'Eigen'
  LA_backend_is_Eigen = True
except:
  LA_backend_is_Eigen = False


home_dir = os.getenv('HOME')
tol = 1E-14


np.random.seed(123456789)



class anisotropic_spde:
  """
  Sampling object for the truncated karhunen-loeve representation of a log-normal
  random field using finite elements.

  Public attributes:
  cnt: Sampling counter.

  Methods:
  __init__(self, w, W, mesh, smp_type='mc', prev_xi=None, nmcmc=None)
  sample(self, xi=None)
  switch(self, smp_type)
  get_xi(self)
  """

  def __init__(self, fe_sampler, symmetry, nd=2):
    """
    Instantiates **.

    symmetry: Symmetry of coefficient in {{'type': 'isotropic',},
                                          {'type': 'deterministic_ratio', 'ratio': ratio,},,
                                          {'type': 'deterministic_ratio_theta', 'ratio': ratio, 'theta': theta},,
                                          {'type': 'deterministic_ratio_random_theta', 'ratio': ratio,},}
    """

    self.__fe_sampler = fe_sampler
    self.__symmetry = symmetry


  def sample(self, return_type='fe_object'):
    """
    Samples

    return_type='fe_object': in {'fe_object', 
                                 'dict',}
    """

    symmetry = self.__symmetry
    scalar_func = np.exp(self.__fe_sampler.sample().vector()[:])
    #
    if symmetry['type'] == 'deterministic_ratio_random_theta':
      theta = (symmetry['theta'][1] - symmetry['theta'][0]) * norm.cdf(self.__fe_sampler.sample().vector()[:])
    #
    elif symmetry['type'] == 'random_u1_u2_theta':
      scalar_func2 = np.exp(self.__fe_sampler.sample().vector()[:])
      theta = (symmetry['theta'][1] - symmetry['theta'][0]) * norm.cdf(self.__fe_sampler.sample().vector()[:])
    #
    #
    #
    if symmetry['type'] == 'isotropic':
      #
      # Define coefficient field
      coeff = fe.Function(V)
      coeff.vector()[:] = scalar_func.copy()
      #fe.local_project(scalar_func, V, coeff)
      # Implement smth like this, as found googling "fenics local_project"
    #
    #
    elif symmetry['type'] == 'deterministic_ratio':
      #
      # Define tensorial function space of coefficient
      VV = fe.TensorFunctionSpace(self.__fe_sampler.mesh, 'CG', 1)
      #
      # Define coefficient field
      coeff = fe.Function(VV)
      components = np.zeros(VV.dim())
      #
      inds_11 = (np.arange(VV.dim()) % VV.num_sub_spaces()) == 0
      components[inds_11] = scalar_func.copy()
      #
      # K_12 and K_21 components:
      inds_12 = (np.arange(VV.dim()) % VV.num_sub_spaces()) == 1
      components[inds_12] = 0
      inds_21 = (np.arange(VV.dim()) % VV.num_sub_spaces()) == 2
      components[inds_21] = 0
      #
      # K_22 components:
      inds_22 = (np.arange(VV.dim()) % VV.num_sub_spaces()) == 3
      components[inds_22] = symmetry['ratio'] * scalar_func.copy()
      #
      coeff.vector()[:] = components.copy()
    #
    #
    elif symmetry['type'] == 'deterministic_ratio_theta':
      #
      # Define tensorial function space of coefficient
      VV = fe.TensorFunctionSpace(self.__fe_sampler.mesh, 'CG', 1)
      #
      # u1 = cos(theta)e1 + sin(theta)e2
      # u2 = sin(theta)e1 - cos(theta)e2
      #
      # K = [u1(theta).u1(theta)^T + ratio * u2(theta).u2(theta)^T] * k(x)
      # 
      # u1(theta).u1(theta)^T = cos^2(theta)e1.e1^T + sin^2(theta)e2.e2^T
      #                         + cos(theta) * sin(theta) * (e1.e2^T + e2.e1^T)
      # u2(theta).u2(theta)^T = sin^2(theta)e1.e1^T + cos^2(theta)e2.e2^T
      #                         - cos(theta) * sin(theta) * (e1.e2^T + e2.e1^T)
      #
      # so that:
      #
      # K_11 = [cos^2(theta) + ratio * sin^2(theta)] * k(x)
      # K_12 = cos(theta) * sin(theta) * (1 - ratio) * k(x)
      # K_22 = [sin^2(theta) + ratio * cos^2(theta)] * k(x)
      #
      #
      # Define coefficient field
      coeff = fe.Function(VV)
      components = np.zeros(VV.dim())
      theta = symmetry['theta']
      ratio = symmetry['ratio']
      #
      # K_11 components:
      inds_11 = (np.arange(VV.dim()) % VV.num_sub_spaces()) == 0
      components[inds_11] = (np.cos(theta) ** 2 + ratio * np.sin(theta) ** 2) * scalar_func
      #
      # K_12 and K_21 components:
      inds_12 = (np.arange(VV.dim()) % VV.num_sub_spaces()) == 1
      components[inds_12] = np.cos(theta) * np.sin(theta) * (1 - ratio) * scalar_func
      inds_21 = (np.arange(VV.dim()) % VV.num_sub_spaces()) == 2
      components[inds_21] = np.cos(theta) * np.sin(theta) * (1 - ratio) * scalar_func
      #
      # K_22 components:
      inds_22 = (np.arange(VV.dim()) % VV.num_sub_spaces()) == 3
      components[inds_22] = (np.sin(theta) ** 2 + ratio * np.cos(theta) ** 2) * scalar_func
      #
      coeff.vector()[:] = components.copy()
    #
    #
    elif symmetry['type'] == 'deterministic_ratio_random_theta':
      #
      # Define tensorial function space of coefficient
      VV = fe.TensorFunctionSpace(self.__fe_sampler.mesh, 'CG', 1)
      #
      # u1 = cos(theta(x))e1 + sin(theta(x))e2
      # u2 = sin(theta(x))e1 - cos(theta(x))e2
      #
      # K = [u1(theta(x)).u1(theta(x))^T + ratio * u2(theta(x)).u2(theta(x))^T] * k(x)
      #
      # so that:
      #
      # K_11 = [cos^2(theta(x)) + ratio * sin^2(theta(x))] * k(x)
      # K_12 = cos(theta(x)) * sin(theta(x)) * (1 - ratio) * k(x)
      # K_22 = [sin^2(theta(x)) + ratio * cos^2(theta(x))] * k(x)
      #
      #
      # Define coefficient field
      coeff = fe.Function(VV)
      components = np.zeros(VV.dim())
      ratio = symmetry['ratio']
      #
      #
      # K_11 components:
      inds_11 = (np.arange(VV.dim()) % VV.num_sub_spaces()) == 0
      components[inds_11] = (np.cos(theta) ** 2 + ratio * np.sin(theta) ** 2) * scalar_func
      #
      # K_12 and K_21 components:
      inds_12 = (np.arange(VV.dim()) % VV.num_sub_spaces()) == 1
      components[inds_12] = np.cos(theta) * np.sin(theta) * (1 - ratio) * scalar_func
      inds_21 = (np.arange(VV.dim()) % VV.num_sub_spaces()) == 2
      components[inds_21] = np.cos(theta) * np.sin(theta) * (1 - ratio) * scalar_func
      #
      # K_22 components:
      inds_22 = (np.arange(VV.dim()) % VV.num_sub_spaces()) == 3
      components[inds_22] = (np.sin(theta) ** 2 + ratio * np.cos(theta) ** 2) * scalar_func
      #
      coeff.vector()[:] = components.copy()  
    #
    #
    elif symmetry['type'] == 'random_u1_u2_theta':
      #
      # Define tensorial function space of coefficient
      VV = fe.TensorFunctionSpace(self.__fe_sampler.mesh, 'CG', 1)
      #
      # u1 = cos(theta(x))e1 + sin(theta(x))e2
      # u2 = sin(theta(x))e1 - cos(theta(x))e2
      #
      # K = k1(x) * u1(theta(x)).u1(theta(x))^T + k2(x) * u2(theta(x)).u2(theta(x))^T
      #
      # so that:
      #
      # K_11 = cos^2(theta(x)) * k1(x) + sin^2(theta(x)) * k2(x)
      # K_12 = cos(theta(x)) * sin(theta(x)) * (k1(x) - k2(x))
      # K_22 = sin^2(theta(x)) * k1(x) + cos^2(theta(x)) * k2(x)
      #
      #
      # Define coefficient field
      coeff = fe.Function(VV)
      components = np.zeros(VV.dim())
      #
      #
      # K_11 components:
      inds_11 = (np.arange(VV.dim()) % VV.num_sub_spaces()) == 0
      components[inds_11] = np.cos(theta) ** 2 * scalar_func + np.sin(theta) ** 2 * scalar_func2
      #
      # K_12 and K_21 components:
      inds_12 = (np.arange(VV.dim()) % VV.num_sub_spaces()) == 1
      components[inds_12] = np.cos(theta) * np.sin(theta) * (scalar_func - scalar_func2)
      inds_21 = (np.arange(VV.dim()) % VV.num_sub_spaces()) == 2
      components[inds_21] = np.cos(theta) * np.sin(theta) * (scalar_func - scalar_func2)
      #
      # K_22 components:
      inds_22 = (np.arange(VV.dim()) % VV.num_sub_spaces()) == 3
      components[inds_22] = np.sin(theta) ** 2 * scalar_func + np.cos(theta) ** 2 * scalar_func2
      #
      coeff.vector()[:] = components.copy()
    #
    #
    #
    if return_type == 'fe_object':
      return coeff
    #
    #
    elif return_type == 'dict':
      #
      if symmetry['type'] == 'isotropic':
        return {'11': scalar_func}
      #
      elif symmetry['type'] == 'deterministic_ratio':
        return {'11': components[inds_11], '22': components[inds_22],}
      #
      elif symmetry['type'] == 'deterministic_ratio_theta':
        return {'11': components[inds_11], '12': components[inds_12],
                                           '22': components[inds_22],}
      #
      elif symmetry['type'] == 'deterministic_ratio_random_theta':
        return {'11': components[inds_11], '12': components[inds_12],
                                           '22': components[inds_22],}
      #
      elif symmetry['type'] == 'random_u1_u2_theta':
        return {'11': components[inds_11], '12': components[inds_12],
                                           '22': components[inds_22],}

  @property
  def mesh(self):
    """
    Returns dolfin.cpp.mesh.Mesh mesh.
    """
    return self.__fe_sampler.mesh


  @property
  def symmetry(self):
    """
    Returns symmetry dictionary.
    """
    return self.__symmetry