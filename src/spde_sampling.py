import os, sys
import numpy as np 

import fenics as fe
try:
  fe.parameters['linear_algebra_backend'] = 'Eigen'
  LA_backend_is_Eigen = True
except:
  LA_backend_is_Eigen = False

home_dir = os.getenv('HOME')

np.random.seed(123456789)

class sampler:
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

  def __init__(self, w, W, mesh, smp_type='mc', prev_xi=None, nmcmc=None):
    """
    Instantiates sampling object.

    w: Vector of eigenvalues s.t. w[i] < w[i+1] for 0 < i < len(w) - 1.
    W: Array of eigenvectors stacked by columns.
    mesh: fenics mesh object.
    smp_type = 'mc': Sampling type among 
                     {'mc', 
                      'mcmc', 
                      'hybrid_md', 
                      'hybrid_ld',
                      'from_xis',}
    prev_xi = None: Previous latent realization for next sample in random walk. 
    nmcmc = None: Sample size for MCMC. 
                  Must not be None if prev_xi in ('mcmc', 'hybrid_ld', 'hybrid_md',)
    """

    self.__w = w
    self.__W = W
    self.__mesh = mesh
    self.__smp = smp_type
    self.cnt = 1
    #
    self.__nt = len(self.__w)
    self.__xi = np.zeros(self.__nt)
    self.__V = fe.FunctionSpace(self.__mesh, 'CG', 1)
    self.__real = fe.Function(self.__V)
    self.__DoF = self.__V.dim()
    if self.__smp == 'mcmc':
      self.__vsig2 = 2.38**2/self.__nt
      self.__initiated = False
      if prev_xi is not None:
        self.__xi = prev_xi
        self.__xi2 = self.__xi.dot(self.__xi)
        self.__initiated = True
    elif ('hybrid_md' in self.__smp) | ('hybrid_ld' in self.__smp):
      self.nmcmc = nmcmc
      self.__vsig2 = 2.38**2/self.nmcmc
      self.__initiated = False
    elif self.__smp == "from-xis":
      pass
  #
  #
  def sample(self, xi=None):
    """
    Draw a sample and returns the realization.

    xi=None: Latent realization.
    """
    if self.__smp == 'mc':
      self.__xi = np.random.normal(size=self.__nt)
    #
    #
    elif self.__smp == 'mcmc':
      #
      if not self.__initiated:
        self.__xi = np.random.normal(size=self.__nt)
        self.__xi2 = self.__xi.dot(self.__xi)
        self.__initiated = True
      #
      else:
        accepted = False
        self.cnt = 0
        while not accepted:
          chi = self.__xi + np.random.normal(scale=self.__vsig2**.5, size=self.__nt)
          chi2 = chi.dot(chi)
          alpha = min(np.exp((self.__xi2 - chi2)/2.), 1)
          if np.random.uniform() < alpha:
            accepted = True
            self.__xi = np.copy(chi)
            self.__xi2 = chi2
          self.cnt += 1
    #
    #
    elif ('hybrid_md' in self.__smp) | ('hybrid_ld' in self.__smp):
      #
      if not self.__initiated:
        self.__xi = np.random.normal(size=self.__nt)
        if 'hybrid_md' in self.__smp:
          self.__xi2 = self.__xi[:self.nmcmc].dot(self.__xi[:self.nmcmc])
        elif 'hybrid_ld' in self.__smp:
          self.__xi2 = self.__xi[-self.nmcmc:].dot(self.__xi[-self.nmcmc:])
        self.__initiated = True
      #
      else:
        accepted = False
        self.cnt = 0
        while not accepted:
          if 'hybrid_md' in self.__smp:
            chi = self.__xi[:self.nmcmc] + np.random.normal(scale=self.__vsig2**.5, size=self.nmcmc)
          elif 'hybrid_ld' in self.__smp:
            chi = self.__xi[-self.nmcmc:] + np.random.normal(scale=self.__vsig2**.5, size=self.nmcmc)
          chi2 = chi.dot(chi)
          alpha = min(np.exp((self.__xi2 - chi2)/2.), 1)
          if np.random.uniform() < alpha:
            accepted = True
            if 'hybrid_md' in self.__smp:
              self.__xi[:self.nmcmc] = np.copy(chi)
              self.__xi[self.nmcmc:] = np.random.normal(size=self.__nt - self.nmcmc)
            if 'hybrid_ld' in self.__smp:
              self.__xi[-self.nmcmc:] = np.copy(chi)
              self.__xi[:self.__nt - self.nmcmc] = np.random.normal(size=self.__nt - self.nmcmc)
            self.__xi2 = chi2
          self.cnt += 1
    #
    #
    elif self.__smp == "from-xis":
      self.__xi = xi
    #
    #
    self.__real.vector()[:] = np.zeros(self.__DoF)
    for i in range(self.__nt):
      self.__real.vector()[:] += self.__xi[i]*self.__w[i]**.5*self.__W[:, i]
    return self.__real
  #
  #
  def switch(self, smp_type):
    """
    Changes sampling type.

    smp_type: sampling type among {'mc', 
                                   'mcmc', 
                                   'hybrid_md', 
                                   'hybrid_ld',
                                   'from_xis',}
    """
    self.__smp = smp_type
  #
  #
  @property
  def xi(self):
    """
    Returns copy of latent realization.

    """
    return self.__xi
  #
  #
  @property
  def w(self):
    """
    Returns KL eigenvalues.

    """
    return self.__w
  #
  #
  @property
  def nKL(self):
    """
    Returns number of KL modes.

    """
    return self.__w.shape[0]
  #
  #    
  @property
  def W(self):
    """
    Returns KL eigenvectors.

    """
    return self.__W
  #
  #
  @property
  def mesh(self):
    """
    Returns dolfin.cpp.mesh.Mesh mesh.

    """
    return self.__mesh