import os, sys
import numpy as np 
import scipy.sparse as sp

import fenics as fe
try:
  fe.parameters['linear_algebra_backend'] = 'Eigen'
  LA_backend_is_Eigen = True
except:
  LA_backend_is_Eigen = False


home_dir = os.getenv('HOME')
tol = 1E-14


np.random.seed(123456789)


def get_system(coeff, mesh, nd, type_domain):
  """
  Returns A, b.

  """
  # Define function space
  V = fe.FunctionSpace(mesh, 'CG', 1)
  u, v = fe.TestFunction(V), fe.TrialFunction(V)
  #
  # Define boundary & Dirichlet BC
  if type_domain == 'block':
    if nd == 1:
      def boundary(x):
        return x[0] == 0
        #return x[0] == 0 | x[0] == 1
      #u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
    if nd == 2:
      def boundary(x):
        return x[0] == 0
        #return x[0] == 0 | x[0] == 1 | x[1] == 0 | x[1] == 1
      #u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
    if nd == 3:
      def boundary(x):
        return x[0] == 0
        #return x[0] == 0 | x[0] == 1 | x[1] == 0 | x[1] == 1 | x[2] == 1 | x[2] == 1
      #u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
  elif (type_domain == 'ublock') & (nd == 2):
    def boundary(x):
      return (x[0] == 0) | (x[0] == 1) | (x[1] == 0)
      #return x[0] == 0 | x[0] == 1 | x[1] == 0 | x[1] == 1
    #u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
  elif (type_domain == 'disk') | (type_domain == 'ellipse'):
    if nd == 2:
      def boundary(x, on_boundary):
        return on_boundary & (x[0] < 0)
  u_D = fe.Constant(0)
  #
  # Define boundary condition
  bc = [fe.DirichletBC(V, u_D, boundary)]
  #
  # Define forcing term
  f = fe.Constant(1)
  #
  # Define bilinear & linear forms
  a = fe.dot(coeff * fe.grad(u), fe.grad(v)) * fe.dx
  L = f * v * fe.dx
  #
  # Assemble linear system
  A, b = fe.assemble_system(a, L, bc)
  if LA_backend_is_Eigen:
    row, col, val = fe.as_backend_type(A).data()
  else:
    row, col, val = fe.as_backend_type(A).mat().getValuesCSR()
  A = sp.csr_matrix((val, col, row))
  b = b.get_local()
  return A, b


def get_median_A(median_coeff, mesh, nd, type_domain):
  median_A, _ = get_system(median_coeff, mesh, nd, type_domain)
  return median_A


def build_preconds(nprec, kmeans_centers, mesh, type_domain_PDE):
  KL_real = fe.Function(V)
  from pyamg.aggregation import smoothed_aggregation_solver 
  preconds = []
  for iprec in range(nprec):
    KL_real.vector()[:] = kmeans_centers[:, iprec]
    A, _ = get_system(KL_real, mesh, nd, type_domain_PDE)
    ml = smoothed_aggregation_solver(A)
    preconds += [ml.aspreconditioner()]
  return preconds