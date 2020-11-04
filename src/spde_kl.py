import time, os, sys
import numpy as np 

import scipy.linalg as LA 
import scipy.sparse as sp
import scipy.sparse.linalg

import fenics as fe
try:
  fe.parameters['linear_algebra_backend'] = 'Eigen'
  LA_backend_is_Eigen = True
except:
  LA_backend_is_Eigen = False

try:
  import pymetis
  pymetis_is_available = True
except:
  pymetis_is_available = False

from fem_util import get_mesh, to_vtk
from kl_util import get_kl_fname, save_KL, load_KL


home_dir = os.getenv('HOME')
tol = 1E-14

np.random.seed(123456789)


def __truncate_KL(evals, delta2, sig2, subarea=1):
  """
  Truncates KL.

  evals is in decreasing order.
  """

  energy, ind = 0., 0
  thresh = (1 - delta2) * sig2
  _subn_al = len(evals)
  for ind in range(_subn_al):
    if evals[_subn_al - ind - 1] > 0:
      energy += evals[_subn_al - ind - 1] / subarea
      if energy >= thresh:
        return ind + 1
    else:
      return ind
  return ind + 1


def __compute_KL(nd, nEl, sig2, L, model, delta2, type_domain='block', eigen_solver='eigh'):  
  """
  Computes KL.
  """

  # Define kernel
  if nd == 1:
    if model == "Exp":
      k_nd = fe.Expression('sig2*exp(-abs(x[0]-y0)/L)', sig2=sig2, L=L, y0=0., degree=1)
    elif model == "SExp":
      k_nd = fe.Expression('sig2*exp(-pow(x[0]-y0,2)/(L*L))', sig2=sig2, L=L, y0=0., degree=1)
  elif nd == 2:
    if model == "Exp":
      k_nd = fe.Expression('sig2*exp(-sqrt(pow(x[0]-y0,2) + pow(x[1]-y1,2))/L)', sig2=sig2, L=L, y0=0., y1=0., degree=1)
    elif model == "SExp":
      k_nd = fe.Expression('sig2*exp(-(pow(x[0]-y0,2) + pow(x[1]-y1,2))/(L*L))', sig2=sig2, L=L, y0=0., y1=0., degree=1)
  #
  # Create mesh
  mesh = get_mesh(nd, nEl, type_domain)
  area = fe.assemble(1 * fe.dx(mesh))
  #
  # Define function space
  V = fe.FunctionSpace(mesh, "CG", 1)
  u, v = fe.TrialFunction(V), fe.TestFunction(V)
  vrt2dof, dof2vrt = fe.vertex_to_dof_map(V), fe.dof_to_vertex_map(V)
  DoF = V.dim()
  print("Mesh created and function space defined with %d DoF." % DoF)
  #
  # Assemble the mass matrix M
  t0 = time.time()
  M = fe.assemble(u * v * fe.dx).array()     # M_ij = \int phi_i(x) * phi_j(x) dx
  #
  # Assemble the kernel matrix K  
  R = np.zeros((DoF, DoF))         # R_ij = \int C(x_i, y) * phi_j(y) dy
  for i, xi in enumerate(mesh.coordinates()):
    if nd == 1:
      k_nd.y0 = xi[0]
    elif nd == 2:
      k_nd.y0, k_nd.y1 = xi
    R[vrt2dof[i], :] = fe.assemble(k_nd * v * fe.dx).get_local()
  K = np.zeros((DoF, DoF))         # K_ij = \int phi_i(x) * ( \int C(x, y) * phi_j(y) dy ) dx
  R_func = fe.Function(V)
  for j in range(DoF):
    R_func.vector()[:] = np.array(R[:, j]) 
    K[:, j] = fe.assemble(R_func * v * fe.dx).get_local()
  print("Eigenvalue problem assembled in %g s." % (time.time()-t0))
  #
  # Solve generalized eigenvalue problem K.W = M.W.diag(w)
  n_al = DoF
  t0 = time.time()
  if eigen_solver == 'eigh':
    w, W = LA.eigh(K, M, eigvals=(DoF - n_al, DoF - 1))
  #
  elif eigen_solver == 'lobpcg':
    W = np.zeros((DoF, n_al))
    np.fill_diagonal(W, 1)
    # precond = (W * w ** -1).dot(W.T)
    w, W = sp.linalg.lobpcg(K, X=W, B=sp.csr_matrix(M))
    w = w[::-1]
    W = W[:, ::-1]

  print(w)
  print("Eigenvalue problem solved by %s in %g s." % (eigen_solver, time.time()-t0))
  # Truncate expansion
  nt = __truncate_KL(w, delta2, sig2, subarea=area)
  selected_modes = range(n_al - 1, n_al - 1 - nt, -1)
  w = w[selected_modes]
  W = W[:, selected_modes]
  message = ""
  energy = w.sum() / area
  if energy < (1. - delta2) * sig2:
    message = "Warning: %d RVs are not enough to reach target variance within prescribed accuracy " % nt
  else:
    message = "%d RVs reach target variance within prescribed accuracy " % nt
  message += "(energy/sig2 = %g/%g)." % (energy, sig2)
  print(message)
  return w, W, mesh 

def __compute_KL_DD(nd, nEl, ndom, sig2, L, model, delta2, type_domain):
  """
  Computes DD KL.
  """

  # Define kernel
  if nd == 1:
    if model == "Exp":
      k_nd = Expression('sig2*exp(-abs(x[0]-y0)/L)', sig2=sig2, L=L, y0=0., degree=1)
    elif model == "SExp":
      k_nd = Expression('sig2*exp(-pow(x[0]-y0,2)/(L*L))', sig2=sig2, L=L, y0=0., degree=1)
  elif nd == 2:
    if model == "Exp":
      k_nd = Expression('sig2*exp(-sqrt(pow(x[0]-y0,2) + pow(x[1]-y1,2))/L)', sig2=sig2, L=L, y0=0., y1=0., degree=1)
    elif model == "SExp":
      k_nd = Expression('sig2*exp(-(pow(x[0]-y0,2) + pow(x[1]-y1,2))/(L*L))', sig2=sig2, L=L, y0=0., y1=0., degree=1)
  elif nd == 3:
    if model == "Exp":
      k_nd = Expression('sig2*exp(-sqrt(pow(x[0]-y0,2) + pow(x[1]-y1,2) + pow(x[2]-y2,2))/L)', sig2=sig2, L=L, y0=0., y1=0., y2=0., degree=1)
    elif model == "SExp":
      k_nd = Expression('sig2*exp(-(pow(x[0]-y0,2) + pow(x[1]-y1,2) + pow(x[2]-y2,2))/(L*L))', sig2=sig2, L=L, y0=0., y1=0., y2=0., degree=1)
  #
  # Create global mesh
  mesh = get_mesh(nd, nEl, type_domain)
  ndoms = ndom ** nd
  #
  # Define global function space
  V = fe.FunctionSpace(mesh, 'CG', 1)
  vrt2dof = fe.vertex_to_dof_map(V)
  DoF = V.dim()
  print("Mesh created and function space defined with %d DoF." % DoF)
  #
  # Create subdomains
  if pymetis_is_available:
    #
    # Build adjacency list of elements
    adj_elem = [[] for i in range(mesh.num_cells())]
    for ifacet in range(mesh.num_facets()):
      facet = fe.MeshEntity(mesh, mesh.topology().dim()-1, ifacet)
      elems = list(map(lambda elem: elem.index(), fe.entities(facet, mesh.topology().dim())))
      if len(elems) == 2:
        adj_elem[elems[0]] += [elems[1]]
        adj_elem[elems[1]] += [elems[0]]
    #
    # Partition mesh on dual graph
    _, partition_inds_of_elems = pymetis.part_graph(ndoms, adj_elem)
    #
    # Define subdomains
    subdomain_marker = fe.MeshFunction('size_t', mesh, mesh.topology().dim())
    subdomain_marker.array()[:] = partition_inds_of_elems
  else:
    subw = 1. / ndom
    subdomain = []
    if nd == 1:
      for idom in range(ndom):
        cond += '(xL - tol <= x[0]) && '
        cond += '(x[0] <= xR + tol)'
        subdomain += [fe.CompiledSubDomain(cond, xL=idom*subw, xR=(idom+1)*subw, tol=tol)]
    elif nd == 2:
      for idom in range(ndom):
        for jdom in range(ndom):
          cond =  '(xL - tol <= x[0]) && '
          cond += '(x[0] <= xR + tol) && '
          cond += '(yD - tol <= x[1]) && '
          cond += '(x[1] <= yU + tol)'
          subdomain += [fe.CompiledSubDomain(cond, xL=idom*subw, xR=(idom+1)*subw, yD=jdom*subw, yU=(jdom+1)*subw, tol=tol)]
    elif nd == 3:
      for idom in range(ndom):
        for jdom in range(ndom):
          for kdom in range(ndom):
            cond =  '(xL - tol <= x[0]) && '
            cond += '(x[0] <= xR + tol) && '
            cond += '(yD - tol <= x[1]) && '
            cond += '(x[1] <= yU + tol) && '
            cond += '(zD - tol <= x[2]) && '
            cond += '(x[2] <= zU + tol)'
            subdomain += [fe.CompiledSubDomain(cond, xL=idom*subw, xR=(idom+1)*subw, yD=jdom*subw, yU=(jdom+1)*subw, zD=kdom*subw, zU=(kdom+1)*subw, tol=tol)]
  print("%d subdomains created." %(ndoms))
  #
  # Define submeshes and local function spaces
  submesh, subV = [], []
  subdof2dof, subvrt2subdof = [], []
  subDoF, subarea = [], []
  for idom in range(ndoms):
    if pymetis_is_available:
      submesh += [fe.SubMesh(mesh, subdomain_marker, idom)]
      if False:
        to_vtk(submesh[idom], 'submesh%d' %idom)
    else:
      submesh += [fe.SubMesh(mesh, subdomain[idom])]
    subV += [fe.FunctionSpace(submesh[idom], 'CG', 1)]
    subdof2subvrt = fe.dof_to_vertex_map(subV[idom])
    subvrt2vrt = submesh[idom].data().array('parent_vertex_indices', 0)
    subdof2dof += [vrt2dof[subvrt2vrt[subdof2subvrt]]]    
    subvrt2subdof += [fe.vertex_to_dof_map(subV[idom])]
    subDoF += [subV[idom].dim()]
    subarea += [fe.assemble(1 * fe.dx(submesh[idom]))]
  repeats = np.zeros(DoF)
  for idom in range(ndoms):
    repeats[subdof2dof[idom]] += 1. 
  print("Submeshes and local function spaces defined.")
  print("min(subDoF) = %d, mean(subDoF) = %d, max(subDoF) = %d." %(min(subDoF), np.mean(subDoF), max(subDoF)))
  #
  # Assemble/solve local eigenvalue problems & define local function subspaces
  w, W, n_al = [], [], []
  subK = []
  t0 = time.time()
  for idom in range(ndoms):
    u, v = fe.TrialFunction(subV[idom]), fe.TestFunction(subV[idom])
    # Assemble local mass matrix M
    subM = fe.assemble(u * v * fe.dx(submesh[idom])).array()
    # Assemble local kernel matrix K  
    R = np.zeros((subDoF[idom], subDoF[idom]))
    for i, xi in enumerate(submesh[idom].coordinates()):
      if nd == 1:
        k_nd.y0 = xi[0]
      elif nd == 2:
        k_nd.y0, k_nd.y1 = xi
      elif nd == 3:
        k_nd.y0, k_nd.y1, k_nd.y2 = xi
      R[subvrt2subdof[idom][i], :] = fe.assemble(k_nd * v * fe.dx(submesh[idom])).get_local()
    subK += [np.zeros((subDoF[idom], subDoF[idom]))]
    R_func = fe.Function(subV[idom])
    for j in range(subDoF[idom]):
      R_func.vector()[:] = np.array(R[:, j])
      subK[idom][:, j] = fe.assemble(R_func * v * fe.dx(submesh[idom])).get_local()
    # Solve local generalized eigenvalue problem K.W = M.W.diag(w)
    _w, _W = LA.eigh(subK[idom], subM)
    # Truncate expansion
    trunc_subn_al = __truncate_KL(_w, delta2, sig2, subarea[idom])
    selected_modes = range(subDoF[idom]-1, subDoF[idom]-1-trunc_subn_al, -1)
    w += [_w[selected_modes]]
    W += [_W[:, selected_modes]]
    n_al += [len(selected_modes)]
  message = ""
  energy = sum([__w.sum() for  __w in w])/sum(subarea)
  if energy < (1. - delta2) * sig2:
    message = "Warning: Lack of local RVs to reach target variance within prescribed accuracy "
  else:
    message = "Local RVs reach target variance within prescribed accuracy "
  message += "(energy/sig2 = %g/%g)." % (energy, sig2)
  print("Local eigenvalue problems assembled and solved in %g s." % (time.time()-t0))
  print("min(n_al) = %d, mean(n_al) = %d, max(n_al) = %d." % (min(n_al), np.mean(n_al), max(n_al)))
  print(message)
  #
  # Assemble reduced global eigenvalue problem K.W = M.W.diag(w)
  nt = sum(n_al)
  K = np.zeros((nt, nt))
  t0 = time.time()
  for idom in range(ndoms):
    for jdom in range(idom+1, ndoms):
      R = np.zeros((subDoF[idom], subDoF[jdom]))         # R_ij = \int C(x_i, y) * phi_j(y) dy
      v = fe.TestFunction(subV[jdom])
      for i, xi in enumerate(submesh[idom].coordinates()):
        if nd == 1:
          k_nd.y0 = xi[0]
        elif nd == 2:
          k_nd.y0, k_nd.y1 = xi
        elif nd == 3:
          k_nd.y0, k_nd.y1, k_nd.y2 = xi
        R[subvrt2subdof[idom][i], :] = fe.assemble(k_nd * v * fe.dx).get_local()
      _K = np.zeros((subDoF[idom], subDoF[jdom]))         # K_ij = \int phi_i(x) * ( \int C(x, y) * phi_j(y) dy ) dx
      R_func = fe.Function(subV[idom])
      v = fe.TestFunction(subV[idom])
      for j in range(subDoF[jdom]):
        R_func.vector()[:] = np.array(R[:, j]) 
        _K[:, j] = fe.assemble(R_func * v * fe.dx).get_local()
      K[sum(n_al[:idom]):sum(n_al[:idom+1]), sum(n_al[:jdom]):sum(n_al[:jdom+1])] = W[idom].T.dot(_K.dot(W[jdom]))
  K += K.T
  for idom in range(ndoms):
    K[sum(n_al[:idom]):sum(n_al[:idom+1]), sum(n_al[:idom]):sum(n_al[:idom+1])] = W[idom].T.dot(subK[idom].dot(W[idom]))
  print("Reduced eigenvalue problem assembled with %d DoF in %g s." % (sum(n_al), time.time()-t0))
  #
  # Solve reduced global eigenvalue problem K.W = M.W.diag(w)
  t0 = time.time()
  w_reduced, W_reduced = LA.eigh(K)
  # Truncate expansion
  _nt = __truncate_KL(w_reduced, delta2, sig2, sum(subarea))
  selected_modes = range(nt-1, nt-1-_nt, -1) 
  w_reduced = w_reduced[selected_modes]
  W_reduced = W_reduced[:, selected_modes]
  nt = _nt
  _W = np.zeros((DoF, nt))
  for i in range(nt):
    for idom in range(ndoms):
      for al in range(n_al[idom]):
        _W[subdof2dof[idom], i] += W_reduced[sum(n_al[:idom])+al, i]*W[idom][:, al]
    _W[:, i] /= repeats
  message = ""
  energy = w_reduced.sum() / sum(subarea)
  if energy < (1. - delta2) * sig2:
    message = "Warning: %d global RVs are not enough to reach target variance within prescribed accuracy " % nt
  else:
    message = "%d global RVs reach target variance within prescribed accuracy " % nt
  message += "(energy/sig2 = %g/%g)." % (energy, sig2)
  print("Reduced eigenvalue problem solved in %g s." % (time.time() - t0))
  print(message)
  return w_reduced, _W, mesh

def get_KL(nd, nEl, ndom, sig2, L, model, delta2, path='./', path_to_meshes=None, type_domain='block', save_on_disk=True):
  """
  Computes or load a KL.
  """

  # Check function call
  possible_request = False
  if (type_domain == 'block') & (nd in (1, 2, 3)):
    possible_request = True
  elif (type_domain in ('disk', 'ellipse', 'ublock')) & (nd == 2):
    possible_request = True
  else:
    print("type_domain = %s is not supported for nd = %d" %(type_domain, nd))
    return 3 * [None]
  #
  # Try loading a previously computed and saved KL
  try:
    w, W, mesh = load_KL(nd, nEl, sig2, L, model, delta2, type_domain=type_domain, path=path, path_to_meshes=path_to_meshes)
  #
  # Compute a new KL
  except:
    if (ndom > 1):
      w, W, mesh = __compute_KL_DD(nd, nEl, ndom, sig2, L, model, delta2, type_domain)
    else:
      w, W, mesh = __compute_KL(nd, nEl, sig2, L, model, delta2, type_domain)
    if save_on_disk:
      save_KL(w, W, mesh, nd, nEl, sig2, L, model, delta2, type_domain=type_domain, path=path, path_to_meshes=path_to_meshes)
  return w, W, mesh
