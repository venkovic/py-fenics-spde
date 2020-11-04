import os, sys
import numpy as np 
import mesher_ellipse, mesher_square

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

try:
  import matplotlib.pyplot as plt
  import matplotlib.tri as tri
  import matplotlib.colors as colors
  plt.set_cmap('jet')
  mpl_is_available = True
except:
  mpl_is_available = False


home_dir = os.getenv('HOME')
tol = 1E-14


def get_mesh_fname(nd, nEl, type_domain):
  return '%dD_%s_%d' % (nd, type_domain, nEl)

def save_mesh(mesh, nd, nEl, type_domain, path='./'):
  if path[-1] != '/': 
    path += '/'
  fname = get_mesh_fname(nd, nEl, type_domain)
  mesh_file = fe.File(path + fname + '.xml')
  mesh_file << mesh


def load_mesh(nd, nEl, type_domain, path='./'):
  if path[-1] != '/':
    path += '/'
  fname = get_mesh_fname(nd, nEl, type_domain)
  mesh = fe.Mesh(path + fname + '.xml')
  print('Loaded %s%s.xml.' % (path, fname))
  return mesh


def get_mesh(nd, nEl, type_domain):
  # Create mesh
  if nd == 1:
    mesh = fe.UnitIntervalMesh(nEl)
  elif nd == 2:
    if type_domain == 'block':
      mesh = fe.UnitSquareMesh(nEl, nEl)
    elif type_domain == 'disk':
      mesh = fe.UnitDiscMesh.create(MPI.comm_world, nEl//2, 1, nd)
    elif type_domain == 'ellipse':
      pts, cells = mesher_ellipse.create_mesh(num_boundary_points=nEl)
      mesher_ellipse.write("ellipse", pts, cells)
      mesh = fe.Mesh("ellipse.xml")
    elif type_domain == 'ublock':
      pts, cells = mesher_square.create_mesh(DOF=nEl)
      mesher_ellipse.write("ublock", pts, cells)
      mesh = fe.Mesh("ublock.xml")
  elif nd == 3:
    mesh = fe.BoxMesh(fe.Point(0, 0, 0), fe.Point(1, 1, 1), nEl, nEl, nEl)
  return mesh


def to_vtk(KL_real, fname='KL_real'):
  # Export data for VTK
  vtkfile = fe.File(fname+'.pvd')
  vtkfile << KL_real


def to_file(KL_real, mesh, V=None, fname='KL_real', ext='png'):
  if mpl_is_available:
    #
    # Create the triangulation
    nvrt = mesh.num_vertices()
    nd_geo = mesh.geometry().dim()
    mesh_coordinates = mesh.coordinates().reshape((nvrt, nd_geo))
    triangles = np.asarray([cell.entities(0) for cell in fe.cells(mesh)])
    triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)
    V = fe.FunctionSpace(mesh, "CG", 1)
    vrt2dof = fe.vertex_to_dof_map(V)
    #
    # Plot realization
    fig, ax = plt.subplots()
    if False:
      ax.set_aspect('equal')
      im = ax.tripcolor(triangulation, KL_real.vector()[vrt2dof], norm=colors.Normalize(vmin=-3.2,vmax=3.2))
      ax.axis('off')
    else:
      im = ax.tripcolor(triangulation, KL_real.vector()[vrt2dof], norm=colors.LogNorm(vmin=np.exp(-3.2),vmax=np.exp(3.2)))
      #cb_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])
      #cbar = fig.colorbar(im, cax=cb_ax)
      #ax.axis('off')
      ax.set_xlim((0, 1))
      ax.set_xticks((0, .25, .5, .75, 1))
      #ax.set_xticklabels((r'$0$', '', r'$0.5$', '', r'$1$'))
      ax.set_xticklabels(('', '', '', '', ''))
      ax.set_ylim((0, 1))
      ax.set_yticks((0, .25, .5, .75, 1))
      #ax.set_yticklabels((r'$0$', r'$0.25$', r'$0.50$', r'$0.75$', r'$1$'))
      ax.set_yticklabels(('', '', '', '', ''))
    #
    plt.savefig('%s.%s' %(fname, ext), bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()


def to_file2(KL_reals, mesh, fname='KL_real', ext='png', labels=None):
  if mpl_is_available:
    plt.rcParams['text.usetex'] = True
    params={'text.latex.preamble':[r'\usepackage{amssymb}',r'\usepackage{amsmath}']}
    plt.rcParams.update(params)
    plt.rcParams['axes.labelsize']=20#19.
    plt.rcParams['axes.titlesize']=20#19.
    plt.rcParams['legend.fontsize']=20.#16.
    plt.rcParams['xtick.labelsize']=20
    plt.rcParams['ytick.labelsize']=20
    plt.rcParams['legend.numpoints']=1  
    #
    # Create the triangulation
    nvrt = mesh.num_vertices()
    nd_geo = mesh.geometry().dim()
    mesh_coordinates = mesh.coordinates().reshape((nvrt, nd_geo))
    triangles = np.asarray([cell.entities(0) for cell in fe.cells(mesh)])
    triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)
    if V is not None:
      V = fe.FunctionSpace(mesh, "CG", 1)
    vrt2dof = fe.vertex_to_dof_map(V)
    #
    # Plot realization

    nreals = len(KL_reals)
    #titles = (r'$\log a(x;\boldsymbol{\xi}_{s-10})$',
    #          r'$\log a(x;\boldsymbol{\xi}_{s-5})$',
    #          r'$\log a(x;\boldsymbol{\xi}_{s})$',
    #          r'$\log a(x;\boldsymbol{\xi}_{s+5})$',
    #          r'$\log a(x;\boldsymbol{\xi}_{s+10})$')
    #titles = (r'$a(x;\boldsymbol{\xi}_{s})$',
    #          r'$a(x;\boldsymbol{\xi}_{s+10})$',
    #          r'$a(x;\boldsymbol{\xi}_{s+100})$',
    #          r'$a(x;\boldsymbol{\xi}_{s+1000})$')
    titles = []
    for i in range(nreals):
      titles += [r'$\|a_p-\hat{a}\|_{\Omega} = %g$' % labels[i]]

    fig, axes = plt.subplots(1 ,nreals, figsize=(12.5, 2.75), sharey=True)
    plt.subplots_adjust(wspace=.1, hspace=.7)
    for i, ax in enumerate(axes):
      ax.set_title(titles[i])
      ax.set_xlim((0, 1))
      ax.set_xticks((0, .25, .5, .75, 1))
      #ax.set_xticklabels((r'$0$', '', r'$0.5$', '', r'$1$'))
      ax.set_xticklabels(('', '', '', '', ''))
      ax.set_ylim((0, 1))
      ax.set_yticks((0, .25, .5, .75, 1))
      #ax.set_yticklabels((r'$0$', r'$0.25$', r'$0.50$', r'$0.75$', r'$1$'))
      ax.set_yticklabels(('', '', '', '', ''))
      #ax.set_aspect('equal')
      #im = ax.tripcolor(triangulation, KL_reals[i].vector()[vrt2dof], norm=colors.Normalize(vmin=-3.2,vmax=3.2))
      im = ax.tripcolor(triangulation, KL_reals[i].vector()[vrt2dof], norm=colors.LogNorm(vmin=np.exp(-3.2),vmax=np.exp(3.2)))
      #ax.axis('off')

    cb_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)

    plt.savefig('%s.%s' %(fname, ext), bbox_inches='tight')
    plt.close(fig)