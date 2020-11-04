import numpy as np
from scipy import special
import meshpy.triangle
import meshio

# Ref: https://github.com/nschloe/meshzoo/blob/master/examples/meshpy/rectangle.py

def get_approx_max_area(DOF):
  return 1./(1.2789 * (DOF - 816) + 1e3)

def create_mesh(edgelength=1.0, DOF=1000):
  #
  # Get maximal element area
  max_area = get_approx_max_area(DOF)
  #
  # dimensions of the rectangle
  lx = edgelength
  ly = edgelength
  #
  # corner points
  boundary_points = [(0, 0), (lx, 0), (lx, lx), (0, lx)]
  #
  info = meshpy.triangle.MeshInfo()
  info.set_points(boundary_points)
  #
  def _round_trip_connect(start, end):
    result = []
    for i in range(start, end):
      result.append((i, i + 1))
    result.append((end, start))
    return result
  #
  info.set_facets(_round_trip_connect(0, len(boundary_points) - 1))
  #
  def _needs_refinement(vertices, area):
    return bool(area > max_area)
  #
  meshpy_mesh = meshpy.triangle.build(info, refinement_func=_needs_refinement)
  #
  # append column
  pts = np.array(meshpy_mesh.points)
  points = np.c_[pts[:, 0], pts[:, 1]]
  #
  return points, np.array(meshpy_mesh.elements)

def write(fname, points, cells, ext='xml'):
  meshio.write_points_cells('%s.%s' %(fname, ext), points, {"triangle": cells})

#pts, cells = create_mesh(edgelength=1.0, DOF=1e4)
#write('square_mesh', pts, cells, ext='xml')
