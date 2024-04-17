from scipy.spatial import Delaunay, ConvexHull
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import alphashape


class Triangulation:
    def __init__(self):
        pass

    def find_neighbors(self, simplices, points):
        """
        List of list of neighbouring points for each point in triangulation
        """
        neighbors = [[] for _ in range(len(points))]
        for simplex in simplices:
            for i, j in zip(simplex, simplex[[1, 2, 0]]):
                neighbors[i].append(j)
                neighbors[j].append(i)
        return [list(set(nbr)) for nbr in neighbors]


class SurfaceTriangulation(Triangulation):
    def triangulate(self, points):
        """
        Computes Delaunay triangulation for surface. Assumes closest plane to be xy-plane.
        TODO: Find actual closest plane to project points onto

        Parameters:

        points:
            Nx3 array of points on surface
        """
        points_2d = points[:, :2]
        tri = Delaunay(points_2d)
        return tri.simplices


class ConvexTriangulation(Triangulation):
    def triangulate(self, points):
        hull = ConvexHull(points)
        return hull.simplices


class NonConvexTriangulation(Triangulation):
    """
    Prerequisites:

    Export points to MATLAB.
    Compute alpha shape.
    Import simplices and points as triangles.csv and alphapoints.csv respectively
    """

    def triangulate(self, points, alpha):
        # simplices = np.loadtxt("simplices.csv", delimiter=",", dtype="int") - 1
        # coords = np.loadtxt("alphapoints.csv", delimiter=",", dtype="float")
        points = [tuple(row) for row in points]
        alpha_shape = alphashape.alphashape(points, alpha)
        # alpha_shape.show()
        simplices = alpha_shape.faces
        return simplices
