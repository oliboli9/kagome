from scipy.spatial import Delaunay
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

        Parameters:
        ----------
        points:
            Nx3 array of points on surface

        Returns:
        ----------
        Array of simplex indices
        """
        points_2d = points[:, :2]
        tri = Delaunay(points_2d)
        return tri.simplices


class ConvexTriangulation(Triangulation):
    def triangulate(self, points):
        """
        Computes 3D Delaunay triangulation for surface.

        Parameters:
        ----------
        points:
            Nx3 array of points on surface

        Returns:
        ----------
        Array of simplex indices
        """
        triangulation = Delaunay(points)
        convex_hull = triangulation.convex_hull
        return convex_hull


class NonConvexTriangulation(Triangulation):
    def triangulate(self, points, alpha=None):
        """
        Computes 3D Delaunay triangulation for non-convex surface.

        Parameters:
        ----------
        points:
            Nx3 array of points on surface
        alpha: float
            alpha shape parameter such that an edge of a disk of radius 1/alpha can be drawn between any two edge members
            of a set of points and still contain all the points

        Returns:
        ----------
        Array of simplex indices
        """
        points = [tuple(row) for row in points]
        alpha_shape = alphashape.alphashape(points, alpha)
        # alpha_shape.show()
        simplices = alpha_shape.faces
        return simplices
