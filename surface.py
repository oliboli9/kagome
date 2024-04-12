import math
import numpy as np
from scipy.integrate import dblquad
import sympy as sp
from sympy.vector import CoordSys3D
from scipy.optimize import minimize


def cartesian_to_spherical(x, y, z):
    rho = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / rho)
    phi = np.arctan2(y, x)
    return rho, theta, phi


def spherical_to_cartesian(rho, theta, phi):
    x = rho * np.sin(theta) * np.cos(phi)
    y = rho * np.sin(theta) * np.sin(phi)
    z = rho * np.cos(theta)
    return x, y, z


class Surface:
    def elevate_to_surface():
        """
        Return array of closest (cartesian) points on surface to given array of positions
        """
        pass

    def normals():
        """
        Return array of normal vectors to given array of positions
        """
        pass


class PeriodicSurface:
    def __init__(self, f, density=None, n=None):
        """
        The number of atoms n is such that the density of atoms on the surface is 1/9 atoms/Å^2
        (optimal density is based on # of Au atoms that make a lattice with Au bond distance)
        """
        x, y = sp.symbols("x y")
        self.f = f(x, y)
        self.density = density
        self.df_dx = sp.diff(self.f, x)
        self.df_dy = sp.diff(self.f, y)
        area_expr = sp.lambdify(
            (x, y), sp.sqrt(1 + self.df_dx**2 + self.df_dy**2), "numpy"
        )
        area, error = dblquad(area_expr, 0, 30, lambda x: 0, lambda x: 30)
        if n is None:
            assert (
                density is not None
            ), "Pass either desired density or desired number of atoms"
            self.n = int(area / density)
        else:
            self.n = n
            self.density = area / n

    def elevate_to_surface(self, positions):
        x_vals = positions[:, 0]
        y_vals = positions[:, 1]
        x, y = sp.symbols("x y")
        f_num = sp.lambdify((x, y), self.f, "numpy")
        return np.vstack((x_vals, y_vals, (f_num(x_vals, y_vals)))).T

    def normals(self, positions):
        x = positions[:, 0]
        y = positions[:, 1]

        dz_dx = self.dx(x, y)
        dz_dy = self.dy(x, y)
        dz_dz = -1 * np.ones(self.n)

        normals = np.stack((dz_dx, dz_dy, dz_dz), axis=-1)
        return normals

    def dx(self, x_vals, y_vals):
        x, y = sp.symbols("x y")
        df_dx_num = sp.lambdify((x, y), self.df_dx, "numpy")
        return df_dx_num(x_vals, y_vals)

    def dy(self, x_vals, y_vals):
        x, y = sp.symbols("x y")
        df_dy_num = sp.lambdify((x, y), self.df_dy, "numpy")
        return df_dy_num(x_vals, y_vals)


class TorusSurface:
    def __init__(self, R, r, centre, n):
        self.R = R  # Major radius
        self.r = r  # Minor radius
        self.n = n
        theta, phi, R, r = sp.symbols("theta phi R r")
        self.centre = np.array(centre)
        cx, cy, cz = centre
        self.f = (
            (self.R + self.r * sp.cos(phi)) * sp.cos(theta) + cx,
            (self.R + self.r * sp.cos(phi)) * sp.sin(theta) + cy,
            (self.r * sp.sin(phi)) + cz,
        )
        self.f_num = sp.lambdify((theta, phi), self.f, "numpy")

    def normals(self, positions):
        rho_vals, theta_vals, phi_vals = cartesian_to_spherical(
            positions[:, 0], positions[:, 1], positions[:, 2]
        )
        theta, phi = sp.symbols("theta phi")
        N = CoordSys3D("N")

        x, y, z = self.f
        P = x * N.i + y * N.j + z * N.k

        P_theta = P.diff(theta)
        P_phi = P.diff(phi)

        normals = P_theta.cross(P_phi)
        normals_matrix = np.empty((len(theta_vals), 3), dtype=object)

        for idx, (t_val, p_val) in enumerate(zip(theta_vals, phi_vals)):
            normal_vector = normals.subs({theta: t_val, phi: p_val}).doit()
            normals_matrix[idx, :] = [
                normal_vector.dot(N.i),
                normal_vector.dot(N.j),
                normal_vector.dot(N.k),
            ]

        return np.array(normals_matrix, dtype=float)

    def elevate_to_surface(self, positions):
        def closest_point(x0, y0, z0):
            point0 = np.array((x0, y0, z0))

            def torus_surface_distance(theta_phi):
                theta, phi = theta_phi
                newpoint = self.f_num(theta, phi)
                return np.sum((newpoint - point0) ** 2)

            initial_guess = [0, 0]
            result = minimize(
                torus_surface_distance,
                initial_guess,
                # args=(x0, y0, z0),
                # method="L-BFGS-B",
                # bounds=[(0, 2 * np.pi), (0, 2 * np.pi)],
            )
            closest_theta_phi = result.x
            closest_point = self.f_num(closest_theta_phi[0], closest_theta_phi[1])
            # print(x0, y0, z0, closest_point)
            return closest_point

        return np.array([closest_point(x, y, z) for x, y, z in positions])

    def distance_to_torus(self, point):
        """
        Calculate the distance from a point (x, y, z) to the surface of a torus centered at (cx, cy, cz).

        :param (x, y, z): Coordinates of a point in 3D space.
        :param center: Coordinates of the center of the torus (cx, cy, cz).
        :return: Distance from the point to the torus surface.
        """
        x, y, z = np.array(point) - np.array(self.centre)
        dist = (self.R - np.sqrt(x**2 + y**2)) ** 2 + z**2 - self.r**2
        return dist

    def generate_torus_level_set(self, grid_size, bounds):
        """
        Generate a 3D level set of a torus.

        :param grid_size: The number of points along each dimension in the grid.
        :param bounds: The bounds of the grid in the format (xmin, xmax, ymin, ymax, zmin, zmax).
        :param R: Major radius of the torus.
        :param r: Minor radius of the torus.
        :return: 3D numpy array representing the level set.
        """
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        x = np.linspace(xmin, xmax, grid_size)
        y = np.linspace(ymin, ymax, grid_size)
        z = np.linspace(zmin, zmax, grid_size)

        level_set = np.empty((grid_size, grid_size, grid_size))

        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    level_set[i, j, k] = self.distance_to_torus(
                        [x[i], y[j], z[k]],
                    )

        return level_set


class SurfaceConstraint:
    def __init__(self, surface):
        self.surface: Surface = surface

    def adjust_positions(self, atoms, newpositions):
        surface_positions = self.surface.elevate_to_surface(newpositions)
        newpositions[:] = surface_positions

    def adjust_forces(self, atoms, forces):
        # Modify the forces to be tangential to the surface
        positions = atoms.get_positions()
        normals = self.surface.normals(positions)
        normals /= np.linalg.norm(normals, axis=1).reshape(-1, 1)
        # forces -= np.dot(forces, normals) * normals
        forces -= np.einsum("ij,ij->ij", forces, normals) * normals
