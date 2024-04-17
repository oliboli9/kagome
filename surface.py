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
        dz_dz = -1 * np.ones(len(dz_dx))

        # print(f"x:{len(dz_dx)}")
        # print(f"y:{len(dz_dy)}")
        # print(f"z:{len(dz_dz)}")

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


class SphereSurface:
    def __init__(self, r, centre, density=None, n=None):
        self.n = n
        self.r = r
        self.centre = np.array(centre)
        theta, phi = sp.symbols("theta phi")
        cx, cy, cz = centre
        self.f = (
            (self.r * sp.cos(theta) * sp.sin(phi)) + cx,
            (self.r * sp.sin(theta) * sp.sin(phi)) + cy,
            (self.r * sp.cos(phi)) + cz,
        )
        self.f_num = sp.lambdify((theta, phi), self.f, "numpy")

        surface_area = 4 * np.pi * r**2

        if density is not None:
            self.n = int(surface_area * density)
        elif n is not None:
            self.n = n
            self.density = surface_area / n
        else:
            raise ValueError("Either density or number of atoms must be provided")

    def normals(self, positions):
        vectors = positions - self.centre
        normals = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
        return normals

    def elevate_to_surface(self, positions):
        def closest_point(point):
            point = np.array(point)
            vector = point - self.centre
            # Normalize the vector
            unit_vector = vector / np.linalg.norm(vector)
            # Scale by the sphere's radius and adjust by the sphere's center
            closest_point_on_sphere = self.centre + unit_vector * self.r
            return closest_point_on_sphere

        return np.array([closest_point(pos) for pos in positions])


class TorusSurface:
    def __init__(self, R, r, centre, density=None, n=None):
        self.R = R  # Major radius
        self.r = r  # Minor radius
        self.centre = np.array(centre)
        theta, phi = sp.symbols("theta phi")
        cx, cy, cz = centre
        self.f = (
            (self.R + self.r * sp.cos(theta)) * sp.cos(phi) + cx,
            (self.R + self.r * sp.cos(theta)) * sp.sin(phi) + cy,
            (self.r * sp.sin(theta)) + cz,
        )
        self.f_num = sp.lambdify((theta, phi), self.f, "numpy")

        # Surface area of torus S = 4*pi^2*R*r
        surface_area = 4 * np.pi**2 * R * r

        if density is not None:
            self.n = int(surface_area * density)
        elif n is not None:
            self.n = n
            self.density = surface_area / n
        else:
            raise ValueError("Either density or number of atoms must be provided")

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
