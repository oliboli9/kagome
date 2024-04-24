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
            # Scale by the sphere's radius and adjust by the sphere's centre
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


class CapsuleSurface:
    def __init__(self, r, h, centre, density=None, n=None):
        self.r = r  # Radius of the hemispheres and cylinder
        self.h = h  # Height of the cylindrical part (excluding hemispheres)
        self.centre = np.array(centre)  # Centre of the capsule
        surface_area = 2 * np.pi * r * h + 4 * np.pi * r**2  # Total surface area

        if density is not None:
            self.n = int(
                surface_area * density
            )  # Number of points based on surface area and density
        elif n is not None:
            self.n = n
            self.density = surface_area / n
        else:
            raise ValueError("Either density or number of atoms must be provided")

    def point_to_params(self, point):
        x, y, z = point - self.centre
        half_height = self.h / 2

        if -half_height <= z <= half_height:
            # Point is on the cylinder
            theta = np.arctan2(y, x)
            return ("cylinder", theta, None, z)
        else:
            # Point is on one of the hemispheres
            if z > half_height:
                z_hemisphere = z - half_height
                part = "top hemisphere"
            else:
                z_hemisphere = z + half_height
                part = "bottom hemisphere"

            r_point = np.sqrt(x**2 + y**2 + z_hemisphere**2)
            theta = np.arccos(z_hemisphere / r_point)
            phi = np.arctan2(y, x)
            return (part, theta, phi, r_point)

    def params_to_point(self, part, theta, phi, rh):
        x = self.r * np.sin(theta) * np.cos(phi)
        y = self.r * np.sin(theta) * np.sin(phi)

        if part == "cylinder":
            z = rh
        elif part == "top hemisphere":
            z = self.h / 2 + self.r * np.cos(theta)
        elif part == "bottom hemisphere":
            z = -self.h / 2 - self.r * np.cos(theta)
        else:
            raise ValueError("Invalid part specification")

        return self.centre + np.array([x, y, z])

    def normals(self, positions):
        x, y, z = positions.T - self.centre
        normals = np.zeros_like(positions)
        upper_mask = z > self.h / 2
        lower_mask = z < -self.h / 2
        cylinder_mask = ~upper_mask & ~lower_mask

        normals[upper_mask] = positions[upper_mask] - (self.centre + [0, 0, self.h / 2])
        normals[lower_mask] = positions[lower_mask] - (
            self.centre + [0, 0, -self.h / 2]
        )
        normals[cylinder_mask] = np.column_stack(
            (x[cylinder_mask], y[cylinder_mask], np.zeros(np.sum(cylinder_mask)))
        )

        return normals

    def elevate_to_surface(self, positions):
        return np.array([self.closest_point_on_surface(pos) for pos in positions])

    def closest_point_on_surface(self, point):
        part, theta, phi, rh = self.point_to_params(point)
        theta = np.clip(theta, 0, np.pi)
        phi = phi % (2 * np.pi) if phi is not None else 0
        if part == "cylinder":
            rh = np.clip(rh, -self.h / 2, self.h / 2)
        return self.params_to_point(part, theta, phi, rh)

    # def elevate_to_surface(self, positions):
    #     cx, cy, cz = self.centre

    #     x = positions[:, 0] - cx
    #     y = positions[:, 1] - cy
    #     z = positions[:, 2] - cz

    #     # Distance from the point to the capsule's axis (ignoring z-component)
    #     dist_to_axis = np.sqrt(x**2 + y**2)

    #     # Allocate array for closest points
    #     closest_points = np.zeros_like(positions)

    #     # Cylinder part
    #     cylinder_mask = (z >= -self.h / 2) & (z <= self.h / 2)
    #     scale = self.r / dist_to_axis[cylinder_mask]
    #     closest_points[cylinder_mask, 0] = x[cylinder_mask] * scale
    #     closest_points[cylinder_mask, 1] = y[cylinder_mask] * scale
    #     closest_points[cylinder_mask, 2] = z[cylinder_mask]

    #     # Hemispheres
    #     for hemi_sign, z_bound in zip([1, -1], [self.h / 2, -self.h / 2]):
    #         hemisphere_mask = z * hemi_sign > self.h / 2
    #         sphere_centre_z = z_bound

    #         # Calculate vectors from hemisphere centres to points
    #         sphere_to_point_vectors = positions[hemisphere_mask] - np.array(
    #             [cx, cy, cz + sphere_centre_z]
    #         )
    #         norms = np.linalg.norm(sphere_to_point_vectors, axis=1)

    #         # Calculate normalized and scaled vectors
    #         normalized_scaled_vectors = (
    #             sphere_to_point_vectors / norms[:, np.newaxis] * self.r
    #         )
    #         closest_points[hemisphere_mask] = normalized_scaled_vectors + np.array(
    #             [cx, cy, cz + sphere_centre_z]
    #         )

    #     # Translate back to original coordinates
    #     closest_points += np.array([cx, cy, cz])

    #     return closest_points


class CylinderSurface:
    def __init__(self, r, h, centre, density=None, n=None):
        self.r = r  # radius
        self.h = h  # height
        self.centre = np.array(centre)
        surface_area = 2 * np.pi * r**2 + 2 * np.pi * r * h

        if density is not None:
            self.n = int(surface_area * density)
        elif n is not None:
            self.n = n
            self.density = surface_area / n
        else:
            raise ValueError("Either density or number of atoms must be provided")

    def elevate_to_surface(self, positions):
        def closest_point_on_cylinder(point):
            x, y, z = point
            (
                cx,
                cy,
                cz,
            ) = self.centre
            h = self.h
            r = self.r
            lower_z = cz - h / 2
            upper_z = cz + h / 2

            # Project (x, y) onto the circle of radius r centered at (cx, cy)
            dx, dy = x - cx, y - cy
            dist_to_center = np.sqrt(dx**2 + dy**2)
            projected_x = cx + r * (dx / dist_to_center)
            projected_y = cy + r * (dy / dist_to_center)

            # Clamp z to be within the bounds of the cylinder's height
            if z < lower_z:
                # Point is below the bottom cap
                projected_z = lower_z
                if dx**2 + dy**2 < r:
                    projected_x = x
                    projected_y = y
            elif z > upper_z:
                # Point is above the top cap
                projected_z = upper_z
                if dx**2 + dy**2 < r:
                    projected_x = x
                    projected_y = y
            else:
                # Point is within the height of the cylinder
                projected_z = z
                if r - dist_to_center > upper_z - z:
                    projected_x = x
                    projected_y = y
                    projected_z = upper_z
                if r - dist_to_center > z - lower_z:
                    projected_x = x
                    projected_y = y
                    projected_z = lower_z

            return projected_x, projected_y, projected_z

        return np.array([closest_point_on_cylinder(point) for point in positions])

    def normals(self, positions):
        def normal_vector(point):
            x, y, z = point
            (
                cx,
                cy,
                cz,
            ) = self.centre
            h = self.hr = self.r
            lower_z = cz - h / 2
            upper_z = cz + h / 2

            if z == lower_z:
                # Point is on the bottom cap
                return (0, 0, -1)
            elif z == upper_z:
                # Point is on the top cap
                return (0, 0, 1)
            else:
                # Point is on the curved surface
                dx, dy = x - cx, y - cy
                return (dx, dy, 0)

        return np.array([normal_vector(pos) for pos in positions])


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
