from scipy.interpolate import splprep, splev
import numpy as np
import matplotlib.pyplot as plt
import os
import read_airfoil


class Interpolate:
    def __init__(self, airfoil_path, airfoil_name):
        """Initialize with airfoil coordinates."""
        xcoords, ycoords = read_airfoil.read_airfoil_coordinates(airfoil_path, airfoil_name)
        self.xcoords = xcoords
        self.ycoords = ycoords

    def remove_duplicates(self, x, y):
        """Remove duplicate x-values while preserving corresponding y-values."""
        unique_x, index = np.unique(x, return_index=True)
        unique_y = np.array(y)[index]  # Ensure y is a Numpy array for proper indexing
        return unique_x, unique_y

    def refine_leading_edge(self, x, n_points):
        """
        Refine the leading edge using cosine interpolation for better resolution in curved regions.

        Parameters:
        - x: Input x-coordinates to refine.
        - n_points: Number of points for refinement.

        Returns:
        - Refined x-coordinates with cosine spacing.
        """
        beta = np.linspace(0, np.pi, n_points)  # Cosine spacing
        refined_x = 0.5 * (1 - np.cos(beta)) * np.max(x)
        return refined_x

    def airfoil_interpolate(self, n_points, meth, foil, airfoil_path, airfoil_name, int_path):
        """Generate interpolated airfoil coordinates."""
        file_path = os.path.join(airfoil_path, airfoil_name)
        coordinates = list(zip(self.xcoords, self.ycoords))
        leading_edge_index = np.argmin(self.xcoords)
        upper_surface = coordinates[:leading_edge_index + 1]
        lower_surface = coordinates[leading_edge_index:]
        lower_surface = lower_surface[::-1]
        x_interp_u, y_interp_u = zip(*upper_surface)
        x_interp_l, y_interp_l = zip(*lower_surface)
        x_interp_u, y_interp_u = self.remove_duplicates(x_interp_u, y_interp_u)
        x_interp_l, y_interp_l = self.remove_duplicates(x_interp_l, y_interp_l)
        x_interp_u = np.array(x_interp_u, dtype=float)
        y_interp_u = np.array(y_interp_u, dtype=float)
        x_interp_l = np.array(x_interp_l, dtype=float)
        y_interp_l = np.array(y_interp_l, dtype=float)

        if len(x_interp_u) != len(y_interp_u):
            raise ValueError("Upper surface x and y arrays must have the same length.")
        if len(x_interp_l) != len(y_interp_l):
            raise ValueError("Lower surface x and y arrays must have the same length.")

        # Refine leading edge with cosine interpolation
        upper_refined_x = self.refine_leading_edge(x_interp_u, len(y_interp_u))
        lower_refined_x = self.refine_leading_edge(x_interp_l, len(y_interp_l))

        # Interpolate y-coordinates for refined x
        y_interp_u = np.interp(upper_refined_x, x_interp_u, y_interp_u)
        y_interp_l = np.interp(lower_refined_x, x_interp_l, y_interp_l)

        # Use splprep for smoothing and interpolation
        tck_u, u_u = splprep([upper_refined_x, y_interp_u], k=3, s=0)
        tck_l, u_l = splprep([lower_refined_x, y_interp_l], k=3, s=0)
        u = np.linspace(0, 1, n_points)
        x_u, y_u = splev(u, tck_u)
        x_l, y_l = splev(u, tck_l)

        # Combine points and save
        upper_surface = [[x, y] for x, y in zip(x_u, y_u)]
        lower_surface = [[x, y] for x, y in zip(x_l, y_l)]

        # Reverse upper surface (so it goes from TE to LE)
        upper_surface = upper_surface[::-1]
        upper_te_point = upper_surface[0]  # Trailing edge upper
        lower_te_point = lower_surface[-1]  # Trailing edge lower

        # Calculate averaged trailing edge point
        te_x = (upper_te_point[0] + lower_te_point[0]) / 2
        te_y = (upper_te_point[1] + lower_te_point[1]) / 2

        # Set both trailing edge points to the same value
        upper_surface[0] = [te_x, te_y]
        lower_surface[-1] = [te_x, te_y]

        # Get the leading edge points
        upper_le_point = upper_surface[-1]  # Leading edge upper
        lower_le_point = lower_surface[0]   # Leading edge lower

        # Calculate averaged leading edge point
        le_x = (upper_le_point[0] + lower_le_point[0]) / 2
        le_y = (upper_le_point[1] + lower_le_point[1]) / 2

        # Set both leading edge points to the same value
        upper_surface[-1] = [le_x, le_y]
        lower_surface[0] = [le_x, le_y]

        self.upper_surface = upper_surface
        self.lower_surface = lower_surface
        
        self.points = self.upper_surface + self.lower_surface[1:]

        x_new = np.concatenate((x_u, x_l))
        y_new = np.concatenate((y_u, y_l))
        data = np.array(self.points)  
        print(f"First point (TE upper): {data[0]}")
        print(f"Last point (TE lower): {data[-1]}")
        print(f"Distance between TE points: {np.linalg.norm(data[0] - data[-1])}")
        
        data_closed = np.vstack([data, data[0]])
        
        print("Extracted points:\n", data_closed)
        x_points = data_closed[:, 0]  
        y_points = data_closed[:, 1]
        os.chdir(int_path)
        np.savetxt('output.dat', data_closed, header="r", comments="")

        return x_points, y_points

    def get_surface(self):
        """Return the lower surface values."""
        return self.upper_surface, self.lower_surface