import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from math import factorial
import read_airfoil


class AirfoilNormalizer:
    """Normalize airfoil coordinates to standard format."""
    
    def __init__(self):
        self.le_index = None
        self.translation = None
        self.rotation_angle = None
        self.scale_factor = None
    
    def normalize(self, xcoords, ycoords):
        xcoords = np.array(xcoords)
        ycoords = np.array(ycoords)
        
        # Find leading edge
        self.le_index = np.argmin(xcoords)
        x_le = xcoords[self.le_index]
        y_le = ycoords[self.le_index]
        
        # Translate to origin
        x_translated = xcoords - x_le
        y_translated = ycoords - y_le
        self.translation = (x_le, y_le)
        
        # Find trailing edge
        x_te = (x_translated[0] + x_translated[-1]) / 2
        y_te = (y_translated[0] + y_translated[-1]) / 2
        
        # Rotate to align with x-axis
        self.rotation_angle = np.arctan2(y_te, x_te)
        cos_theta = np.cos(-self.rotation_angle)
        sin_theta = np.sin(-self.rotation_angle)
        
        x_rotated = x_translated * cos_theta - y_translated * sin_theta
        y_rotated = x_translated * sin_theta + y_translated * cos_theta
        
        # Scale to chord = 1
        chord_length = np.sqrt(x_te**2 + y_te**2)
        self.scale_factor = chord_length
        
        x_norm = x_rotated / chord_length
        y_norm = y_rotated / chord_length
        
        return x_norm, y_norm
    
    def denormalize(self, x_norm, y_norm):
        x_norm = np.array(x_norm)
        y_norm = np.array(y_norm)
        
        # Unscale
        x_scaled = x_norm * self.scale_factor
        y_scaled = y_norm * self.scale_factor
        
        # Unrotate
        cos_theta = np.cos(self.rotation_angle)
        sin_theta = np.sin(self.rotation_angle)
        
        x_unrotated = x_scaled * cos_theta - y_scaled * sin_theta
        y_unrotated = x_scaled * sin_theta + y_scaled * cos_theta
        
        # Untranslate
        x_original = x_unrotated + self.translation[0]
        y_original = y_unrotated + self.translation[1]
        
        return x_original, y_original


class CST:
    def __init__(self, airfoil_path, airfoil_name):
        xcoords, ycoords = read_airfoil.read_airfoil_coordinates(airfoil_path, airfoil_name)
        
        # Store original coordinates
        self.xcoords_original = xcoords
        self.ycoords_original = ycoords
        
        # Normalize coordinates
        self.normalizer = AirfoilNormalizer()
        self.xcoords, self.ycoords = self.normalizer.normalize(xcoords, ycoords)

    def ClassShape(self, w, x, N1, N2, dz):
        # Class function; taking input of N1 and N2
        C = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            C[i, 0] = x[i] ** N1 * ((1 - x[i]) ** N2)
        n = w.shape[0] - 1

        K = np.zeros(n + 1)
        for i in range(n + 1):
            K[i] = factorial(n) / (factorial(i) * factorial(n - i))

        S = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            S[i, 0] = 0
            for j in range(n + 1):
                term = w[j] * K[j] * x[i] ** j * ((1 - x[i]) ** (n - j))
                S[i, 0] += term
        y = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            y[i, 0] = C[i, 0] * S[i, 0] + x[i] * dz

        return y

    def CSTForFitBuild(self, x, w, flw, N1, N2):
        wu = w[:flw - 1]
        wl = w[flw - 1:-1]
        dz = w[-1] / 2

        zerind = np.argmin(x)
        xu = x[:zerind]
        xl = x[zerind:]

        yl = self.ClassShape(wl, xl, N1, N2, -dz)
        yu = self.ClassShape(wu, xu, N1, N2, dz)

        y = np.concatenate((yu, yl))

        return y.flatten()

    def getCST(self, xcoords, ycoords):
        wu_g = [0.1, 0.1, 0.1, 0.1]
        wl_g = [-0.1, -0.1, -0.1, -0.1]
        dz_g = 0
        p0 = wu_g + wl_g + [dz_g]

        flw = len(wu_g) + 1
        nTries = 5

        def Fun2Min(w):
            return np.abs(self.CSTForFitBuild(self.xcoords, w, flw, 0.5, 1) - ycoords)

        options = {
            'method': 'trf',
            'ftol': 1e-6,
            'max_nfev': 1200,
            'diff_step': 0.000001
        }

        results = []
        for _ in range(nTries):
            initial_guess = p0 + np.random.uniform(-0.1, 0.1, size=len(p0))
            result = least_squares(Fun2Min, initial_guess, **options)
            results.append(result)
        best_result = min(results, key=lambda res: res.cost)
        weights = best_result.x

        foil2 = self.CSTForFitBuild(self.xcoords, weights, flw, 0.5, 1)
        
        return foil2

    def foil(self, return_original_coords=True):
        """
        Fit CST parameterization to airfoil.
        
        Parameters:
        -----------
        return_original_coords : bool
            If True, return coordinates in original coordinate system.
            If False, return normalized coordinates.
            
        Returns:
        --------
        foil_coords : numpy array
            Fitted y-coordinates (normalized or original based on parameter)
        """
        foil_normalized = self.getCST(self.xcoords, self.ycoords)
        
        if return_original_coords:
            # Convert back to original coordinate system
            _, foil_original = self.normalizer.denormalize(self.xcoords, foil_normalized)
            return foil_original
        else:
            return foil_normalized