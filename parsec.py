import math
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import numpy as np
import read_airfoil
from scipy.linalg import solve

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


class Parsec:
    def __init__(self, airfoil_path, airfoil_name):
        self.airfoil_path = airfoil_path 
        self.airfoil_name = airfoil_name
        xcoords, ycoords = read_airfoil.read_airfoil_coordinates(airfoil_path, airfoil_name)
        
        # Store original coordinates
        self.xcoords_original = xcoords
        self.ycoords_original = ycoords
        
        # Normalize coordinates
        self.normalizer = AirfoilNormalizer()
        self.xcoords, self.ycoords = self.normalizer.normalize(xcoords, ycoords)
        
        self.p = None

    def analysis(self):
        xcoords = self.xcoords
        ycoords = self.ycoords
        ymax = max(ycoords)
        ymin = min(ycoords)
        xmax = max(xcoords)
        xmin = min(xcoords)

        ymax_index = np.argmax(ycoords)
        ymin_index = np.argmin(ycoords)
        x_maxy = xcoords[ymax_index]
        x_miny = xcoords[ymin_index]
        
        # Ensure x positions are within valid range (0.01 to 0.99)
        x_maxy = np.clip(x_maxy, 0.01, 0.99)
        x_miny = np.clip(x_miny, 0.01, 0.99)
        
        te_t = ycoords[0] - ycoords[-1]
        y_te = (ycoords[0] + ycoords[-1]) / 2

        # Upper surface
        x_up = xcoords[ymax_index - 1:ymax_index + 2]
        y_up = ycoords[ymax_index - 1:ymax_index + 2]
        coeffs_up = np.polyfit(x_up, y_up, 2)
        rUp = 1 / (2 * coeffs_up[0]) if coeffs_up[0] != 0 else 1.0
        
        # Lower surface
        x_low = xcoords[ymin_index - 1:ymin_index + 2]
        y_low = ycoords[ymin_index - 1:ymin_index + 2]
        coeffs_low = np.polyfit(x_low, y_low, 2)
        rlow = 1 / (2 * coeffs_low[0]) if coeffs_low[0] != 0 else -1.0

        # Slope calculations
        slopeUp = 2 * coeffs_up[0]
        slopeLow = 2 * coeffs_low[0]

        # Angle calculations
        det_value = abs((xcoords[1] - xcoords[0]) * (ycoords[-2] - ycoords[-1]) - (ycoords[1] - ycoords[0]) * (xcoords[-2] - xcoords[-1]))
        dot_product = (xcoords[1] - xcoords[0]) * (xcoords[-2] - xcoords[-1]) + (ycoords[1] - ycoords[0]) * (ycoords[-2] - ycoords[-1])
        beta = math.atan2(det_value, dot_product) * 180 / math.pi

        le_index = np.argmin(xcoords)
        
        xx = np.zeros((5, 1))
        yy = np.zeros((5, 1))
        
        xx[0] = xcoords[le_index]
        yy[0] = ycoords[le_index]
        
        xx[1] = (xcoords[1] + xcoords[-2]) / 2
        xx[2] = (xcoords[0] + xcoords[-1]) / 2
        xx[3] = xx[1]
        yy[1] = (ycoords[1] + ycoords[-2]) / 2
        yy[2] = (ycoords[0] + ycoords[-1]) / 2
        yy[3] = yy[1]

        x1, y1 = xx[0], yy[0]
        x2, y2 = xx[1], yy[1]
        x3, y3 = xx[2], yy[2]

        det_value = abs((x1 - x2) * (y3 - y2) - (y1 - y2) * (x3 - x2))
        dot_product = (x1 - x2) * (x3 - x2) + (y1 - y2) * (y3 - y2)
        alpha = math.atan2(det_value, dot_product) * 180 / math.pi

        p = np.array([rUp, rlow, x_maxy, ymax, slopeUp, x_miny, ymin, slopeLow, te_t, y_te, alpha, beta])
        self.p = p
        print("Parsec parameters:", p)
        return p

    def fun_to_min(self, p):
        try:
            result = self.parsec_for_fit_build(self.xcoords, p) - self.ycoords
            # Return large error if result contains NaN or inf
            if not np.isfinite(result).all():
                return np.ones_like(self.ycoords) * 1e6
            return result
        except (ValueError, RuntimeWarning):
            # Return large error if computation fails
            return np.ones_like(self.ycoords) * 1e6

    def parsec_for_fit_build(self, x, p):
        # Validate parameters to prevent NaN/inf
        # p[2] = x_maxy (upper surface max location)
        # p[5] = x_miny (lower surface max location)
        p[2] = np.clip(p[2], 0.01, 0.99)  # Keep in valid range
        p[5] = np.clip(p[5], 0.01, 0.99)  # Keep in valid range
        
        locc = np.argmin(x)
        xUp = x[:locc]
        xLow = x[locc:]
        angle = -p[10] + p[11] / 2
        if not (-math.pi / 2 < angle < math.pi / 2):
            angle = np.mod(angle + np.pi / 2, np.pi) - np.pi / 2
            
        def Foil(x, aa):
            # Add small epsilon to avoid exact zero
            x_safe = np.maximum(x, 1e-10)
            return aa[0] * x_safe**(1/2) + aa[1] * x_safe**(3/2) + aa[2] * x_safe**(5/2) + aa[3] * x_safe**(7/2) + aa[4] * x_safe**(9/2) + aa[5] * x_safe**(11/2)
    
        # Upper surface calculations
        c1 = np.array([1, 1, 1, 1, 1, 1])
        c2 = np.array([p[2]**(1/2), p[2]**(3/2), p[2]**(5/2), p[2]**(7/2), p[2]**(9/2), p[2]**(11/2)])
        c3 = np.array([1/2, 3/2, 5/2, 7/2, 9/2, 11/2])
        c4 = np.array([(1/2) * p[2]**(-1/2), (3/2) * p[2]**(1/2), (5/2) * p[2]**(3/2),(7/2) * p[2]**(5/2), (9/2) * p[2]**(7/2), (11/2) * p[2]**(9/2)])
        c5 = np.array([(-1/4) * p[2]**(-3/2), (3/4) * p[2]**(-1/2), (15/4) * p[2]**(1/2), (35/4) * p[2]**(3/2), (63/4) * p[2]**(5/2), (99/4) * p[2]**(7/2)])
        c6 = np.array([1, 0, 0, 0, 0, 0])
    
        Cup = np.vstack((c1, c2, c3, c4, c5, c6))
        bup = np.array([p[9] +p[8]/2, p[3], np.tan(-p[10] - p[11] / 2), 0, p[4], math.sqrt(2 * abs(p[0]))])
        
        try:
            aup = solve(Cup, bup)
            foilUp = np.real(Foil(xUp, aup))
        except:
            raise ValueError("Upper surface calculation failed")
    
        # Lower surface calculations
        c7 = np.array([1, 1, 1, 1, 1, 1])
        c8 = np.array([p[5]**(1/2), p[5]**(3/2), p[5]**(5/2), p[5]**(7/2), p[5]**(9/2), p[5]**(11/2)])
        c9 = np.array([1/2, 3/2, 5/2, 7/2, 9/2, 11/2])
        c10 = np.array([(1/2) * p[5]**(-1/2), (3/2) * p[5]**(1/2), (5/2) * p[5]**(3/2), (7/2) * p[5]**(5/2), (9/2) * p[5]**(7/2), (11/2) * p[5]**(9/2)])
        c11 = np.array([(-1/4) * p[5]**(-3/2), (3/4) * p[5]**(-1/2), (15/4) * p[5]**(1/2),(35/4) * p[5]**(3/2), (63/4) * p[5]**(5/2), (99/4) * p[5]**(7/2)])
        c12 = np.array([1, 0, 0, 0, 0, 0])
    
        Clo = np.vstack((c7, c8, c9, c10, c11, c12))
        angle = -p[10] + p[11] / 2
        blo = np.array([p[9] - p[8] / 2,p[6],np.tan(angle), 0, p[7],-math.sqrt(2 * abs(p[1]))])
        
        if not np.isfinite(Clo).all():
            raise ValueError("Clo contains inf or NaN values")
        if not np.isfinite(blo).all():
            raise ValueError("blo contains inf or NaN values")
        
        try:
            alower = solve(Clo, blo)
            foilLow = np.real(Foil(xLow, alower))
        except:
            raise ValueError("Lower surface calculation failed")
       
        foil = np.concatenate((foilUp, foilLow))
        return foil

    def foil(self, return_original_coords=True):
        """
        Fit PARSEC parameterization to airfoil.
        
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
        if self.p is None:
            self.analysis()
        
        # Set bounds for parameters to keep them physically valid
        # [rUp, rlow, x_maxy, ymax, slopeUp, x_miny, ymin, slopeLow, te_t, y_te, alpha, beta]
        lower_bounds = [-np.inf, -np.inf, 0.01, -np.inf, -np.inf, 0.01, -np.inf, -np.inf, -np.inf, -np.inf, -180, -180]
        upper_bounds = [np.inf, np.inf, 0.99, np.inf, np.inf, 0.99, np.inf, np.inf, np.inf, np.inf, 180, 180]
            
        options = {
            'method': 'trf',
            'ftol': 1e-6,
            'max_nfev': 1200,
            'x_scale': [0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 2, 2],
            'diff_step': 0,
        }
        
        try:
            result = least_squares(self.fun_to_min, self.p, bounds=(lower_bounds, upper_bounds), **options)
            para = result.x
            foil_normalized = self.parsec_for_fit_build(self.xcoords, para)
        except Exception as e:
            print(f"PARSEC fitting failed: {e}")
            print("Returning best available fit or original coordinates")
            # Return original y-coordinates if fitting completely fails
            foil_normalized = self.ycoords
        
        if return_original_coords:
            _, foil_original = self.normalizer.denormalize(self.xcoords, foil_normalized)
            return foil_original
        else:
            return foil_normalized