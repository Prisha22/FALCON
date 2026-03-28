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

        self.le_index = np.argmin(xcoords)
        x_le = xcoords[self.le_index]
        y_le = ycoords[self.le_index]

        x_translated = xcoords - x_le
        y_translated = ycoords - y_le
        self.translation = (x_le, y_le)

        x_te = (x_translated[0] + x_translated[-1]) / 2
        y_te = (y_translated[0] + y_translated[-1]) / 2

        self.rotation_angle = np.arctan2(y_te, x_te)
        cos_theta = np.cos(-self.rotation_angle)
        sin_theta = np.sin(-self.rotation_angle)

        x_rotated = x_translated * cos_theta - y_translated * sin_theta
        y_rotated = x_translated * sin_theta + y_translated * cos_theta

        chord_length = np.sqrt(x_te ** 2 + y_te ** 2)
        self.scale_factor = chord_length

        x_norm = x_rotated / chord_length
        y_norm = y_rotated / chord_length

        return x_norm, y_norm

    def denormalize(self, x_norm, y_norm):
        x_norm = np.array(x_norm)
        y_norm = np.array(y_norm)

        x_scaled = x_norm * self.scale_factor
        y_scaled = y_norm * self.scale_factor

        cos_theta = np.cos(self.rotation_angle)
        sin_theta = np.sin(self.rotation_angle)

        x_unrotated = x_scaled * cos_theta - y_scaled * sin_theta
        y_unrotated = x_scaled * sin_theta + y_scaled * cos_theta

        x_original = x_unrotated + self.translation[0]
        y_original = y_unrotated + self.translation[1]

        return x_original, y_original


class Parsec:
    def __init__(self, airfoil_path, airfoil_name):
        self.airfoil_path = airfoil_path
        self.airfoil_name = airfoil_name
        xcoords, ycoords = read_airfoil.read_airfoil_coordinates(airfoil_path, airfoil_name)

        self.xcoords_original = xcoords
        self.ycoords_original = ycoords

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

        x_maxy = np.clip(x_maxy, 0.01, 0.99)
        x_miny = np.clip(x_miny, 0.01, 0.99)

        te_t = ycoords[0] - ycoords[-1]
        y_te = (ycoords[0] + ycoords[-1]) / 2

        x_up = xcoords[ymax_index - 1:ymax_index + 2]
        y_up = ycoords[ymax_index - 1:ymax_index + 2]
        coeffs_up = np.polyfit(x_up, y_up, 2)

        x_low = xcoords[ymin_index - 1:ymin_index + 2]
        y_low = ycoords[ymin_index - 1:ymin_index + 2]
        coeffs_low = np.polyfit(x_low, y_low, 2)

        slopeUp = 2 * coeffs_up[0]
        slopeLow = 2 * coeffs_low[0]

        le_index = np.argmin(xcoords)
        n_le_points = min(5, le_index, len(xcoords) - le_index - 1)

        x_le_up = xcoords[le_index:le_index + n_le_points]
        y_le_up = ycoords[le_index:le_index + n_le_points]
        if len(x_le_up) >= 3:
            coeffs_le_up = np.polyfit(x_le_up, y_le_up, 2)
            rUp = abs(1 / (2 * coeffs_le_up[0])) if coeffs_le_up[0] != 0 else 0.01
        else:
            rUp = 0.01

        x_le_low = xcoords[le_index - n_le_points:le_index + 1]
        y_le_low = ycoords[le_index - n_le_points:le_index + 1]
        if len(x_le_low) >= 3:
            coeffs_le_low = np.polyfit(x_le_low, y_le_low, 2)
            rlow = abs(1 / (2 * coeffs_le_low[0])) if coeffs_le_low[0] != 0 else 0.01
        else:
            rlow = 0.01

        det_value = abs((xcoords[1] - xcoords[0]) * (ycoords[-2] - ycoords[-1]) -
                        (ycoords[1] - ycoords[0]) * (xcoords[-2] - xcoords[-1]))
        dot_product = ((xcoords[1] - xcoords[0]) * (xcoords[-2] - xcoords[-1]) +
                       (ycoords[1] - ycoords[0]) * (ycoords[-2] - ycoords[-1]))
        beta = math.atan2(det_value, dot_product) * 180 / math.pi

        xx = np.zeros(5)
        yy = np.zeros(5)

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
            p_clipped = p.copy()
            p_clipped[2] = np.clip(p_clipped[2], 0.01, 0.99)  # x_maxy
            p_clipped[5] = np.clip(p_clipped[5], 0.01, 0.99)  # x_miny
            p_clipped[0] = max(p_clipped[0], 1e-6)  # rUp
            p_clipped[1] = max(p_clipped[1], 1e-6)  # rlow

            result = self.parsec_for_fit_build(self.xcoords, p_clipped) - self.ycoords

            if not np.isfinite(result).all():
                return np.ones_like(self.ycoords) * 1e6
            return result
        except (ValueError, RuntimeWarning, np.linalg.LinAlgError):
            return np.ones_like(self.ycoords) * 1e6

    def parsec_for_fit_build(self, x, p):
        locc = np.argmin(x)
        xUp = x[:locc + 1]
        xLow = x[locc:]

        def Foil(x_vals, aa):
            x_safe = np.maximum(x_vals, 1e-12)
            return (aa[0] * x_safe ** (1 / 2) + aa[1] * x_safe ** (3 / 2) +
                    aa[2] * x_safe ** (5 / 2) + aa[3] * x_safe ** (7 / 2) +
                    aa[4] * x_safe ** (9 / 2) + aa[5] * x_safe ** (11 / 2))

        c1 = np.array([1, 1, 1, 1, 1, 1])
        c2 = np.array([p[2] ** (1 / 2), p[2] ** (3 / 2), p[2] ** (5 / 2),
                       p[2] ** (7 / 2), p[2] ** (9 / 2), p[2] ** (11 / 2)])
        c3 = np.array([1 / 2, 3 / 2, 5 / 2, 7 / 2, 9 / 2, 11 / 2])
        c4 = np.array([(1 / 2) * p[2] ** (-1 / 2), (3 / 2) * p[2] ** (1 / 2),
                       (5 / 2) * p[2] ** (3 / 2), (7 / 2) * p[2] ** (5 / 2),
                       (9 / 2) * p[2] ** (7 / 2), (11 / 2) * p[2] ** (9 / 2)])
        c5 = np.array([(-1 / 4) * p[2] ** (-3 / 2), (3 / 4) * p[2] ** (-1 / 2),
                       (15 / 4) * p[2] ** (1 / 2), (35 / 4) * p[2] ** (3 / 2),
                       (63 / 4) * p[2] ** (5 / 2), (99 / 4) * p[2] ** (7 / 2)])
        c6 = np.array([1, 0, 0, 0, 0, 0])

        Cup = np.vstack((c1, c2, c3, c4, c5, c6))
        angle_up_deg = -p[10] - p[11] / 2
        angle_up_rad = np.deg2rad(angle_up_deg)

        bup = np.array([
            p[9] + p[8] / 2,  # p(10) + p(9)/2
            p[3],  # p(4) = y_up
            np.tan(angle_up_rad),  # tand(-p(11)-p(12)/2)
            0,  # 0
            p[4],  # p(5) = y_xxup
            np.sqrt(2 * p[0])  # sqrt(2*p(1)) = sqrt(2*r_leup)
        ])

        try:
            aup = solve(Cup, bup)
            foilUp = np.real(Foil(xUp, aup))
        except np.linalg.LinAlgError:
            raise ValueError("Upper surface calculation failed")

        c7 = np.array([1, 1, 1, 1, 1, 1])
        c8 = np.array([p[5] ** (1 / 2), p[5] ** (3 / 2), p[5] ** (5 / 2),
                       p[5] ** (7 / 2), p[5] ** (9 / 2), p[5] ** (11 / 2)])
        c9 = np.array([1 / 2, 3 / 2, 5 / 2, 7 / 2, 9 / 2, 11 / 2])
        c10 = np.array([(1 / 2) * p[5] ** (-1 / 2), (3 / 2) * p[5] ** (1 / 2),
                        (5 / 2) * p[5] ** (3 / 2), (7 / 2) * p[5] ** (5 / 2),
                        (9 / 2) * p[5] ** (7 / 2), (11 / 2) * p[5] ** (9 / 2)])
        c11 = np.array([(-1 / 4) * p[5] ** (-3 / 2), (3 / 4) * p[5] ** (-1 / 2),
                        (15 / 4) * p[5] ** (1 / 2), (35 / 4) * p[5] ** (3 / 2),
                        (63 / 4) * p[5] ** (5 / 2), (99 / 4) * p[5] ** (7 / 2)])
        c12 = np.array([1, 0, 0, 0, 0, 0])

        Clo = np.vstack((c7, c8, c9, c10, c11, c12))
        angle_lo_deg = -p[10] + p[11] / 2
        angle_lo_rad = np.deg2rad(angle_lo_deg)

        blo = np.array([
            p[9] - p[8] / 2,  # p(10) - p(9)/2
            p[6],  # p(7) = y_low
            np.tan(angle_lo_rad),  # tand(-p(11)+p(12)/2)
            0,  # 0
            p[7],  # p(8) = y_xxlow
            -np.sqrt(2 * p[1])  # -sqrt(2*p(2)) = -sqrt(2*r_lelo)
        ])

        if not np.isfinite(Clo).all() or not np.isfinite(blo).all():
            raise ValueError("Lower surface matrices contain inf or NaN")

        try:
            alower = solve(Clo, blo)
            foilLow = np.real(Foil(xLow, alower))
        except np.linalg.LinAlgError:
            raise ValueError("Lower surface calculation failed")

        foil = np.concatenate((foilUp[:-1], foilLow))
        return foil

    def foil(self, return_original_coords=True):
        """
        Fit PARSEC parameterization to airfoil.
        """
        if self.p is None:
            self.analysis()

        # [rUp, rlow, x_maxy, ymax, slopeUp, x_miny, ymin, slopeLow, te_t, y_te, alpha, beta]
        lower_bounds = np.array([
            1e-6,  # rUp
            1e-6,  # rlow
            0.05,  # x_maxy
            -0.5,  # ymax
            -50,  # slopeUp
            0.05,  # x_miny
            -0.5,  # ymin
            -50,  # slopeLow
            -0.1,  # te_t
            -0.5,  # y_te
            -45,  # alpha
            -45  # beta
        ])

        upper_bounds = np.array([
            1.0,  # rUp
            1.0,  # rlow
            0.95,  # x_maxy
            0.5,  # ymax
            50,  # slopeUp
            0.95,  # x_miny
            0.5,  # ymin
            50,  # slopeLow
            0.1,  # te_t
            0.5,  # y_te
            45,  # alpha
            45  # beta
        ])

        p0 = np.clip(self.p, lower_bounds, upper_bounds)

        options = {
            'ftol': 1e-8,
            'xtol': 1e-8,
            'gtol': 1e-8,
            'max_nfev': 2000,
            'verbose': 0
        }

        try:
            result = least_squares(
                self.fun_to_min,
                p0,
                bounds=(lower_bounds, upper_bounds),
                method='trf',
                **options
            )

            if not result.success:
                print(f"Warning: Optimization did not fully converge: {result.message}")

            para = result.x
            para[2] = np.clip(para[2], 0.01, 0.99)
            para[5] = np.clip(para[5], 0.01, 0.99)
            para[0] = max(para[0], 1e-6)
            para[1] = max(para[1], 1e-6)

            foil_normalized = self.parsec_for_fit_build(self.xcoords, para)
            print(f"Optimization complete. Final cost: {result.cost:.6e}")

        except Exception as e:
            print(f"PARSEC fitting failed: {e}")
            print("Returning original coordinates")
            foil_normalized = self.ycoords

        if return_original_coords:
            _, foil_original = self.normalizer.denormalize(self.xcoords, foil_normalized)
            return foil_original
        else:
            return foil_normalized