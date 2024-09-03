from numpy import array, sqrt, dot, exp
from scipy.stats import exponnorm, norm
from scipy.optimize import minimize, root


class PeakDefinitionError(ValueError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class PeakWidth:
    """
    Contains peak width parameters and values
    """

    def __init__(self, start, end, middle, height, method="height", tangents=False):
        self.start = float(start)
        self.end = float(end)
        self.middle = float(middle)
        self.width = float(end - start)
        self.height = float(height)
        self.left = float(middle - start)
        self.right = float(end - middle)
        self.type = method
        self.tangents = tangents


class PeakTangent:
    """
    Contains peak tangent parameters and values
    """

    def __init__(self, poi_x, poi_y, slope):
        self.poi_x = float(poi_x)
        self.poi_y = float(poi_y)
        self.slope = float(slope)

    def get_point_on_tangent(self, x=None, y=None):
        """
        returns a point on the tangent line calculated using point-slope formula
        """

        if x is not None:
            return self.slope * (x - self.poi_x) + self.poi_y
        elif y is not None:
            return (y - self.poi_y) / self.slope + self.poi_x


class Peak:
    """
    Defines peak using an exponentially modified gaussian distribution and calculates parameters for it
    """

    def __init__(self, rt=1.0, k=1, sigma=0.08, height=1200) -> None:
        """
        store user parameters and start calculating things
        """

        self.sigma = sigma
        self.loc = rt
        self.rt = rt
        self.height = height
        self.k = k
        if k == 0:
            self.distribution = norm
            self.reverse = False
            self.params = {"loc": self.loc, "scale": sigma}
        elif k > 0:
            self.distribution = exponnorm
            self.reverse = False
            self.params = {"K": k, "loc": self.loc, "scale": sigma}
        else:
            self.distribution = exponnorm
            self.reverse = True
            self.params = {"K": -k, "loc": -self.loc, "scale": sigma}

        # points of inflections in normal distributions are +/- 1 SD
        self.left_guess = self.rt - self.sigma
        self.right_guess = self.rt + self.sigma
        self.set_peak_mode()
        self.find_peak_tangents(self.left_guess, self.right_guess)
        self.widths = self.Widths(self)
        self.statistics = self.Statistic(self)

    def set_peak_mode(self):
        self.scale = 1

        def negative_pdf(x):
            return -self.get_signal(x)

        val = minimize(negative_pdf, x0=self.loc)
        true_mode = val.x
        self.pdf_maximum = -val.fun
        self.scale = 1 / self.pdf_maximum * self.height
        self.loc = 2 * self.loc - true_mode
        self.params["loc"] = self.loc

    def get_signal(self, times):
        if not self.reverse:
            return self.scale * self.distribution.pdf(times, **self.params)
        else:
            return self.scale * self.distribution.pdf(
                times,
                K=self.params["K"],
                loc=-self.params["loc"],
                scale=self.sigma,
            )

    def derivative(self, x, dx=1e-5, n=1, order=3):
        """
        approximates peak first derivative using finite differences
        """
        # pre-computed for n=1 and 2 and low-order for speed.
        if n == 1:
            if order == 3:
                weights = array([-1, 0, 1]) / 2.0
            elif order == 5:
                weights = array([1, -8, 0, 8, -1]) / 12.0
        elif n == 2:
            if order == 3:
                weights = array([1, -2.0, 1])
            elif order == 5:
                weights = array([-1, 16, -30, 16, -1]) / 12.0
        val = 0.0
        ho = order >> 1
        k = x + (array(range(order)) - ho) * dx
        val = dot(weights, self.get_signal(k))
        val /= dx**n
        return val

    def get_width(self, value, method="height_percent", tangents=False):
        """
        method: select from "sigma", "baseline", or "height_percent". If method is not recognized, height percent is used by default.
        """

        def shifted_pdf(x, shift):
            """
            for finding zeros of shifted values
            """
            return self.get_signal(x) - shift

        if "sigma" in method:
            z = value / 2
            mod_height = exp(-(z**2) / 2) * self.height
        else:
            mod_height = value / 100 * self.height

        if tangents:
            left = self.left_tangent.get_point_on_tangent(y=mod_height)
            right = self.right_tangent.get_point_on_tangent(y=mod_height)
        else:
            left = root(shifted_pdf, self.left_guess, mod_height).x
            right = root(shifted_pdf, self.right_guess, mod_height).x
        return PeakWidth(
            left, right, self.rt, mod_height, method=method, tangents=tangents
        )

    def find_peak_tangents(self, left_guess, right_guess):
        left_poi_x = root(self.derivative, left_guess, args=(1e-5, 2)).x
        right_poi_x = root(self.derivative, right_guess, args=(1e-5, 2)).x
        left_poi_y, right_poi_y = self.get_signal(array([left_poi_x, right_poi_x]))
        left_slope = self.derivative(left_poi_x, n=1)
        right_slope = self.derivative(right_poi_x, n=1)
        self.left_tangent = PeakTangent(left_poi_x, left_poi_y, left_slope)
        self.right_tangent = PeakTangent(right_poi_x, right_poi_y, right_slope)

    class Widths:
        def __init__(self, peak) -> None:
            self.baseline_tangent = peak.get_width(
                0, method="height_percent", tangents=True
            )
            self.sigma_4 = peak.get_width(4, method="sigma")
            self.sigma_5 = peak.get_width(5, method="sigma")
            self.height_5 = peak.get_width(5, method="height_percent")
            self.height_10 = peak.get_width(10, method="height_percent")
            self.height_50 = peak.get_width(50, method="height_percent")

    class Statistic:
        def __init__(self, peak) -> None:
            self.mean, self.var, self.skew, self.kurt = peak.distribution.stats(
                **peak.params, moments="mvsk"
            )
            self.stddev = sqrt(self.var)
            self.asymmetry_USP = self.asymmetry_EP = (
                peak.widths.height_5.right + peak.widths.height_5.left
            ) / (2 * peak.widths.height_5.left)
            self.asymmetry_AIA = (
                peak.widths.height_10.right / peak.widths.height_10.left
            )
            self.asymmetry_stat_moment = (self.mean - peak.rt) / self.stddev
