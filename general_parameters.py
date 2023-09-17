from numpy.random import uniform
import math
from scipy.stats import exponnorm
from scipy.optimize import minimize_scalar

class signal_globals:
    SAMPLE_RATE = 10 #Hz
    BASELINE_NOISE = 0.2
    RUN_LENGTH = 15 # minutes

class PeakCreator:
    DEFAULT_PEAK_WIDTH = 0.18
    SIGNAL_MULIPLIER = 5 #multipier for signal
    DEFAULT_BASE_ASYMMETRY = 1.03
    ASYMMETRY_DEPENDENCE_ON_RETENTION_TIME = 1.004
    WIDENING_CONSTANT = 1.01
    OVERALL_HEIGHT_RANDOM_NOISE = .007 #relative size of noise variation in peak height
    INDIVIDUAL_HEIGHT_RANDOM_NOISE = 0.002
    RETENTION_TIME_RANDOM_OFFSET_MAX = 0.003
    INDIVIDUAL_RETENTION_TIME_RANDOM_NOISE = 0.001

    def __init__(self):
        self.retention_time_offset = uniform(-self.RETENTION_TIME_RANDOM_OFFSET_MAX, self.RETENTION_TIME_RANDOM_OFFSET_MAX)
        self.height_modifier = uniform(1 - self.OVERALL_HEIGHT_RANDOM_NOISE, 1 + self.OVERALL_HEIGHT_RANDOM_NOISE)
    
    def peak(self, 
             retention_time: float,
             height: float,
             name: str = None, 
             base_width: float = DEFAULT_PEAK_WIDTH,
             base_asymmetry: float = DEFAULT_BASE_ASYMMETRY) -> dict :
        
        sigma = base_width * math.pow(self.WIDENING_CONSTANT, retention_time) / (2 * math.sqrt(2 * math.log(2)))
        
        asymmetry = 1./ (base_asymmetry * math.pow(self.ASYMMETRY_DEPENDENCE_ON_RETENTION_TIME, retention_time))
        
        mod_height = height * self.height_modifier * self.SIGNAL_MULIPLIER \
            * uniform(1 - self.INDIVIDUAL_HEIGHT_RANDOM_NOISE, 
                            1 + self.INDIVIDUAL_HEIGHT_RANDOM_NOISE)

        desired_mode = retention_time + self.retention_time_offset \
            + uniform(-self.INDIVIDUAL_RETENTION_TIME_RANDOM_NOISE, 
                            self.INDIVIDUAL_RETENTION_TIME_RANDOM_NOISE)

        def find_exponnorm_mode():
            exponnorm_dist = exponnorm(scale=sigma, K=asymmetry)

            # Find the mode
            # Define a function that returns the negative PDF (to maximize)
            def neg_pdf(x):
                return -exponnorm_dist.pdf(x)

            # Use optimization to find the mode
            result = minimize_scalar(neg_pdf)
            return result.x
        
        exponentially_modified_gaussian_mean = desired_mode - find_exponnorm_mode()
        
        return dict(
            name = name,
            time = exponentially_modified_gaussian_mean,
            height = mod_height,
            width = sigma,
            asymmetry = asymmetry
        )
