import math

# Method Parameters
SAMPLE_RATE = 5  # in Hz; default if not specified in method
RUN_LENGTH = 15  # in min; default if not specified in method
FLOW_RATE = 5  # in ml/min; default if not specified in method
SOLVENT_PROFILE_CONVOLUTION_WIDTH = (
    1  # smooting of solvent profile from nominal, in units of min
)

# Chromatogram creation filterss
## Peak shape parameters
DEFAULT_PEAK_WIDTH = (
    0.06  # default peak width for peak eluting at t = 0, in units of min
)
SIGNAL_MULTIPLIER = 1  # multipier for signal
DEFAULT_BASE_ASYMMETRY = 1.03  # default peak asymmetry
ASYMMETRY_DEPENDENCE_ON_RETENTION_TIME = (
    1.003  # how much peaks become more asymmetric over time, in units of min^-1
)
WIDENING_CONSTANT = 1.02  # how much peaks widen over time, in units of min^-1
LINEAR_LIMIT = (
    2000  # mAU after which peak height is no longer linear with concentration
)
SATURATION_SCALE = (
    LINEAR_LIMIT / 5
)  # range over LINEAR_LIMIT in which saturation occurs

## Peak Noise Parameters
OVERALL_HEIGHT_RANDOM_NOISE = (
    0.01  # relative size of noise variation in peak height between injections
)
INDIVIDUAL_HEIGHT_RANDOM_NOISE = (
    0.005  # fluctuation of individual height from ideal within injections
)
RETENTION_TIME_RANDOM_OFFSET_MAX = (
    0.01  # fluctuation of retention time between injections
)
INDIVIDUAL_RETENTION_TIME_RANDOM_NOISE = (
    0.002  # fluctuation of retention time within injections
)

## Background Noise Parameters
BACKGROUND_MEAN = 0  #
BASELINE_NOISE = 0.2 * math.sqrt(SAMPLE_RATE)  # Standard deviation of background noise
BASELINE_AUTOCORRELATION_PARAMETER = math.sqrt(1 / SAMPLE_RATE)
BASELINE_MULTIPLIER = 2  # multiplier for amplitude of background

# Sample creation parameters
## Peak Naming Parameters
ID_PEAK_PREFIX = "JSK"
IMP_PEAK_PREFIX = "IMP"
RANDOM_PEAK_ID_DIGITS = 4  # number of digits to put into to a random peak id

# Analysis Parameters
BACKGROUND_NOISE_RANGE = [0, 100]  # indices of background noise signal
NOISE_THRESHOLD_MULTIPLIER = 10  # sets multiplier to be able to find peaks. Higher values result in fewer peaks being found.
PEAK_LIMIT = 3  # sets value to be able to find peaks. Higher values result in fewer peaks being found; lower values may produce false positive peaks.
MINIMUM_HEIGHT = 1.5  # Minimum height for a peak to be kept in peak finding algorithm
MINIMUM_HEIGHT_METHOD = (
    "baseline_noise_multiplier"  # How minimum height is to be calculated.
)
MINIMUM_AREA = 0.05  # Minimum area for a peak to be kept in peak finding algorithm
