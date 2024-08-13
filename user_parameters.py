# Method Parameters
SAMPLE_RATE = 10  # in Hz; default if not specified in method
RUN_LENGTH = 15  # in minutes; default if not specified in method
FLOW_RATE = 5  # in ml/minutes; default if not specified in method

# Peak shape parameters
# DEFAULT_PEAK_WIDTH = 0.18
DEFAULT_PEAK_WIDTH = 0.08
SIGNAL_MULIPLIER = 1  # multipier for signal
DEFAULT_BASE_ASYMMETRY = 1.03
ASYMMETRY_DEPENDENCE_ON_RETENTION_TIME = 1.004
WIDENING_CONSTANT = 1.03
LINEAR_LIMIT = (
    2500  # mAU after which peak height is no longer linear with concentration
)
SATURATION_SCALE = LINEAR_LIMIT / 10  # range over which saturation occurs

# Peak Noise Parameters
OVERALL_HEIGHT_RANDOM_NOISE = 0.007  # relative size of noise variation in peak height
INDIVIDUAL_HEIGHT_RANDOM_NOISE = 0.002
RETENTION_TIME_RANDOM_OFFSET_MAX = 0.003
INDIVIDUAL_RETENTION_TIME_RANDOM_NOISE = 0.001

# Background Noise Parameters
BACKGROUND_MEAN = 0
BASELINE_NOISE = 1
BASELINE_AUTOCORRELATION_PARAMETER = 0.9
# BACKGROUND_NOISE = None
# BACKGROUND_SMOOTHING = 21  # points for smoothing

# Method Paramters
SOLVENT_PROFILE_CONVOLUTION_WIDTH = 1

# Peak Naming Parameters
ID_PEAK_PREFIX = "JSK"
IMP_PEAK_PREFIX = "IMP"
RANDOM_PEAK_ID_DIGITS = 4  # number of digits to put into to a random peak id
