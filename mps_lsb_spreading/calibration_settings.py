# ----------------------------------------------------------
# Calibration settings for SuperMUSR Calibration Scripts
# ----------------------------------------------------------


#COMMON SETTINGS FOR CALIBRATION SCRIPTS

# IP address of the digitizer
ip = "130.246.55.111"

# only channels in ENABLED_CH will be calibrated,
# pay attention is not possible to calibrate only one side of the channel
ENABLED_CH=[0,1,2,3,4,5,6,7]

# Protostave should be in section 0
# Full-stave it is important to set the correct section (0,1,2 or 3)
# ----------------------------------------------------------
# | A_0 | A_1 |     B_2      |           C_3               |
# ----------------------------------------------------------

SECTION = 0

# Channel mapping 
# -------------------------------------------------------
# | DIGITIZER  |  LEFT  | RIGHT  |  HV LEFT  | HV RIGHT |
# -------------------------------------------------------
# |    0       |  0     |    4   |    0 *    |    4     |
# |    1       |  1     |    5   |    1      |    5     |
# |    2       |  2     |    6   |    2      |    6     |
# |    3       |  3     |    7   |    3      |    7     |
# |    4       |  8     |   12   |    8      |   12     |
# |    5       |  9     |   13   |    9      |   13     |
# |    6       |  10    |   14   |    10     |   14     |
# |    7       |  11    |   15   |    11     |   15     |
# ------------------------------------------------------
# this number is plus 16 for section 1, plus 32 for section 2 and plus 48 for section 3

# Preamp mapping
# ------------------------------
# | SECTION  |  LEFT  | RIGHT  |
# ------------------------------
# |    0     |   1    |   0    |
# |    1     |   3    |   2    |
# |    2     |   5    |   4    |
# |    3     |   7    |   6    |
# ------------------------------
MAP_LEFT = [0, 1, 2, 3, 8, 9, 10, 11]
MAP_RIGHT = [4, 5, 6, 7, 12, 13, 14, 15]


# Target voltage used in PULSER scan. This voltage is already corrected
# tp the module type (A,B or C) and this is the real (approximated) voltage
# that will be applied to the SiPMs during the PULSER scan
VTARGET = 56

# Module type: A, B or C! DANGER SET THE CORRECT ONE OR YOU CAN DAMAGE THE MODULE! 
MODULE = 'A'  # 'A', 'B' or 'C'

# Estimation of the single photon PE width LSB distance @ VTARGET
# This is important for starting point for the PULSER scan and the HV scan

SINGLE_PHOTON_PEAK = 30



# ===================================================================
# OFFSET CALIBRATION SETTINGS

# Offset we want to get on the wavefrorm after the offset calibration
# This is the baseline of the ADC you will get on digitized waveforms
TARGET_OFFSET = 120
# minimum DC voltage applied during the offset scan 
OFFSET_CAL_START_OFFSET = 0.6
# maximum DC voltage applied during the offset scan
OFFSET_CAL_END_OFFSET = 0.9
# Step in V of the offset scan in the coarse scan
OFFSET_CAL_RAW_STEP = 0.025
# Minimum number of points to consider a solution valid
OFFSET_CAL_FINE_STEP = 0.004
# Number of iterations for the fine scan
OFFSET_CAL_N_FINE_STEP = 5

# ====================================================================
# PULSER CALIBRATION SETTINGS
# Minimum voltage of the pulser in the scan
PULSER_MIN_VOLTAGE = 1750
# Maximum voltage of the pulser in the scan - DO NOT EXCEED 2400 mV
PULSER_MAX_VOLTAGE = 2060
# Step in mV of the pulser scan in the coarse scan
PULSER_STEP_RAW = 20
# Minimum number of peaks to consider a solution valid
PULSER_MIN_PEAKS_COUNT = 5
PULSER_MAX_PEAKS_COUNT = 15
# Number of iterations for the fine scan
PULSER_FINE_SCAN=10
# Step in mV of the pulser scan in the fine scan
PULSER_FINE_STEP=5
# Weights for the weighted score
PULSER_PPFACTOR_WEIGHT = 0.80      # Peak to valley factor abs(max peak - next valley) / max peak
PULSER_PEAK_WEIGHT = 0.20
# Noise floor calibration parameters
PULSER_NOISE_FLOOR_INTEGRATION_TIME = 10  # seconds

# ====================================================================
# HV SCAN SETTINGS  

# How long integrate spectrum for each HV step (seconds)
HV_SCAN_TIME_STEP = 10  

# STEPS in HV scan (Volts will be scanned)
V1 = [57, 58,  59, 56,  55]

# Target value for the single photon peak position
# in ADC LSB/PE
PE_TARGET = 20 

# ====================================================================
# VBR SETTINGS
# VBR scan parameters
VBR_HV_MIN = 51.0  # Minimum HV to scan
VBR_HV_MAX = 58.0  # Maximum HV to scan
VBR_HV_STEP = 0.1  # HV step
VBR_N_WAVEFORMS = 5  # Number of waveforms to acquire per HV point
# trim this parameter if there is too much or too few light
VBR_PULSER_VOLTAGE = 1480  # mV




# ====================================================================
# !!!!!!   DO NOT TOUCH THIS PART OF CODE FOR SAFERY REASONS   !!!!!!
# ====================================================================
# * Se lo fai ti taglio l'uccello!!!!

if MODULE == 'A':
    V_cQ = 0
    MAIN_HV=59.9
else:
    V_cQ = 2.5 
    MAIN_HV=62.0

V_cM = 1