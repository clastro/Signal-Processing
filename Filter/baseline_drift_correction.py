# using median filter

from scipy.signal import medfilt

window_size = 133 # Size of the moving window
ecg_filtered = medfilt(ecg, window_size)
corrected_ecg = ecg - ecg_filtered
