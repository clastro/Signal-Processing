def make_zscore(ecg):
    return np.abs(stats.zscore(ecg))
  
def remove_outlier(ecg):
    z_threshold = 10
    z = make_zscore(ecg)
    outlier_index = np.where(z > z_threshold)
    return np.delete(ecg, outlier_index)
  
  
