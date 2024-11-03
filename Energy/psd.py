import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.cluster import KMeans

def make_subplot_layout(signal_data, res, cluster, rows=5, cols=5):
    indices = np.where(res == cluster)[0]
    cluster_signal = signal_data[indices][:rows * cols]
    
    plt.figure(figsize=(12, 8))
    for i, sig in enumerate(cluster_signal):
        plt.subplot(rows, cols, i + 1)
        plt.plot(sig)
        plt.axis('off')  # Optional: turn off axes for cleaner look
    plt.tight_layout()
    plt.show()

def psd_score(ecg, tag=0):
    paral_results = parallelizing_sig(ecg)  # Assuming this is defined elsewhere
    
    half_index = (
        sum(abs(paral_results[:len(paral_results) // 2])) + 1e-10
    ) / (sum(abs(paral_results[len(paral_results) // 2:])) + 1e-10)
    
    freqs, psd = signal.welch(paral_results)

    if tag == 1:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(ecg)
        plt.title('Original ECG')
        
        plt.subplot(1, 3, 2)
        plt.plot(paral_results)
        plt.title('Processed Signal')
        
        plt.subplot(1, 3, 3)
        plt.plot(freqs, psd)
        plt.title('Power Spectral Density')
        
        plt.tight_layout()
        plt.show()

    power_index = np.sum(psd)
    wave_form_index = (np.sum(psd[25:50]) + 1e-10) / (np.sum(psd[:25]) + 1e-10)
    ratio_index = power_index * wave_form_index
    
    return power_index, wave_form_index, ratio_index, half_index

# Clustering
kmeans = KMeans(n_clusters=4)
res = kmeans.fit_predict(feature)  # Assuming 'feature' is defined
