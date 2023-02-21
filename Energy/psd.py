from scipy import signal
import seaborn as sns
from sklearn.cluster import KMeans

def make_subplot_layout(signal,res,cluster,row=5,col=5):
    _index = np.where(res == cluster)
    cluster_signal = signal[_index[0][:row*col]]
    plt.figure(figsize=(12,8))
    for i in range(len(cluster_signal)) : 
        plt.subplot(row,col,i+1) 
        plt.plot(cluster_signal[i])
    plt.show()

def psd_score(ecg,tag = 0):
    paral_results = parallelizing_sig(ecg)
    #partial_norm_result = partial_normalizing_sig(paral_results)
    
    half_index = (sum(abs(paral_results[0:int(len(paral_results)/2)]))+ 1e-10)/(sum(abs(paral_results[int(len(paral_results)/2):]))+ 1e-10)
    
    freqs, psd = signal.welch(paral_results)
    if(tag == 1):
        plt.plot(ecg)
        plt.show()
        plt.plot(paral_results)
        plt.show()
        plt.plot(psd)
        plt.show()
    power_index = sum(psd)
    wave_form_index = (sum(psd[25:50])+ 1e-10)/(sum(psd[0:25])+ 1e-10)
    ratio_index = power_index * wave_form_index
    
    return power_index,wave_form_index,ratio_index,half_index

Kmean = KMeans(n_clusters=4)
res = Kmean.fit_predict(feature)
