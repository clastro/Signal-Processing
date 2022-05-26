import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import scipy.signal
from scipy.signal import butter, iirnotch, lfilter

ecg_wave = np.load('test.npy') # 10초 ECG (sampling rate=500)
#ecg_wave.shape -> (5000,)

def bandpass(lowcut, highcut, order=5):
    nyq = 0.5 * 500
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

b , a = bandpass(0.5, 45)
filtered_ecg = signal.filtfilt(b,a,ecg_wave[250:4750])

_, rpeaks = nk.ecg_peaks(filtered_ecg, sampling_rate=500)

_, waves_cwt = nk.ecg_delineate(filtered_ecg, rpeaks, sampling_rate=500, method="dwt", show=False, show_type='peaks')

# 관심 ECG 설정

interest_ecg = filtered_ecg[rpeaks['ECG_R_Peaks'][0]:rpeaks['ECG_R_Peaks'][1]]

# template ECG 설정

template_ecg =[]
for i in range(len(rpeaks['ECG_R_Peaks'])):
    try:
        template_ecg.append(filtered_ecg[rpeaks['ECG_R_Peaks'][i+1]:rpeaks['ECG_R_Peaks'][i+2]])
    except:
        continue
        
# Template ECG의 가장 짧은 길이로 맞추기

for i in range(len(template_ecg)):
    if(i ==0):
        min_value = template_ecg[i].shape[0]
    else:
        min_value = min(min_value,template_ecg[i].shape[0])
        
# Interest ECG 길이 맞추기

grad_interest_ecg = np.gradient(interest_ecg)
idx = (-abs(grad_interest_ecg)).argsort()[:min_value]
idx.sort()
new_interest_ecg = interest_ecg[idx]

# Template ECG 전체 길이 맞추기

corr_score = []
new_template_ecg = {}
for i in range(len(template_ecg)):
    grad_template_ecg = np.gradient(template_ecg[i])
    idx = (-abs(grad_template_ecg)).argsort()[:min_value]
    idx.sort()
    new_template_ecg[i] = template_ecg[i][idx]
    corr_score.append(np.corrcoef(new_interest_ecg,new_template_ecg[i])[1][0])
    
template_number = np.argpartition(corr_score, len(corr_score) // 2)[len(corr_score) // 2] # correlation 중간값을 Template으로

f_ecg = new_interest_ecg - new_template_ecg[template_number]

b_f , a_f = bandpass(4, 9)
filtered_interest_ecg = signal.filtfilt(b_f,a_f,new_interest_ecg)
filtered_f_ecg = signal.filtfilt(b_f,a_f,f_ecg)

ffi, Pfi = scipy.signal.periodogram(filtered_interest_ecg, 500, nfft=2**12)
ff, Pf = scipy.signal.periodogram(filtered_f_ecg, 500, nfft=2**12)

interest_power = np.dot(ffi,Pfi)
f_wave_power = np.dot(ff,Pf)
f_wave_score = f_wave_power/interest_power
        

  
