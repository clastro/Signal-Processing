#!/usr/bin/env python
# coding: utf-8

# In[113]:


import pyedflib
import glob
import pandas as pd
import scipy.ndimage
import scipy.signal
import warnings
import numpy as np
from math import floor, ceil
import math
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.fft import fft, ifft
import pywt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[174]:


def _integrate(arr, step):
    integral = np.cumsum(arr)
    integral *= step
    return integral


def scale2frequency(wavelet, scale, precision=8):
    _,psi,x = wavelet.wavefun(10)    
    #psi, x = functions_approximations[1], functions_approximations[-1]
    domain = float(x[-1] - x[0])
    index = np.argmax(abs(fft(psi)[1:])) + 2
    if index > len(psi) / 2:
        index = len(psi) - index + 2

    return (1.0 / (domain / (index - 1))) / scale

def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def detect(signal, cwt_window=256*60*10):
    result_dump = []
    window_num = ceil(len(signal)/cwt_window)
    for idx in range(window_num):
        sliced_signal = signal[idx*cwt_window:(idx+1)*cwt_window]

        out, frequencies = cwt(sliced_signal, np.array([12], dtype=np.int32))
        # max_scale_factor = np.argmax(out.max(axis=-1)) + 1

        # out, frequencies = cwt(sliced_signal, np.array([max_scale_factor], dtype=np.int32))
        result_dump += list(out.reshape(-1))
    return np.array(result_dump, dtype=np.float32)


# In[361]:


def signal_smooth(signal, kernel="boxzen", size=10, alpha=0.1):
    if isinstance(signal, pd.Series):
        signal = signal.values
    length = len(signal)
    if isinstance(kernel, str) is False:
        raise TypeError("NeuroKit error: signal_smooth(): 'kernel' should be a string.")
    # Check length.
    size = int(size)
    if size > length or size < 1:
        raise TypeError("NeuroKit error: signal_smooth(): 'size' should be between 1 and length of the signal.")

    if kernel == "boxcar":
        # This is faster than using np.convolve (like is done in _signal_smoothing)
        # because of optimizations made possible by the uniform boxcar kernel shape.
        smoothed = scipy.ndimage.uniform_filter1d(signal, size, mode="nearest")

    elif kernel == "boxzen":
        # hybrid method
        # 1st pass - boxcar kernel
        x = scipy.ndimage.uniform_filter1d(signal, size, mode="nearest")

        # 2nd pass - parzen kernel
        smoothed = _signal_smoothing(x, kernel="parzen", size=size)

    elif kernel == "median":
        smoothed = _signal_smoothing_median(signal, size)

    else:
        smoothed = _signal_smoothing(signal, kernel=kernel, size=size)

    return smoothed

def _signal_smoothing_median(signal, size=5):

    # Enforce odd kernel size.
    if size % 2 == 0:
        size += 1

    smoothed = scipy.signal.medfilt(signal, kernel_size=size)
    return smoothed


def _signal_smoothing(signal, kernel, size=5):

    # Get window.
    window = scipy.signal.get_window(kernel, size)
    w = window / window.sum()

    # Extend signal edges to avoid boundary effects.
    x = np.concatenate((signal[0] * np.ones(size), signal, signal[-1] * np.ones(size)))

    # Compute moving average.
    smoothed = np.convolve(w, x, mode="same")
    smoothed = smoothed[size:-size]
    return smoothed

def bandpass_filter(src, sample_freq, high_cut, low_cut):
    if(high_cut < 0 or high_cut > 0.5 * sample_freq):
        high_cut = 0.5 * sample_freq - 0.1
    if(low_cut < 0):
        low_cut = 0.1

    nyq = 0.5 * sample_freq
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(1, [low, high], btype='band')

    y = lfilter(b, a, src)

    return y


# In[362]:


def signal_amp(ecg_signal):
    grad = np.gradient(ecg_signal) * 10 #기울기 x 10 : R만 1보다 크게 키우기 위함
    return abs(grad) ** 3 #3제곱


# In[363]:


def r_peak_detect(signal, sampling_rate=256, smoothwindow=0.1,
                    avgwindow=0.5,
                    gradthreshweight=1.5,
                    minlenweight=0.4,
                    maxlenweight=20,
                    mindelay=0.3,
                    if_show=False, start_time=None, dur_time=None):
    
    if if_show: #Graph show switch
        fig = plt.figure(figsize=(20,3*4))
        ax_1 = fig.add_subplot(411)
        ax_2 = fig.add_subplot(412)
        ax_3 = fig.add_subplot(413)
        ax_4 = fig.add_subplot(414)
        
        start_idx = int(sampling_rate*start_time)
        end_idx = int(sampling_rate*start_time+sampling_rate*dur_time)
    
    grad = np.gradient(signal)
    absgrad = np.abs(grad)
    smooth_kernel = int(np.rint(smoothwindow * sampling_rate))
    avg_kernel = int(np.rint(avgwindow * sampling_rate))
    #smoothgrad = moving_average(absgrad, n=5)
    smoothgrad = signal_smooth(absgrad, kernel="boxzen", size=smooth_kernel)
    avggrad = signal_smooth(smoothgrad, kernel="triang", size=avg_kernel)
    gradthreshold = gradthreshweight * avggrad
    mindelay = int(np.rint(sampling_rate * mindelay))
    
    if if_show:
        ax_1.plot(signal[start_idx:end_idx])
        ax_1.set_title("Original Signal")
        
        ax_2.plot(smoothgrad[start_idx:end_idx], color="b")
        ax_2.plot(avggrad[start_idx:end_idx], color="orange")
        ax_2.plot(gradthreshold[start_idx:end_idx], color="r")
        ax_2.set_title("SmoothGrad & GradThreshold")
        
            
    qrs = (smoothgrad > gradthreshold) & (smoothgrad > 0.01) # smooth의 기울기가 어느 정도 있어야 qrs로 인정
    beg_qrs = np.where(np.logical_and(np.logical_not(qrs[0:-1]), qrs[1:]))[0]
    end_qrs = np.where(np.logical_and(qrs[0:-1], np.logical_not(qrs[1:])))[0]
    
    end_qrs = end_qrs[end_qrs > beg_qrs[0]]
    num_qrs = min(beg_qrs.size, end_qrs.size)
    min_len = np.mean(end_qrs[:num_qrs] - beg_qrs[:num_qrs]) * minlenweight
    peaks = [0]
    
    if if_show:
        tmp_beg_qrs_idx = beg_qrs[np.logical_and(beg_qrs>=start_idx, beg_qrs<end_idx)] - start_idx
        tmp_end_qrs_idx = end_qrs[np.logical_and(end_qrs>=start_idx, end_qrs<end_idx)] - start_idx
        
        ax_3.plot(signal[start_idx:end_idx])
        ax_3.scatter(tmp_beg_qrs_idx, signal[start_idx:end_idx][tmp_beg_qrs_idx], color="r")
        ax_3.scatter(tmp_end_qrs_idx, signal[start_idx:end_idx][tmp_end_qrs_idx], color="g")
        ax_3.set_title("QRS begin & End")
    
    for i in range(num_qrs):

        beg = beg_qrs[i]
        end = end_qrs[i]
        len_qrs = end - beg

        if len_qrs < min_len:
            continue

        # Find local maxima and their prominence within QRS.
        data  = signal_amp(signal[beg:end])
        locmax, props = scipy.signal.find_peaks(data, prominence=(None, None))

        if locmax.size > 0:
            # Identify most prominent local maximum.
            peak = beg + locmax[np.argmax(props["prominences"])]
            # Enforce minimum delay between peaks.
            if abs(signal[peak]) > maxlenweight: #maximum value cut!
                continue
            #elif peak < np.mean(signal[:peak])*0.7:
            #    continue
            if peak - peaks[-1] > mindelay:
                peaks.append(peak)
    
    peaks.pop(0)
    
    peaks = np.asarray(peaks).astype(int)  # Convert to int
    
    if if_show:
        tmp_peask_idx = peaks[np.logical_and(peaks>=start_idx, peaks<end_idx)]-start_idx + 3 #기울기가 끝나는 시점 3
        ax_4.plot(signal[start_idx:end_idx])
        ax_4.scatter(tmp_peask_idx, signal[start_idx:end_idx][tmp_peask_idx], color="r")
        ax_4.set_title("R peaks")
        plt.show()
        
    return peaks


# In[364]:


edf_files = glob.glob('/works/datalake/data/**/*.edf', recursive=True)
len(edf_files)


# In[365]:


edf_files[332]


# In[366]:


path = '/works/datalake/data/AU/38/46/0308182.edf'


# In[367]:


#path = '/works/datalake/data/AU/41/189/6170217.edf' #R-peak 반전


# In[368]:


#224 : High peak 걸러야함
#275 :낮은 파형도 인식 문제 해결
#293 : 반전 issue
#312 : Noise 유형 파악 - 분석이 가능한 파형인지 파악할 것
#316 : Noise 유형 파악
#332 : R peak 질문
#337 : 음수 양수 혼합 peak 문제 db3 -> db6로 해결


# In[380]:


#f = pyedflib.EdfReader(edf_files[337]) #283 
f = pyedflib.EdfReader(edf_files[338]) #283 


# In[381]:


f.getNSamples()


# In[382]:


sample_rate = int(f.getSampleFrequency(0))


# In[383]:


f.readSignal(0)


# In[384]:


ecg_signal = f.readSignal(0)[2000:4000]
plt.plot(ecg_signal)


# In[385]:


plt.plot(signal_amp(ecg_signal))


# In[386]:


plt.plot(ecg_signal[0:292])


# In[387]:


sigbufs = np.zeros((1, f.getNSamples()[0]))
sigbufs[0, :] = f.readSignal(0)
sigbufs.shape
signal = sigbufs[0]
original_signal = signal[2000:4000]
bandpass_filtered = bandpass_filter(original_signal, sample_rate, 100, 0.1)
f.close()


# In[388]:


start_time = 30
dur_time = 10
sampling_rate = int(f.getSampleFrequency(0))
sampling_rate


# In[389]:


r_peak_detect(ecg_signal, sampling_rate=sampling_rate, smoothwindow=0.5,
                        avgwindow=1,
                        gradthreshweight=1,
                        minlenweight=0.6,
                        maxlenweight=50,
                        mindelay=0.12,
                        if_show=True, start_time=0, dur_time=20)

