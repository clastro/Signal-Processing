import neurokit2 as nk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
import scipy
from tqdm import tqdm
from scipy import signal
from scipy.signal import butter, iirnotch, lfilter

df_train = pd.read_csv('',encoding='utf-8-sig')

columns = ['P_amplitude_mean','P_amplitude_median','P_amplitude_std','P_amplitude_min','P_amplitude_max',\
           'P_on_amplitude_mean','P_on_amplitude_median','P_on_amplitude_std','P_on_amplitude_min','P_on_amplitude_max',\
           'P_off_amplitude_mean','P_off_amplitude_median','P_off_amplitude_std','P_off_amplitude_min','P_off_amplitude_max',\
           'Q_amplitude_mean','Q_amplitude_median','Q_amplitude_std','Q_amplitude_min','Q_amplitude_max',\
           'R_amplitude_mean','R_amplitude_median','R_amplitude_std','R_amplitude_min','R_amplitude_max',\
           'S_amplitude_mean','S_amplitude_median','S_amplitude_std','S_amplitude_min','S_amplitude_max',\
           'T_amplitude_mean','T_amplitude_median','T_amplitude_std','T_amplitude_min','T_amplitude_max',\
           'RR_interval_mean','RR_interval_median','RR_interval_std','RR_interval_min','RR_interval_max',\
           'PP_interval_mean','PP_interval_median','PP_interval_std','PP_interval_min','PP_interval_max',\
           'TT_interval_mean','TT_interval_median','TT_interval_std','TT_interval_min','TT_interval_max',\
           'PR_interval_mean','PR_interval_median','PR_interval_std','PR_interval_min','PR_interval_max',\
           'P_width_mean','P_width_median','P_width_std','P_width_min','P_width_max',\
           'QRS_width_mean','QRS_width_median','QRS_width_std','QRS_width_min','QRS_width_max',\
           'HRV_MeanNN','HRV_SDNN','HRV_MedianNN']

df_train_wave = pd.DataFrame(columns = columns)

ecg_array = {}

def bandpass(lowcut, highcut, order=5):
    nyq = 0.5 * 500
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
  
i = 0
for unique_id in tqdm(df_train['unique_id']):
    
    ecg_array[unique_id] = {}

    ecg_wave = np.load('/smc_work/data/smc_numpy_data/II/'+unique_id+'.npy')
    
    b , a = bandpass(0.5, 45)
    filtered_ecg = signal.filtfilt(b,a,ecg_wave)
    
    _, rpeaks = nk.ecg_peaks(filtered_ecg, sampling_rate=500)
    _, waves_peak = nk.ecg_delineate(filtered_ecg, rpeaks, sampling_rate=500, method="peak")
    try:
        _, waves_cwt = nk.ecg_delineate(filtered_ecg, rpeaks, sampling_rate=500, method="cwt", show=False, show_type='peaks')
    except :
        _, waves_cwt = nk.ecg_delineate(filtered_ecg, rpeaks, sampling_rate=500, method="dwt", show=False, show_type='peaks')
    waves_cwt['ECG_RR_interval'] = np.diff(rpeaks['ECG_R_Peaks']).tolist() ## R-R interval
    waves_cwt['ECG_PP_interval'] =  np.diff(waves_cwt['ECG_P_Peaks']).tolist() ## P-P interval
    waves_cwt['ECG_TT_interval'] =  np.diff(waves_cwt['ECG_T_Peaks']).tolist() ## T-T interval
    R_len = len(rpeaks['ECG_R_Peaks'])
    P_len = len(waves_cwt['ECG_P_Peaks'])
    if(R_len == P_len):
        waves_cwt['ECG_PR_interval'] = (rpeaks['ECG_R_Peaks'] - waves_cwt['ECG_P_Peaks']).tolist() ## P-R interval
    elif(R_len == P_len + 1):
        waves_cwt['ECG_PR_interval'] = (rpeaks['ECG_R_Peaks'][1:] - waves_cwt['ECG_P_Peaks']).tolist() ## P-R interval
    waves_cwt['ECG_P_width'] = (np.array(waves_cwt['ECG_P_Offsets']) - waves_cwt['ECG_P_Onsets']).tolist()
    waves_cwt['ECG_QRS_width'] = (np.array(waves_cwt['ECG_S_Peaks']) - waves_cwt['ECG_Q_Peaks']).tolist()
    waves_cwt['ECG_R_Peaks'] = rpeaks['ECG_R_Peaks'].tolist()
    
    p_peak_array = np.array(waves_cwt['ECG_P_Peaks'])
    p_peak_list = p_peak_array[~np.isnan(p_peak_array)].astype(int).tolist()
    
    if(len(p_peak_list)>0):
        
        ecg_array[unique_id]['P_amplitude_mean'] = np.nanmean(ecg_wave[p_peak_list])
        ecg_array[unique_id]['P_amplitude_median']  = np.nanmedian(ecg_wave[p_peak_list])
        ecg_array[unique_id]['P_amplitude_std'] = np.nanstd(ecg_wave[p_peak_list])
        ecg_array[unique_id]['P_amplitude_min'] = np.nanmin(ecg_wave[p_peak_list])
        ecg_array[unique_id]['P_amplitude_max'] = np.nanmax(ecg_wave[p_peak_list])
    
    p_onset_array = np.array(waves_cwt['ECG_P_Onsets'])
    p_onset_list = p_onset_array[~np.isnan(p_onset_array)].astype(int).tolist()
    
    
    if(len(p_onset_list)>0):
        
        ecg_array[unique_id]['P_on_amplitude_mean'] = np.nanmean(ecg_wave[p_onset_list])
        ecg_array[unique_id]['P_on_amplitude_median']  = np.nanmedian(ecg_wave[p_onset_list])
        ecg_array[unique_id]['P_on_amplitude_std'] = np.nanstd(ecg_wave[p_onset_list])
        ecg_array[unique_id]['P_on_amplitude_min'] = np.nanmin(ecg_wave[p_onset_list])
        ecg_array[unique_id]['P_on_amplitude_max'] = np.nanmax(ecg_wave[p_onset_list])
    
    p_offset_array = np.array(waves_cwt['ECG_P_Offsets'])
    p_offset_list = p_offset_array[~np.isnan(p_offset_array)].astype(int).tolist()
    
    
    if(len(p_offset_list)>0):
        
        ecg_array[unique_id]['P_off_amplitude_mean'] = np.nanmean(ecg_wave[p_offset_list])
        ecg_array[unique_id]['P_off_amplitude_median']  = np.nanmedian(ecg_wave[p_offset_list])
        ecg_array[unique_id]['P_off_amplitude_std'] = np.nanstd(ecg_wave[p_offset_list])
        ecg_array[unique_id]['P_off_amplitude_min'] = np.nanmin(ecg_wave[p_offset_list])
        ecg_array[unique_id]['P_off_amplitude_max'] = np.nanmax(ecg_wave[p_offset_list])
    
    q_amplitude_array = np.array(waves_cwt['ECG_Q_Peaks'])
    q_amplitude_list = q_amplitude_array[~np.isnan(q_amplitude_array)].astype(int).tolist()
    
    if(len(q_amplitude_list)>0):
        ecg_array[unique_id]['Q_amplitude_mean'] = np.nanmean(ecg_wave[q_amplitude_list])
        ecg_array[unique_id]['Q_amplitude_median']  = np.nanmedian(ecg_wave[q_amplitude_list])
        ecg_array[unique_id]['Q_amplitude_std'] = np.nanstd(ecg_wave[q_amplitude_list])
        ecg_array[unique_id]['Q_amplitude_min'] = np.nanmin(ecg_wave[q_amplitude_list])
        ecg_array[unique_id]['Q_amplitude_max'] = np.nanmax(ecg_wave[q_amplitude_list])
    """
    else:
        ecg_array[unique_id]['Q_amplitude_mean'] = 0
        ecg_array[unique_id]['Q_amplitude_median']  = 0
        ecg_array[unique_id]['Q_amplitude_std'] = 0
        ecg_array[unique_id]['Q_amplitude_min'] = 0
        ecg_array[unique_id]['Q_amplitude_max'] = 0
    """
    r_amplitude_array = np.array(waves_cwt['ECG_R_Peaks'])
    r_amplitude_list = r_amplitude_array[~np.isnan(r_amplitude_array)].astype(int).tolist()
    
    if(len(r_amplitude_list)>0):
        
        ecg_array[unique_id]['R_amplitude_mean'] = np.nanmean(ecg_wave[r_amplitude_list])
        ecg_array[unique_id]['R_amplitude_median']  = np.nanmedian(ecg_wave[r_amplitude_list])
        ecg_array[unique_id]['R_amplitude_std'] = np.nanstd(ecg_wave[r_amplitude_list])
        ecg_array[unique_id]['R_amplitude_min'] = np.nanmin(ecg_wave[r_amplitude_list])
        ecg_array[unique_id]['R_amplitude_max'] = np.nanmax(ecg_wave[r_amplitude_list])
    
    s_amplitude_array = np.array(waves_cwt['ECG_S_Peaks'])
    s_amplitude_list = s_amplitude_array[~np.isnan(s_amplitude_array)].astype(int).tolist()
    
    if(len(s_amplitude_list)>0):
    
        ecg_array[unique_id]['S_amplitude_mean'] = np.nanmean(ecg_wave[s_amplitude_list])
        ecg_array[unique_id]['S_amplitude_median']  = np.nanmedian(ecg_wave[s_amplitude_list])
        ecg_array[unique_id]['S_amplitude_std'] = np.nanstd(ecg_wave[s_amplitude_list])
        ecg_array[unique_id]['S_amplitude_min'] = np.nanmin(ecg_wave[s_amplitude_list])
        ecg_array[unique_id]['S_amplitude_max'] = np.nanmax(ecg_wave[s_amplitude_list])
    
    t_amplitude_array = np.array(waves_cwt['ECG_T_Peaks'])
    t_amplitude_list = t_amplitude_array[~np.isnan(t_amplitude_array)].astype(int).tolist()
    
    if(len(t_amplitude_list)>0):
        
        ecg_array[unique_id]['T_amplitude_mean'] = np.nanmean(ecg_wave[t_amplitude_list])
        ecg_array[unique_id]['T_amplitude_median']  = np.nanmedian(ecg_wave[t_amplitude_list])
        ecg_array[unique_id]['T_amplitude_std'] = np.nanstd(ecg_wave[t_amplitude_list])
        ecg_array[unique_id]['T_amplitude_min'] = np.nanmin(ecg_wave[t_amplitude_list])
        ecg_array[unique_id]['T_amplitude_max'] = np.nanmax(ecg_wave[t_amplitude_list])
    
    ecg_array[unique_id]['RR_interval_mean'] = np.nanmean(waves_cwt['ECG_RR_interval'])
    ecg_array[unique_id]['RR_interval_median'] = np.nanmedian(waves_cwt['ECG_RR_interval'])
    ecg_array[unique_id]['RR_interval_std'] = np.nanstd(waves_cwt['ECG_RR_interval'])
    ecg_array[unique_id]['RR_interval_min'] = np.nanmin(waves_cwt['ECG_RR_interval'])
    ecg_array[unique_id]['RR_interval_max'] = np.nanmax(waves_cwt['ECG_RR_interval'])
    
    ecg_array[unique_id]['PP_interval_mean'] = np.nanmean(waves_cwt['ECG_PP_interval'])
    ecg_array[unique_id]['PP_interval_median'] = np.nanmedian(waves_cwt['ECG_PP_interval'])
    ecg_array[unique_id]['PP_interval_std'] = np.nanstd(waves_cwt['ECG_PP_interval'])
    ecg_array[unique_id]['PP_interval_min'] = np.nanmin(waves_cwt['ECG_PP_interval'])
    ecg_array[unique_id]['PP_interval_max'] = np.nanmax(waves_cwt['ECG_PP_interval'])
    
    ecg_array[unique_id]['TT_interval_mean'] = np.nanmean(waves_cwt['ECG_TT_interval'])
    ecg_array[unique_id]['TT_interval_median'] = np.nanmedian(waves_cwt['ECG_TT_interval'])
    ecg_array[unique_id]['TT_interval_std'] = np.nanstd(waves_cwt['ECG_TT_interval'])
    ecg_array[unique_id]['TT_interval_min'] = np.nanmin(waves_cwt['ECG_TT_interval'])
    ecg_array[unique_id]['TT_interval_max'] = np.nanmax(waves_cwt['ECG_TT_interval'])
    
    ecg_array[unique_id]['PR_interval_mean'] = np.nanmean(waves_cwt['ECG_PR_interval'])
    ecg_array[unique_id]['PR_interval_median'] = np.nanmedian(waves_cwt['ECG_PR_interval'])
    ecg_array[unique_id]['PR_interval_std'] = np.nanstd(waves_cwt['ECG_PR_interval'])
    ecg_array[unique_id]['PR_interval_min'] = np.nanmin(waves_cwt['ECG_PR_interval'])
    ecg_array[unique_id]['PR_interval_max'] = np.nanmax(waves_cwt['ECG_PR_interval'])
    
    ecg_array[unique_id]['P_width_mean'] = np.nanmean(waves_cwt['ECG_P_width'])
    ecg_array[unique_id]['P_width_median'] = np.nanmedian(waves_cwt['ECG_P_width'])
    ecg_array[unique_id]['P_width_std'] = np.nanstd(waves_cwt['ECG_P_width'])
    ecg_array[unique_id]['P_width_min'] = np.nanmin(waves_cwt['ECG_P_width'])
    ecg_array[unique_id]['P_width_max'] = np.nanmax(waves_cwt['ECG_P_width'])
    
    ecg_array[unique_id]['QRS_width_mean'] = np.nanmean(waves_cwt['ECG_QRS_width'])
    ecg_array[unique_id]['QRS_width_median'] = np.nanmedian(waves_cwt['ECG_QRS_width'])
    ecg_array[unique_id]['QRS_width_std'] = np.nanstd(waves_cwt['ECG_QRS_width'])
    ecg_array[unique_id]['QRS_width_min'] = np.nanmin(waves_cwt['ECG_QRS_width'])
    ecg_array[unique_id]['QRS_width_max'] = np.nanmax(waves_cwt['ECG_QRS_width'])
    
    hrv = nk.hrv_time(rpeaks, sampling_rate=500, show=True)
    
    ecg_array[unique_id]['HRV_MeanNN'] = hrv['HRV_MeanNN'][0]
    ecg_array[unique_id]['HRV_SDNN'] = hrv['HRV_SDNN'][0]
    ecg_array[unique_id]['HRV_MedianNN'] = hrv['HRV_MedianNN'][0]     
    ecg_array[unique_id]['label'] = df_train[df_train['unique_id']==unique_id]['label'][i]
    if( i % 100 == 0):
        plt.cla() # Clear the current axes
        plt.clf() # Clear the current figure
        plt.close()
    i += 1
