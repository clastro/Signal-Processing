import neurokit2 as nk
from itertools import tee
import numpy as np
import scipy
from scipy import signal
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import butter, iirnotch, lfilter
import pywt

def select_r_peaks(ecg, tag_inverse=1):
    """

    r_peak extraction from ecg

    Args:
        ecg (ndarray) : ecg numpy array 5000
        tag_inverse (int) : 1 (default) or -1 (if inversed signal)
    Returns:
        rpeaks (list) : rpeaks List
    """

    try:
        _, rpeaks = nk.ecg_peaks(ecg*tag_inverse, sampling_rate=500,method='neurokit2')
    except:
        rpeaks = 0

    return rpeaks

def get_pqrst_feature(filtered_ecg,rpeaks,method = "cwt"):

    sampling_rate = 500
    
    _, waves_cwt = nk.ecg_delineate(filtered_ecg, rpeaks, sampling_rate=sampling_rate, method=method, show=False, show_type='peaks')
    return waves_cwt

def make_ecg_beats(filtered_ecg, rpeaks):

    """

    divided into ecg beats from rhythm

    Args:
        filtered_ecg (ndarray) : ecg numpy array 5000
        rpeaks (list) : rpeaks list
    Returns:
        interest_ecg (ndarray) : first beat from ECG Rhythm
        template_ecg (ndarray) : Other beats except first beat from ECG Rhythm
    """

    ecg_beats = []
    
    for i in [*pairwise(rpeaks['ECG_R_Peaks'])]:
        try:
            ecg_beats.append(filtered_ecg[i[0]:i[1]])
        except:
            continue

    interest_ecg = ecg_beats[0]
    template_ecg = ecg_beats[1:]
    
    return interest_ecg, template_ecg

def sum_p_wave(P_on,P_off,wave):
    p_wave = np.array([])
    for a,b in zip(P_on,P_off):
        p_wave = np.append(p_wave,wave[a:b])
    return p_wave

def skew_kurtosis(p_wave):
    return stats.skew(p_wave), stats.kurtosis(p_wave)

def cross_entropy(template_p_wave, interest_p_wave):

    # 교차 엔트로피 합
    delta = 1e-7        

    return -np.sum(interest_p_wave * np.log(template_p_wave + delta))

def calc_corr(template_ecg,tuned_interest_ecg,min_length):

    """
    Args:
        template_ecg (ndarray) : template_ecg
        tuned_interest_ecg (ndarray) : fixed length interest ecg
        min_length (int) : minimum value of beats in ECG rhythm # 가장 작은 값으로 맞춰야 비교 가능 
    Returns:
        average_correlation (float) : mean of correlation between beats
        tuned_template_ecg (list) : tuned
        template_idx (int) : template ecg index
    """

    corr_score = []
    tuned_template_ecg = []
    
    for i in range(len(template_ecg)):
        one_tuned_template_ecg = tuning_ecg_length(template_ecg[i], min_length)
        corr_score.append(np.corrcoef(tuned_interest_ecg,one_tuned_template_ecg)[1][0]) #상관계수 구하기
        tuned_template_ecg.append(one_tuned_template_ecg)
        
    average_correlation = np.mean(corr_score)
    template_idx = np.argpartition(corr_score, len(corr_score) // 2)[len(corr_score) // 2] # correlation 중간값을 Template으로  
    
    return average_correlation,tuned_template_ecg,template_idx


def calc_f_score(tuned_interest_ecg,ecg_difference):

    """
    calculation fibrilatory wave score (AFib score)

    Args:
        tuned_interest_ecg (ndarray) 
        ecg_difference (ndarray) : diffrence between template ecg and interest ecg
    Returns:
        f_wave_score (float) : fibrilatory wave score
    """
    sampling_rate = 500
    nfft_value = 2**12

    afib_low_frequnecy = 4
    afib_high_frequency = 9

    b , a = bandpass(afib_low_frequnecy, afib_high_frequency)

    filtered_interest_ecg = signal.filtfilt(b,a,tuned_interest_ecg)
    filtered_ecg_difference = signal.filtfilt(b,a,ecg_difference)

    ffi, Pfi = signal.periodogram(filtered_interest_ecg, sampling_rate, nfft=nfft_value)
    ff, Pf = signal.periodogram(filtered_ecg_difference, sampling_rate, nfft=nfft_value)

    interest_power = np.dot(ffi,Pfi)
    f_wave_power = np.dot(ff,Pf)
    f_wave_score = f_wave_power/interest_power
    
    return f_wave_score

def fill_nans_scipy1(padata, pkind='linear'):
    """
    Interpolates data to fill nan values

    Parameters:
        padata : nd array 
            source data with np.NaN values

    Returns:
        nd array 
            resulting data with interpolated values instead of nans
    """
    padata = np.array(padata)
    aindexes = np.arange(padata.shape[0])
    agood_indexes, = np.where(np.isfinite(padata))
    f = interp1d(agood_indexes
               , padata[agood_indexes]
               , bounds_error=False
               , copy=False
               , fill_value="extrapolate"
               , kind=pkind)
    result = f(aindexes)
    result = np.array(result,dtype='int64')
    return result.tolist()

def statistics(wave_list):
    
    nanmean = np.nanmean(wave_list)
    nanstd = np.nanstd(wave_list)
    nanmin = np.nanmin(wave_list)
    nanmax = np.nanmax(wave_list)
    
    return nanmean,nanstd,nanmin,nanmax

def pairwise(iterable):

    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)

    return zip(a, b)

def calc_min(interest_ecg,template_ecg):

    for i in range(len(template_ecg)):
        if(i ==0):
            min_length = template_ecg[i].shape[0]
        else:
            min_length = min(min_length,template_ecg[i].shape[0])

    min_length = min(min_length,interest_ecg.shape[0])
    
    return min_length 


def tuning_ecg_length(ecg,min_length):
    
    grad_ecg = np.gradient(ecg)
    idx = (-abs(grad_ecg)).argsort()[:min_length]
    idx.sort()
    tuned_ecg = ecg[idx]
    
    return tuned_ecg

def change_column(columns,lead_num):

    new_columns = [] #새로운 칼럼값을 넣을 List
    for column in columns.tolist():
        if((column == 'unique_id')|(column == 'sex')|(column == 'age')):
            mod_column = column
        else:
            mod_column = lead_num +column
        new_columns.append(mod_column)

    return new_columns

def Norm(ecg_input):

    norm_ecg = (ecg_input - np.mean(ecg_input))/np.std(ecg_input)

    return norm_ecg

def psd_ecg(filtered_ecg):
    
    min_freq = 0
    max_freq = 40
    window = 0.5
    welch = nk.signal_psd(filtered_ecg, method="welch", min_frequency=min_freq, max_frequency=max_freq,window=window)

    return welch['Power'].values

def psd_score(ecg):
    """
    PSD score calculation

    Args: 
        ecg (array): ecg numpy array 5000

    Returns:
        low_freq_index (float) : PSD in low frequency ( 1 ~ 5 Hz )
        low_freq_index (float) : PSD in semi low frequency ( 5 ~ 10 Hz )
        low_freq_index (float) : PSD in high low frequency ( 30 ~ 50 Hz )
    """
    freqs, psd = signal.welch(ecg)
    
    power_index = sum(psd) + 1e-10 #very tiny number for 0 division error
    
    low_freq_index = sum(psd[1:5])/power_index
    semi_low_freq_index = sum(psd[5:10])/power_index
    noise_index = sum(psd[30:50])/power_index
    
    return low_freq_index,semi_low_freq_index,noise_index

def bandpass(lowcut, highcut, order=5):
    """
    butter BandPass Filter in Signal

    Args:
        lowcut (int) : low frequency threshold
        hightcut (int) : high frequency threshold
    Returns:
        b,a (ndarray) : Numerator (b) and denominator (a) polynomials of the IIR filter

    """
    nyq = 0.5 * 500
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    return b, a

def filter_wave(ecg_wave):

    """
    filtfilt noise filtering

    Args:
        ecg_wave (ndarray) : ecg numpy array
    Returns:
        filtered_ecg (ndarray) : filtered by butter_bandpass filter

    """
    b , a = bandpass(0.5, 45) # Lower : 0.5Hz , Upper : 45Hz
    filtered_ecg = signal.filtfilt(b,a,ecg_wave)

    return filtered_ecg

def make_zscore(ecg):
    return np.abs(stats.zscore(ecg))
  
def remove_outlier(ecg):
    z_threshold = 10
    z = make_zscore(ecg)
    outlier_index = np.where(z > z_threshold)
    return np.delete(ecg, outlier_index)

def sorted_col(df):
    ECG_columns = ['P_Onsets', 'P_Peaks','P_Offsets','R_Onsets', 'Q_Peaks','R_Peaks',\
               'R_Offsets','S_Peaks','T_Onsets','T_Peaks','T_Offsets']
    return df.loc[ECG_columns]

def data_interpolate(df,opt = 'linear'):
    return df.interpolate(opt)

def shift_col(df,col):
    
    n = 5 # 4번째 Peak까지 보면서 정렬
    
    '''
    각 단계별로 Rank를 계산하여 P-Q-R-S-T 순서대로 맞추는 함수
    R_Onsets, R_Offsets 의 경우 경계가 다소 모호하기 때문에 순서가 조금씩 바뀌는 경우를 허용하였음.
    '''
    
    original_time_rank = {'P_Onsets':1,'P_Peaks':2,'P_Offsets':3,'R_Onsets':4,'Q_Peaks':5,\
            'R_Peaks':6,'S_Peaks':7,'R_Offsets':8,'T_Onsets':9,'T_Peaks':10,'T_Offsets':11}
    
    for i in range(0,n):
        if (np.isnan(df.loc[col][i])):
            fill_value = 0
        else:
            fill_value = 5001 # 5001 : 실제 나올 수 없는 숫자
        
        score_rank = np.round(df.fillna(fill_value)[i].rank(method='min')[col])
        
        if((col == 'P_Offsets') & ( score_rank == 4)):
            score_rank = 3
        if((col == 'R_Onsets') & (score_rank == 5)):
            score_rank = 4
        if((col == 'R_Onsets') & ( score_rank == 3)):
            score_rank = 4
        if((col == 'Q_Peaks') & ( score_rank == 4)):
            score_rank = 5
        if((col == 'R_Offsets') & (score_rank == 7)):
            score_rank = 8 
        if((col == 'S_Peaks') & ( score_rank == 8)):
            score_rank = 7
        
        if(score_rank > original_time_rank[col]):
            df.loc[col] = df.loc[col].shift()
        

"""
def shift_col(df, col):
    n = 5  # 4번째 Peak까지 보면서 정렬
    
    # Original rank mapping
    original_time_rank = {
        'P_Onsets': 1, 'P_Peaks': 2, 'P_Offsets': 3, 'R_Onsets': 4,
        'Q_Peaks': 5, 'R_Peaks': 6, 'S_Peaks': 7, 'R_Offsets': 8,
        'T_Onsets': 9, 'T_Peaks': 10, 'T_Offsets': 11
    }

    # Validate if col exists in df
    if col not in df.columns:
        raise ValueError(f"Column {col} does not exist in the DataFrame.")
    
    for i in range(n):
        fill_value = 0 if np.isnan(df.loc[col][i]) else 5001
        
        # Calculate rank
        score_rank = np.round(df.fillna(fill_value)[i].rank(method='min')[col])
        
        # Adjust score_rank based on conditions
        if (col == 'P_Offsets' and score_rank == 4):
            score_rank = 3
        elif (col == 'R_Onsets' and score_rank in [3, 5]):
            score_rank = 4
        elif (col == 'Q_Peaks' and score_rank == 4):
            score_rank = 5
        elif (col == 'R_Offsets' and score_rank == 7):
            score_rank = 8
        elif (col == 'S_Peaks' and score_rank == 8):
            score_rank = 7
        
        # Shift if necessary
        if score_rank > original_time_rank[col]:
            df.loc[col] = df.loc[col].shift()

"""
