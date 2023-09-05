import pandas as pd
import numpy as np
import os
import sys
import time
from tqdm import tqdm
import warnings
import pre_processing as pp

lead_dict = {0:'I',1:'II',2:'III',3:'aVR',4:'aVL',5:'aVF',6:'V1',7:'V2',8:'V3',9:'V4',10:'V5',11:'V6'}

use_lead = [0,1,6,8,9,10]

def find_peak(original_ecg_array):
    
    ecg_dict = {}
    Lead_II_wave = pp.remove_outlier(original_ecg_array[:,1])
        
    for lead_number in use_lead:    
        ecg_dict[lead_number] = {}
        
        ecg_wave = pp.remove_outlier(original_ecg_array[:,lead_number])

        """
        0 : Lead I
        1 : Lead II 
        2 : Lead III
        3 : aVR 
        4 : aVL
        5 : aVF 
        6 : Lead V1 
        7 : Lead V2 
        8 : Lead V3
        9 : Lead V4
        10 : Lead V5 
        11 : Lead V6 
        """

        ##################
        ### preprocessing!
        ##################

        ecg_wave = pp.Norm(ecg_wave)

        if(lead_number != 0):
            rpeaks = pp.select_r_peaks(Lead_II_wave) #Lead I 제외하고 Lead II로 R peak 정함
        else:
            rpeaks = pp.select_r_peaks(ecg_wave) 

        if(rpeaks == 0):
            continue

        # 첫번째 R-peak 이후 PQRST Beat에서  마지막 R-peak 이전 PQRST Beat 까지의 Rhythm

        ### F-score, correlation
        try:
            interest_ecg, template_ecg = pp.make_ecg_beats(ecg_wave,rpeaks)

            # 비트마다 데이터 포인트가 일정하지 않으므로 각 비트 길이를 모두 구해서 최소값 불러오기 - 최소값에 맞춰서 Correlation 구해야 하므로

            min_length = pp.calc_min(interest_ecg, template_ecg)
            # 기울기를 구해서 기울기가 가장 작은 인덱스는 최소 길이에 맞춰서 버림
            tuned_interest_ecg = pp.tuning_ecg_length(interest_ecg, min_length)
            average_correlation, tuned_template_ecg, template_idx =pp.calc_corr(template_ecg,tuned_interest_ecg,min_length)
            ecg_difference = tuned_interest_ecg - tuned_template_ecg[template_idx]
            f_wave_score = pp.calc_f_score(tuned_interest_ecg,ecg_difference)

        except:

            f_wave_score = None
            average_correlation = None

        ecg_dict[lead_number]['f_wave_score'] = f_wave_score
        ecg_dict[lead_number]['corr_mean'] = average_correlation
        #ecg_array[unique_id]['low_psd'],ecg_array[unique_id]['semi_low_psd'],ecg_array[unique_id]['high_psd'] = sp.psd_score(ecg_wave)

        try:
            waves_cwt = pp.get_pqrst_feature(ecg_wave, rpeaks)

        except KeyError as e:
            try:
                rpeaks['ECG_R_Peaks'] = rpeaks['ECG_R_Peaks'][:-1]     
                waves_cwt = pp.get_pqrst_feature(ecg_wave, rpeaks, 'cwt')
            except:
                ecg_dict[lead_number]['desc'] = 'R_peak_error'
                continue
            #print('DWT 변환 오류 해결 : ' + unique_id)
        except ValueError as e:
            print('DWT 변환 오류 : ' )
            ecg_dict[lead_number]['desc'] = 'PVC_suspect'
            continue
        except ZeroDivisionError as e:
            print('DWT 변환 오류 : ' )
            ecg_dict[lead_number]['desc'] = 'zero_division'
            print(ecg_wave)
            continue

        waves_cwt['ECG_R_Peaks'] = rpeaks['ECG_R_Peaks']

        try:
            #df_ecg = pd.DataFrame(waves_cwt)
            df_ecg = pd.DataFrame.from_dict(waves_cwt, orient='index')
            df_ecg = pp.sorted_col(df_ecg)
            for col in df_ecg.index:
                pp.shift_col(df_ecg,col)
        except:
            ecg_dict['desc'] = 'peak_length_error'
            continue
        
        df_ecg = df_ecg.T

        waves_cwt['ECG_QRS_Complex'] = df_ecg['ECG_R_Offsets'].values - df_ecg['ECG_R_Onsets'].values
        waves_cwt['ECG_PR_Interval'] = df_ecg['ECG_R_Onsets'].values - df_ecg['ECG_P_Onsets'].values
        waves_cwt['ECG_PR_Segment'] = df_ecg['ECG_R_Onsets'].values - df_ecg['ECG_P_Offsets'].values
        waves_cwt['ECG_ST_Segment'] = df_ecg['ECG_T_Onsets'].values - df_ecg['ECG_R_Offsets'].values
        waves_cwt['ECG_QT_Interval'] = df_ecg['ECG_T_Offsets'].values - df_ecg['ECG_R_Onsets'].values
        waves_cwt['ECG_P_duration'] = df_ecg['ECG_P_Offsets'].values - df_ecg['ECG_P_Onsets'].values

        waves_cwt['ECG_PP_interval'] = df_ecg['ECG_P_Peaks'].diff(1).values
        waves_cwt['ECG_QQ_interval'] = df_ecg['ECG_Q_Peaks'].diff(1).values
        waves_cwt['ECG_RR_interval'] = df_ecg['ECG_R_Peaks'].diff(1).values ## R-R interval
        waves_cwt['ECG_SS_interval'] = df_ecg['ECG_S_Peaks'].diff(1).values
        waves_cwt['ECG_TT_interval'] = df_ecg['ECG_T_Peaks'].diff(1).values

        P_len = df_ecg['ECG_P_Peaks'].notnull().sum()
        Q_len = df_ecg['ECG_Q_Peaks'].notnull().sum()
        R_len = df_ecg['ECG_R_Peaks'].notnull().sum()
        S_len = df_ecg['ECG_S_Peaks'].notnull().sum()
        T_len = df_ecg['ECG_T_Peaks'].notnull().sum()
        
        # No need for inference
        try:
            P_ON_list = pp.fill_nans_scipy1(waves_cwt['ECG_P_Onsets'])
            P_OFF_list = pp.fill_nans_scipy1(waves_cwt['ECG_P_Offsets'])
        except ValueError as e:
            print(waves_cwt)
            print('P_wave Null값 오류 : ' )
            ecg_dict[lead_number]['P_wave_skew'], ecg_dict[lead_number]['P_wave_kurtosis'] = np.nan, np.nan
        
        p_waves = pp.sum_p_wave(P_ON_list,P_OFF_list,ecg_wave)
        ecg_dict[lead_number]['P_wave_skew'], ecg_dict[lead_number]['P_wave_kurtosis'] = pp.skew_kurtosis(p_waves)

        peak_dict = {'P_Peaks':'P', 'P_Onsets':'P_on', 'P_Offsets':'P_off','Q_Peaks':'Q', 
                        'R_Peaks':'R', 'S_Peaks':'S', 'T_Peaks':'T'}

        if((lead_number == 6 )|( lead_number == 7 )|(lead_number == 8)): # V1,V2,V3면 Q Peak 제외
            peak_dict.pop('Q_Peaks') 

        for peak in list(peak_dict.keys()):
            name = peak_dict[peak]
            peak_array = np.array(waves_cwt[f'ECG_{peak}'])
            peak_list = peak_array[~np.isnan(peak_array)].astype(int).tolist()

            if(len(peak_list)>0):
                ecg_dict[lead_number][f'{name}_amplitude_mean'],\
                ecg_dict[lead_number][f'{name}_amplitude_std'],\
                ecg_dict[lead_number][f'{name}_amplitude_min'],\
                ecg_dict[lead_number][f'{name}_amplitude_max'] = pp.statistics(ecg_wave[peak_list])

            else:
                ecg_dict[lead_number][f'{name}_amplitude_mean'] = np.nan
                ecg_dict[lead_number][f'{name}_amplitude_std'] = np.nan
                ecg_dict[lead_number][f'{name}_amplitude_min'] = np.nan
                ecg_dict[lead_number][f'{name}_amplitude_max'] = np.nan


        durs_list = ['ECG_QRS_Complex', 'ECG_PR_Interval', 'ECG_PR_Segment','ECG_ST_Segment','ECG_QT_Interval', 'ECG_P_duration',\
            'ECG_PP_interval','ECG_QQ_interval','ECG_RR_interval','ECG_SS_interval','ECG_TT_interval']

        if((lead_number != 6 )&( lead_number != 7 )&(lead_number != 8)): # V1-V3이 아니면 QQ_interval 추가
            durs_list.insert(0, 'ECG_QQ_interval')

        for dur in durs_list:

            ecg_dict[lead_number][f'{dur}_mean'],ecg_dict[lead_number][f'{dur}_std'],ecg_dict[lead_number][f'{dur}_min'],\
            ecg_dict[lead_number][f'{dur}_max'] = pp.statistics(waves_cwt[dur])#df_ecg.describe()[f'{dur}'][['mean','std','min','max']]

        ecg_dict[lead_number]['P_wave_number'] = P_len
        ecg_dict[lead_number]['R_wave_number'] = R_len
        ecg_dict[lead_number]['S_wave_number'] = S_len
        ecg_dict[lead_number]['T_wave_number'] = T_len

        if((lead_number != 6 )&( lead_number != 7 )&(lead_number != 8)): # V1-V3이 아니면 Q_wave_number 추가
            ecg_dict[lead_number]['Q_wave_number'] = Q_len

        warnings.filterwarnings("ignore") #코드 중간에 삽입해야 함

    return ecg_dict