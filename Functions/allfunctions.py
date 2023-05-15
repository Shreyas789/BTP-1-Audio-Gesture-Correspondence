from dtaidistance.subsequence.dtw import subsequence_alignment, subsequence_search
from dtaidistance.subsequence import SubsequenceAlignment
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from IPython import display
from IPython.display import Audio
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.ndimage import gaussian_filter
import librosa

import warnings
warnings.filterwarnings('ignore')


def dtw_operation(series, target, plot_series = True, plot_path = True, **kwargs):
    
    dist, matrix = dtw.warping_paths(series, target, **kwargs)
    result, path = dtw.warp(series,target,**kwargs)
    print("DTW distance:",dist)
    print("Path length:", len(path))
    print("Window:", kwargs['window'])
    print()
    
    if plot_series:
        fig,axes = plt.subplots(3,1,figsize = (8,8))
        dtwvis.plot_warp(series, target, result, path,fig = fig, axs= axes)

        for ax in axes:
            ax.set_xlim(0,max(len(target),len(series))-1)
            ax.set_ylim(0,max(list(target)+list(series)))

        axes[0].legend(labels = ['Series (S1)'])
        axes[1].legend(labels = ['Target (S2)'])
        axes[2].legend(labels = ["Warped Result (S1')"]);
        plt.show();
    
    if plot_path:
        fig = plt.figure(figsize = (10,10))
        dtwvis.plot_warpingpaths(series,target,matrix, figure = fig, showlegend=True)
        plt.show();
    
    return result

def split_into_segments(data, min_length = 0.5):
    '''
    Takes a pitch contour of entire pakad and splits it at parts of silence
    '''
#     data = pd.read_csv(pakad_pitch_file, names = ['time', 'pitch', 'energy'], header = 0)
    data['voiced'] = data['pitch'] != -3000 #boolean
    voiced = np.array(data['voiced'])
    start = 0
    end = 0
    segments = []
    for i in data.index.values[1:]:
        if (not voiced[i-1]) and voiced[i]:#if preceding one is False and this one is true
            start = i
        elif (voiced[i-1] and (not voiced[i])) or i == data.index.values[-1]: #if preceding is True and this is False
            end = i
            if data['time'][end] - data['time'][start] >= min_length:
                segments.append((round(data['time'][start],2), round(data['time'][end],2)))
        else:
            continue
    return data, segments


def all_occurences_of_phrase(phrase_df, pakad_pitch_contour_dir, target_idx):
    '''
    Returns:
    - original_list: List of unwarped (original) query templates
    - warped_list: List of warped (to the median length) query templates
    '''
    # phrase_df = pakad_timestamps.loc[pakad_timestamps['Phrase']==phrase]
    # phrase_df['Length'] = phrase_df['End'] - phrase_df['Start']
    
    # n = len(phrase_df['Length'])
    # phrase_df = phrase_df.sort_values(by = 'Length').reset_index()
    # median = phrase_df['Length'][np.floor(n/2)]
    # phrase_df['Valid'] = phrase_df['Length'].apply(lambda x: x>median/2 and x<median*2)
    # phrase_df = phrase_df.loc[phrase_df['Valid']].reset_index(drop=True)
    # n = len(phrase_df['Length'])
    # new_target_idx = phrase_df.loc[phrase_df['Length']==median].index.values[0]
    # print('Target File:',phrase_df.loc[new_target_idx,'File'])
    query_start2 = phrase_df.loc[target_idx,'Start']
    query_end2 = phrase_df.loc[target_idx,'End']
    # query_start2 = phrase_df.loc[new_target_idx,'Start']
    # query_end2 = phrase_df.loc[new_target_idx,'End']
    # pitch_contour_pakad2 = pd.read_csv(f"{pakad_pitch_contour_dir}/{phrase_df.loc[new_target_idx,'File']}.csv")
    pitch_contour_pakad2 = pd.read_csv(f"{pakad_pitch_contour_dir}/{phrase_df.loc[target_idx,'File']}.csv")
    pitch_contour_pakad2['Select'] = pitch_contour_pakad2['time'].apply(lambda x: x >= query_start2 and x <= query_end2)
    pitch_contour_query2 = pitch_contour_pakad2.loc[pitch_contour_pakad2['Select']]
    voiced_part = split_into_segments(pitch_contour_query2.reset_index())[1][0]
    pitch_contour_query2 = pitch_contour_query2.loc[int(voiced_part[0]*100)+1:int(voiced_part[1]*100)-1]
    phrase2 = np.array(pitch_contour_query2['pitch'])
    
    result_list = []
    original_list = []
    for i in phrase_df.index.values:
        
#         singer_raga1 = pakad_timestamps.loc[pakad_timestamps['File'] == singer1+'_Pakad_'+raga]
#         print(singer_raga1)
        query_start1 = phrase_df.loc[i,'Start']
        query_end1 = phrase_df.loc[i,'End']
        pitch_contour_pakad1 = pd.read_csv(f"{pakad_pitch_contour_dir}/{phrase_df.loc[i,'File']}.csv")
        pitch_contour_pakad1['Select'] = pitch_contour_pakad1['time'].apply(lambda x: x >= query_start1 and x <= query_end1)
        pitch_contour_query1 = pitch_contour_pakad1.loc[pitch_contour_pakad1['Select']]
        voiced_part = split_into_segments(pitch_contour_query1.reset_index())[1][0]
        pitch_contour_query1 = pitch_contour_query1.loc[int(voiced_part[0]*100)+1:int(voiced_part[1]*100)-1]
        
        phrase1 = np.array(pitch_contour_query1['pitch'])
        original_list.append(phrase1)
        
        kwargs = {'window':100,'max_step':None, 'penalty':200, 'max_dist':None}
        print(phrase_df.loc[i,'File'])
        result = dtw_operation(series = phrase1, target = phrase2,plot_series = False,**kwargs)
        result_list.append(result)
    
    return original_list, result_list


###############From FMP notebook############################

def sonify_trajectory_with_sinusoid(traj, audio_len, Fs=22050, amplitude=0.3, smooth_len=11):
    """Sonification of trajectory with sinusoidal

    Notebook: C8/C8S2_FundFreqTracking.ipynb

    Args:
        traj (np.ndarray): F0 trajectory (time in seconds, frequency in Hz)
        audio_len (int): Desired audio length in samples
        Fs (scalar): Sampling rate (Default value = 22050)
        amplitude (float): Amplitude (Default value = 0.3)
        smooth_len (int): Length of amplitude smoothing filter (Default value = 11)

    Returns:
        x_soni (np.ndarray): Sonification
    """
    # unit confidence if not specified
    if traj.shape[1] < 3:
        confidence = np.zeros(traj.shape[0])
        confidence[traj[:, 1] > 0] = amplitude
    else:
        confidence = traj[:, 2]

    # initialize
    x_soni = np.zeros(audio_len)
    amplitude_mod = np.zeros(audio_len)

    # Computation of hop size
    # sine_len = int(2 ** np.round(np.log(traj[1, 0]*Fs) / np.log(2)))
    sine_len = int(traj[1, 0] * Fs)

    t = np.arange(0, sine_len) / Fs
    phase = 0

    # loop over all F0 values, insure continuous phase
    for idx in np.arange(0, traj.shape[0]):
        cur_f = traj[idx, 1]
        cur_amp = confidence[idx]

        if cur_f == 0:
            phase = 0
            continue

        cur_soni = np.sin(2*np.pi*(cur_f*t+phase))
        diff = np.maximum(0, (idx+1)*sine_len - len(x_soni))
        if diff > 0:
            x_soni[idx * sine_len:(idx + 1) * sine_len - diff] = cur_soni[:-diff]
            amplitude_mod[idx * sine_len:(idx + 1) * sine_len - diff] = cur_amp
        else:
            x_soni[idx*sine_len:(idx+1)*sine_len-diff] = cur_soni
            amplitude_mod[idx*sine_len:(idx+1)*sine_len-diff] = cur_amp

        phase += cur_f * sine_len / Fs
        phase -= 2 * np.round(phase/2)

    # filter amplitudes to avoid transients
    amplitude_mod = np.convolve(amplitude_mod, np.hanning(smooth_len)/np.sum(np.hanning(smooth_len)), 'same')
    x_soni = x_soni * amplitude_mod
    return x_soni


def cents_to_hz(F_cent, F_ref=55.0):
    """Converts frequency in cents to Hz

    Notebook: C8/C8S2_FundFreqTracking.ipynb

    Args:
        F_cent (float or np.ndarray): Frequency in cents
        F_ref (float): Reference frequency in Hz (Default value = 55.0)

    Returns:
        F (float or np.ndarray): Frequency in Hz
    """
    F = F_ref * 2 ** (F_cent / 1200)
    return F

################################################################################################

def segmentwise_search(aalap_file,aalap_F0_folder,warped_phrases_list,aalap_data,show_details = True, save_results = True, output_folder = None, penalty = 0.1):

    pitch_contour_aalap = pd.read_csv(f'{aalap_F0_folder}/{aalap_file}.csv')
    
#     y,sr = librosa.load(f'./../Aalap and Pakad/Aalap/{ref_singer}_Aalap1_{raga}.wav')
    
    aalap_segments = aalap_data.loc[aalap_data['Aalap File'] == aalap_file + ".csv"]
    
    to_write = []
    dataframe = []
    all_costs = []
    all_start_end = []
    for segment_id, i in enumerate(aalap_segments.index.values):
        start = aalap_segments.loc[i,'Start']
        end = aalap_segments.loc[i,'End']
        pitch_contour_aalap['Select'] = pitch_contour_aalap['time'].apply(lambda x: x >= start and x <= end)
        pitch_contour_ref = pitch_contour_aalap.loc[pitch_contour_aalap['Select']]
    #     display(pitch_contour_ref) 
        costs = []
        segment_df = []
        start_end = []
        for query_idx,query in enumerate(warped_phrases_list):
#             print("Processing query idx", query_idx)
            sa = SubsequenceAlignment(query, np.array(pitch_contour_ref['pitch']),penalty=penalty,use_c = True)
#             print(len(np.array(pitch_contour_ref['pitch'])))
            k = 1
            for kmatch in sa.kbest_matches_fast(k):
#                 print("Found kmatch")
                start_idx, end_idx = kmatch.segment[0], kmatch.segment[1]
                start_sec = np.array(pitch_contour_ref['time'])[start_idx]
                end_sec = np.array(pitch_contour_ref['time'])[end_idx]
                cost = round(kmatch.value,2)
#                 print('\nSegment Id:', segment_id)
#                 print('Query index',query_idx)
#                 print('Start:', start_sec)
#                 print('End:', end_sec)
                segment_df.append((start_sec, end_sec,segment_id,query_idx,cost))
                
                start_end.append((start_sec,end_sec))
        
                costs.append(cost)
#                 print('DTW Cost:', cost)
#                 if int(start_sec*sr) != int(end_sec*sr):
#                     .display(Audio(y[int(start_sec*sr):int(end_sec*sr)], rate = sr))
        if costs == []:
            print(f"Note! Match not found for Aalap file {aalap_file}, segment id: {segment_id}")
            all_costs.append(np.array([np.NaN]*6))
            all_start_end.append(np.array([np.NaN]*6))
        else:
#             print(costs)
            costs.append(f'{aalap_file}_S{segment_id}')
            start_end.append(f'{aalap_file}_S{segment_id}')
            segment_df = pd.DataFrame(segment_df, columns = ['Start','End','Segment Id','Query','Cost'])
            min_cost_query = np.argmin(np.array(costs[:-1]))
            min_cost = np.array(costs[:-1])[min_cost_query]
            if show_details:
                print("\nResults for silence delimited segment no.", segment_id, ":")
                display.display(segment_df)
                print(f'Min Cost Query for Segment {segment_id}: {min_cost_query}, Min cost: {min_cost}\n')
            if save_results:
                segment_df.to_csv(f'{output_folder}/{aalap_file}_S{segment_id}_{start}_{end}.csv')

            dataframe.append((start_sec,end_sec,segment_id,min_cost_query,min_cost))
            all_costs.append(costs)
            all_start_end.append(start_end)
        
    return pd.DataFrame(dataframe,columns=['Start','End','Segment Id','Min Cost Query','Min Cost']),np.array(all_costs), np.array(all_start_end)


def segmentwise_search_multiple(aalap_file,aalap_F0_folder,warped_phrases_list,aalap_data,show_details = True, output_folder = None,k=20, penalty = 0.1):
    
    length_thresh = 0.5
    pitch_contour_aalap = pd.read_csv(f'{aalap_F0_folder}/{aalap_file}.csv')
    aalap_segments = aalap_data.loc[aalap_data['Aalap File'] == aalap_file + ".csv"]

    to_write = []
    dataframe = []
    all_costs = [] 
    all_start_end = []
    for segment_id, i in enumerate(aalap_segments.index.values):
        start = aalap_segments.loc[i,'Start']
        end = aalap_segments.loc[i,'End']
        pitch_contour_aalap['Select'] = pitch_contour_aalap['time'].apply(lambda x: x >= start and x <= end)
        pitch_contour_ref = pitch_contour_aalap.loc[pitch_contour_aalap['Select']]
    #     display(pitch_contour_ref) 
        costs = []
        segment_df = []
        start_end = []
        for query_idx,query in enumerate(warped_phrases_list):
            sa = SubsequenceAlignment(query, np.array(pitch_contour_ref['pitch']),penalty=penalty,use_c = True)

            for kmatch in sa.kbest_matches_fast(k):
                start_idx, end_idx = kmatch.segment[0], kmatch.segment[1]
                start_sec = np.array(pitch_contour_ref['time'])[start_idx]
                end_sec = np.array(pitch_contour_ref['time'])[end_idx]
                cost = round(kmatch.value,2)
                segment_df.append((start_sec, end_sec,segment_id,query_idx,cost))
                start_end.append((start_sec,end_sec))
                costs.append(cost)
        if costs == []:
            print(f"Note! Match not found for Aalap file {aalap_file}, segment id: {segment_id}")
            all_costs.append('Empty')
            all_start_end.append('Empty')
        else:
            costs.append(f'{aalap_file}_S{segment_id}')
            start_end.append(f'{aalap_file}_S{segment_id}')
            segment_df = pd.DataFrame(segment_df, columns = ['Start','End','Segment Id','Query','Cost'])
            print("\nResults for SDS no.", segment_id, ":")
            segment_df['Length'] = segment_df['End'] - segment_df['Start']
            # print('\nLengths:',segment_df['Length'].values)
            # print(len(segment_df['Length'].values))
            valid_df = segment_df.loc[segment_df['Length'] > length_thresh]
            # display.display(valid_df)
            pair_df = []
            for query_idx,query in enumerate(warped_phrases_list):
                valid_costs_this_pair = list(valid_df.loc[valid_df['Query']==query_idx]['Cost']) 
                #list of costs for this SDS-query pair whose lengths are > length_thresh
                if valid_costs_this_pair == []:
                    pair_cost = 'Invalid'
                    # print(f'Pair cost for seg {segment_id}, query {query_idx}:', pair_cost)
                    pair_df_row = ('Invalid','Invalid',segment_id, query_idx,pair_cost,'Invalid')
                else:
                    pair_cost = min(valid_costs_this_pair)
                    pair_start = valid_df.loc[valid_df['Cost']==pair_cost]['Start'].iloc[0]
                    pair_end = valid_df.loc[valid_df['Cost']==pair_cost]['End'].iloc[0]
                    pair_start_end = (pair_start,pair_end)
                    # print(f'Pair cost for seg {segment_id}, query {query_idx}:', pair_cost, 'Timestamps:', pair_start_end)
                    pair_df_row = (pair_start,pair_end,segment_id, query_idx,pair_cost,pair_end - pair_start)
                pair_df.append(pair_df_row)
                dataframe.append(pair_df_row)
                
            pair_df = pd.DataFrame(pair_df, columns = ['Start','End','Segment Id','Query','Cost','Length']) 
            # display.display(pair_df)
            
            all_costs.append(costs)
            all_start_end.append(start_end)
            
    df2 = pd.DataFrame(dataframe,columns = ['Start','End','Segment Id','Query','Cost','Length'])
    
    df3 = df2.loc[df2['Length'] != 'Invalid']
    df4 = []
    for seg_id in df3['Segment Id'].unique():
        seg_df = df3.loc[df3['Segment Id']==seg_id]
        queries = np.array(seg_df['Query'])
        costs = np.array(seg_df['Cost'])
        i = np.argmin(costs)
        df4.append(seg_df.iloc[i].values)
    df4 = pd.DataFrame(df4, columns = df3.columns.values)
    df4 = df4.sort_values('Cost')

    cost_thresh = 2 * np.mean(df4['Cost'].iloc[0:3])
    df4['Matched'] = df4['Cost'].apply(lambda x: 'Matched' if x<=cost_thresh else 'Unmatched')

    df3['Matched'] = df3['Segment Id'].apply(lambda x: df4.loc[df4['Segment Id']==x]['Matched'].item())

    return df3, df4