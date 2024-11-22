
import pickle
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import warnings
warnings.filterwarnings('ignore')

from sklearn import tree

from utils import dater, event_plotter, depths, attention, multivariate_attention, thresholds, distances, kl_divergence
from utils import attention_plotter, multivariate_attention_plotter, threshold_plotter, distance_plotter, kl_plotter, tree_plotter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""This file implements the Visual Paths algorithm"""

def get_starts_ends(num_win_high, num_win_med, num_win_low):

    """Extacts the start and end window index for each resolution and save it in a 2D list"""

    starts_ends = []
    for event_number in range(len(num_win_high)):

        event_start_high = sum(num_win_high[:event_number]) + event_number
        event_end_high = sum(num_win_high[:event_number + 1]) + 1 + event_number

        event_start_med = sum(num_win_med[:event_number]) + event_number
        event_end_med = sum(num_win_med[:event_number + 1]) + 1 + event_number

        event_start_low = sum(num_win_low[:event_number]) + event_number
        event_end_low = sum(num_win_low[:event_number + 1]) + 1 + event_number

        starts_ends.append([[event_start_high, event_end_high], [event_start_med, event_end_med], [event_start_low, event_end_low]])

    return starts_ends

def majority_vote(high, med, low):
    
    vote_high = sum(high) / len(high)
    vote_med = sum(med) / len(med)
    vote_low = sum(low) / len(low)

    if (1/3 * vote_high + 1/3 * vote_med + 1/3 * vote_low) >= 0.9:
        return 1
    elif (1/3 * vote_high + 1/3 * vote_med + 1/3 * vote_low) <= 0.1:
        return 0

if __name__ == '__main__':

    station = 901

    window_size_high, window_size_med, window_size_low = 32, 16, 8

    # Load models
    iteration = 9

    filename = f'models/rf_model_high_{iteration}.sav'
    model_high = pickle.load(open(filename, 'rb'))

    filename = f'models/rf_model_med_{iteration}.sav'
    model_med = pickle.load(open(filename, 'rb'))

    filename = f'models/rf_model_low_{iteration}.sav'
    model_low = pickle.load(open(filename, 'rb'))

    # # Plot a tree
    # tree_plotter(model_high, 'high', tree_number=0)

    # Load the anomalies data
    file_anomalies = open(f'pickels/anomaly_data_pred.pkl', 'rb')
    anomalies_windows = pickle.load(file_anomalies)
    file_anomalies.close()
    
    # Load the background data
    file_background = open(f'pickels/background_data_pred.pkl', 'rb')
    background_windows = pickle.load(file_background)
    file_background.close()

    # Get windowed data and rename it to X_anomalies and X_background
    X_anomalies = anomalies_windows[0]
    X_background = background_windows[0]

    # Get the lengths of the windows
    lengths_anomalies = anomalies_windows[-1]
    lengths_background = background_windows[-1]

    # Get the number of windows for each resolution, anomalies and background
    number_windows_high_anomalies = [i - window_size_high + 1 for i in lengths_anomalies]
    number_windows_med_anomalies = [i - window_size_med + 1 for i in lengths_anomalies]
    number_windows_low_anomalies = [i - window_size_low + 1 for i in lengths_anomalies]

    number_windows_high_background = [i - window_size_high + 1 for i in lengths_background]
    number_windows_med_background = [i - window_size_med + 1 for i in lengths_background]
    number_windows_low_background = [i - window_size_low + 1 for i in lengths_background]

    # Find the start and end window index for each resolution and save it in a 2D list
    starts_ends_anomalies = get_starts_ends(number_windows_high_anomalies, number_windows_med_anomalies, number_windows_low_anomalies)
    starts_ends_background = get_starts_ends(number_windows_high_background, number_windows_med_background, number_windows_low_background)
    
    # Read background results
    y_hats_high = np.load('preds/y_hats_high.npy', allow_pickle=False, fix_imports=False)
    y_hats_med = np.load('preds/y_hats_med.npy', allow_pickle=False, fix_imports=False)
    y_hats_low = np.load('preds/y_hats_low.npy', allow_pickle=False, fix_imports=False)
        
    # Get multiresolution votes for each background event
    votes = [majority_vote(y_hats_high[i[0][0]:i[0][1]], y_hats_med[i[1][0]:i[1][1]], y_hats_low[i[2][0]:i[2][1]]) for i in starts_ends_background]

    # Ge the index of the true anomalies and background events which the background set
    background_anomalies_events = np.where(np.array(votes) == 1)[0]
    background_background_events = np.where(np.array(votes) == 0)[0]
    print('Background anomalies:', background_anomalies_events)
    print('True background:', background_background_events)

    # Get the number of actual anomalous events
    anomalies_events = range(len(number_windows_high_anomalies))

    # Initialize the list to store the multivariate attention maps to get the Kullback-Leibler divergence among them
    attention_multivariate_maps = [[], [], []]
    #%% Get the results for the labeled anomalies
    for event_number_main in anomalies_events:
        if event_number_main == 1:
            logging.info('Processing anomaly event number %d', event_number_main)

            # Update data type, starts_ends and X
            data_type = 'anomalies'
            starts_ends = starts_ends_anomalies
            X = X_anomalies

            # Plot the event
            event_plotter(starts_ends, X, event_number_main, station=station, type=data_type[:2])
            logging.info('Finished event plot')

            # Get the depths of the variables
            variables_depths, variables_thresholds, variables_distances, max_depth = depths(starts_ends, X, models=[model_high, model_med, model_low], event_number=event_number_main)

            # Get the attention maps
            attention_am, attention_co, attention_do, attention_ph, attention_tu, attention_wt = attention(variables_depths, max_depth)

            # Get the multivariate attention map
            attention_multivariate = multivariate_attention(attention_am, attention_co, attention_do, attention_ph, attention_tu, attention_wt)

            # Get the threshold maps
            threshold_am, threshold_co, threshold_do, threshold_ph, threshold_tu, threshold_wt = thresholds(variables_thresholds)

            # Get the distance maps
            distance_am, distance_co, distance_do, distance_ph, distance_tu, distance_wt = distances(variables_distances)

            # Plot the attention
            attention_maps = [attention_am, attention_co, attention_do, attention_ph, attention_tu, attention_wt]
            attention_plotter(attention_maps, event_number=event_number_main, station=station, type=data_type[:2])

            # Plot the multivariate attention
            multivariate_attention_plotter(attention_multivariate, event_number=event_number_main, station=station, type=data_type[:2])
            logging.info('Finished attention maps')

            # Plot the threshold
            threshold_maps = [threshold_am, threshold_co, threshold_do, threshold_ph, threshold_tu, threshold_wt]
            threshold_plotter(threshold_maps, event_number=event_number_main, station=station, type=data_type[:2])
            logging.info('Finished threshold maps')

            # Plot the distance
            distance_maps = [distance_am, distance_co, distance_do, distance_ph, distance_tu, distance_wt]
            distance_plotter(distance_maps, event_number=event_number_main, station=station, type=data_type[:2])
            logging.info('Finished distance maps')

            # Store the multivariate attention map
            attention_multivariate_maps[0].append(attention_multivariate)
    
    #%% Get the results for the detected anomalies
    for event_number_main in background_anomalies_events:
        if event_number_main == 42:
            logging.info('Processing detected anomaly event number %d', event_number_main)
            
            # Update data type, starts_ends and X
            data_type = 'background'
            starts_ends = starts_ends_background
            X = X_background

            # Plot the event
            event_plotter(starts_ends, X, event_number_main, station=station, type=data_type[:2])
            logging.info('Finished event plot')

            # Get the depths of the variables
            variables_depths, variables_thresholds, variables_distances, max_depth = depths(starts_ends, X, models=[model_high, model_med, model_low], event_number=event_number_main)

            # Get the attention maps
            attention_am, attention_co, attention_do, attention_ph, attention_tu, attention_wt = attention(variables_depths, max_depth)
            
            # Get the multivariate attention map
            attention_multivariate = multivariate_attention(attention_am, attention_co, attention_do, attention_ph, attention_tu, attention_wt)

            # Get the threshold maps
            threshold_am, threshold_co, threshold_do, threshold_ph, threshold_tu, threshold_wt = thresholds(variables_thresholds)

            # Get the distance maps
            distance_am, distance_co, distance_do, distance_ph, distance_tu, distance_wt = distances(variables_distances)

            # Plot the attention
            attention_maps = [attention_am, attention_co, attention_do, attention_ph, attention_tu, attention_wt]
            attention_plotter(attention_maps, event_number=event_number_main, station=station, type=data_type[:2])

            # Plot the multivariate attention
            multivariate_attention_plotter(attention_multivariate, event_number=event_number_main, station=station, type=data_type[:2])
            logging.info('Finished attention maps')

            # Plot the threshold
            threshold_maps = [threshold_am, threshold_co, threshold_do, threshold_ph, threshold_tu, threshold_wt]
            threshold_plotter(threshold_maps, event_number=event_number_main, station=station, type=data_type[:2])
            logging.info('Finished threshold maps')

            # Plot the distance
            distance_maps = [distance_am, distance_co, distance_do, distance_ph, distance_tu, distance_wt]
            distance_plotter(distance_maps, event_number=event_number_main, station=station, type=data_type[:2])
            logging.info('Finished distance maps')

            # Store the multivariate attention map
            attention_multivariate_maps[1].append(attention_multivariate)

    #%% Get the results for the true background events
    for event_number_main in background_background_events:
        if event_number_main == 25:
            logging.info('Processing true background event number %d', event_number_main)

            # Update data type, starts_ends and X
            data_type = 'background'
            starts_ends = starts_ends_background
            X = X_background

            # Plot the event
            event_plotter(starts_ends, X, event_number_main, station=station, type=data_type[:2])
            logging.info('Finished event plot')

            # Get the depths of the variables
            variables_depths, variables_thresholds, variables_distances, max_depth = depths(starts_ends, X, models=[model_high, model_med, model_low], event_number=event_number_main)

            # Get the attention maps
            attention_am, attention_co, attention_do, attention_ph, attention_tu, attention_wt = attention(variables_depths, max_depth)
            
            # Get the multivariate attention map
            attention_multivariate = multivariate_attention(attention_am, attention_co, attention_do, attention_ph, attention_tu, attention_wt)

            # Get the threshold maps
            threshold_am, threshold_co, threshold_do, threshold_ph, threshold_tu, threshold_wt = thresholds(variables_thresholds)

            # Get the distance map, distance_co, distance_do, distance_ph, distance_tu, distance_wt = distances(variables_distances)
            distance_am, distance_co, distance_do, distance_ph, distance_tu, distance_wt = distances(variables_distances)

            # Plot the attention
            attention_maps = [attention_am, attention_co, attention_do, attention_ph, attention_tu, attention_wt]
            attention_plotter(attention_maps, event_number=event_number_main, station=station, type=data_type[:2])

            # Plot the multivariate attention
            multivariate_attention_plotter(attention_multivariate, event_number=event_number_main, station=station, type=data_type[:2])
            logging.info('Finished attention maps')

            # Plot the threshold
            threshold_maps = [threshold_am, threshold_co, threshold_do, threshold_ph, threshold_tu, threshold_wt]
            threshold_plotter(threshold_maps, event_number=event_number_main, station=station, type=data_type[:2])
            logging.info('Finished threshold maps')

            # Plot the distance
            distance_maps = [distance_am, distance_co, distance_do, distance_ph, distance_tu, distance_wt]
            distance_plotter(distance_maps, event_number=event_number_main, station=station, type=data_type[:2])
            logging.info('Finished distance maps')

            # Store the multivariate attention map
            attention_multivariate_maps[2].append(attention_multivariate)

    # # Save the attention maps. This wont work when dealing with all samples, because there are different number of anomalies, detected anomalies and true background events
    # np.save(f'results/attention_multivariate_maps_{station}.npy', attention_multivariate_maps)

    # # Load the attention maps
    # # attention_multivariate_maps = np.load(f'results/attention_multivariate_maps_{station}.npy', allow_pickle=True, fix_imports=False).tolist()

    # #%% Get the Kullback-Leibler divergence between each anomaly, detected anomaly and true background with every other event
    # kl_distances = [[], [], []]
    # # TODO: the following code can be optimized
    # # Compare the labeled anomalies
    # for event_number, event_attention in zip(anomalies_events, attention_multivariate_maps[0]):

    #     data_type = 'anomalies'

    #     # Compare with the labeled anomalies
    #     for compare_event_number, compare_event_attention in enumerate(attention_multivariate_maps[0]):
    #         kl_distance = kl_divergence(event_attention, compare_event_attention)
            
    #         # Append to the corresponding list based on the group being compared to
    #         kl_distances[0].append(kl_distance)
        
    #     # Compare with the detected anomalies
    #     for compare_event_number, compare_event_attention in enumerate(attention_multivariate_maps[1]):
    #         kl_distance = kl_divergence(event_attention, compare_event_attention)
            
    #         # Append to the corresponding list based on the group being compared to
    #         kl_distances[1].append(kl_distance)
        
    #     # Compare with the true background
    #     for compare_event_number, compare_event_attention in enumerate(attention_multivariate_maps[2]):
    #         kl_distance = kl_divergence(event_attention, compare_event_attention)
            
    #         # Append to the corresponding list based on the group being compared to
    #         kl_distances[2].append(kl_distance)

    #     # Plot the KL divergences
    #     kl_plotter(kl_distances, event_number, station, data_type[:2])

    #     # Reset the list
    #     kl_distances = [[], [], []]
    # logging.info('Finished kl distances with labeled anomalies')
    
    # # Compare the detected anomalies
    # for event_number, event_attention in zip(background_anomalies_events, attention_multivariate_maps[1]):
        
    #     data_type = 'background'

    #     # Compare with the labeled anomalies
    #     for compare_event_number, compare_event_attention in enumerate(attention_multivariate_maps[0]):
    #         kl_distance = kl_divergence(event_attention, compare_event_attention)
            
    #         # Append to the corresponding list based on the group being compared to
    #         kl_distances[0].append(kl_distance)
        
    #     # Compare with the detected anomalies
    #     for compare_event_number, compare_event_attention in enumerate(attention_multivariate_maps[1]):
    #         kl_distance = kl_divergence(event_attention, compare_event_attention)
            
    #         # Append to the corresponding list based on the group being compared to
    #         kl_distances[1].append(kl_distance)
        
    #     # Compare with the true background
    #     for compare_event_number, compare_event_attention in enumerate(attention_multivariate_maps[2]):
    #         kl_distance = kl_divergence(event_attention, compare_event_attention)
            
    #         # Append to the corresponding list based on the group being compared to
    #         kl_distances[2].append(kl_distance)

    #     # Plot the KL divergences
    #     kl_plotter(kl_distances, event_number, station, data_type[:2])

    #     # Reset the list
    #     kl_distances = [[], [], []]
    # logging.info('Finished kl distances with detected anomalies')
        
    # # Compare the true background
    # for event_number, event_attention in zip(background_background_events, attention_multivariate_maps[2]):
            
    #     data_type = 'background'

    #     # Compare with the labeled anomalies
    #     for compare_event_number, compare_event_attention in enumerate(attention_multivariate_maps[0]):
    #         kl_distance = kl_divergence(event_attention, compare_event_attention)
            
    #         # Append to the corresponding list based on the group being compared to
    #         kl_distances[0].append(kl_distance)
        
    #     # Compare with the detected anomalies
    #     for compare_event_number, compare_event_attention in enumerate(attention_multivariate_maps[1]):
    #         kl_distance = kl_divergence(event_attention, compare_event_attention)
            
    #         # Append to the corresponding list based on the group being compared to
    #         kl_distances[1].append(kl_distance)
        
    #     # Compare with the true background
    #     for compare_event_number, compare_event_attention in enumerate(attention_multivariate_maps[2]):
    #         kl_distance = kl_divergence(event_attention, compare_event_attention)
            
    #         # Append to the corresponding list based on the group being compared to
    #         kl_distances[2].append(kl_distance)

    #     # Plot the KL divergences
    #     kl_plotter(kl_distances, event_number, station, data_type[:2])

    #     # Reset the list
    #     kl_distances = [[], [], []]

    # logging.info('Finished kl distances with background')
