import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def dater(station, window):

    """This function returns the dates corresponding to a window.
    ---------
    Arguments:
    station: The station number.
    window: The window to be converted.
    
    Returns:
    date_indices: The dates corresponding to the window."""

    # Read data
    data = pd.read_csv(f'data/labeled_{station}_smo.csv', sep=',', encoding='utf-8', parse_dates=['date'], index_col=['date'])
    data = data.iloc[:, :-2]

    # Reshape window and define mask
    window = np.array(window).reshape(-1, 6)
    mask = np.zeros(len(data), dtype=bool)
    for window_row in window:
        mask |= (data.values == window_row).all(axis=1)
    
    # Extract dates
    indices = np.where(mask)[0]
    
    date_indices = data.index[indices]
    
    return date_indices

def plotter(data, num_variables, windowed):
    
    """This function plots the original data,
    and the windowed data at all resolution levels.
    ---------
    Arguments:
    data: The data to be plotted.
    num_variables: The number of variables in the data.
    windowed: Whether the data is windowed or not.
    
    Returns:
    None"""

    variables_names = ["Ammonium", "Conductivity", "Dissolved oxygen", "pH", "Turbidity", "Water temperature"]

    if windowed == False:

        data_reshaped = data.reshape(-1, num_variables)

        # Plot each variable
        for i in range(num_variables):
            plt.plot(dater(901, data), data_reshaped[:, i], label=f'{variables_names[i]}')

        plt.xlabel('Time/Index')
        plt.ylabel('Variable Value')
        plt.legend()
        plt.show()

    if windowed == True:

        for window_index, window in enumerate(data):
            
            # Reshape the window
            window_reshaped = window.reshape(-1, num_variables)

            # Create a new figure for each window
            plt.figure(window_index)

            # Plot each variable
            for i in range(num_variables):
                plt.plot(dater(901, data), window_reshaped[:, i], label=f'{variables_names[i]}')

            plt.xlabel('Time/Index')
            plt.ylabel('Variable Value')
            plt.legend()
            plt.title(f'Window {window_index+1}')

        plt.show()

def explainer(X, model, resolution, window_to_explain):

    """This function explains the decision of a Random Forest model
    for a given window."""

    # Create an empty list to store decision path for each window and all of the decision paths
    all_decision_paths = []
    window_decision_paths =[]

    # Traverse each tree in the Random Forest to get the decision path across all tree for each window
    for window in range(len(X)):
        for tree in model.estimators_:
            tree_decision_path = tree.decision_path(X[window][np.newaxis, :]).toarray()
            window_decision_paths.append(tree_decision_path)
        all_decision_paths.append(window_decision_paths) # The first elemnt would be all decision paths of the first window
        window_decision_paths = []

    # Retrieve the decision paths of this window across all trees in the RF
    decision_paths = all_decision_paths[window_to_explain]

    # Get the indices where the window has passed through
    passed_nodes_indices = [np.where(decision_path == 1)[1] for decision_path in decision_paths]

    # Get the thresholds and feature values of the nodes in each decision tree in the Random Forest
    tree_feature_thresholds = [model.estimators_[i].tree_.threshold[e] for i, e in enumerate(passed_nodes_indices)]
    tree_feature_indices = [model.estimators_[i].tree_.feature[e] for i, e in enumerate(passed_nodes_indices)]

    # Define feature names for all resolution levels
    feature_names_high = [
                    'am-16', 'co-16', 'do-16', 'ph-16', 'wt-16', 'tu-16',
                    'am-15', 'co-15', 'do-15', 'ph-15', 'wt-15', 'tu-15',
                    'am-14', 'co-14', 'do-14', 'ph-14', 'wt-14', 'tu-14',
                    'am-13', 'co-13', 'do-13', 'ph-13', 'wt-13', 'tu-13',
                    'am-12', 'co-12', 'do-12', 'ph-12', 'wt-12', 'tu-12',
                    'am-11', 'co-11', 'do-11', 'ph-11', 'wt-11', 'tu-11',
                    'am-10', 'co-10', 'do-10', 'ph-10', 'wt-10', 'tu-10',
                    'am-9', 'co-9', 'do-9', 'ph-9', 'wt-9', 'tu-9',
                    'am-8', 'co-8', 'do-8', 'ph-8', 'wt-8', 'tu-8',
                    'am-7', 'co-7', 'do-7', 'ph-7', 'wt-7', 'tu-7',
                    'am-6', 'co-6', 'do-6', 'ph-6', 'wt-6', 'tu-6',
                    'am-5', 'co-5', 'do-5', 'ph-5', 'wt-5', 'tu-5',
                    'am-4', 'co-4', 'do-4', 'ph-4', 'wt-4', 'tu-4',
                    'am-3', 'co-3', 'do-3', 'ph-3', 'wt-3', 'tu-3',
                    'am-2', 'co-2', 'do-2', 'ph-2', 'wt-2', 'tu-2',
                    'am-1', 'co-1', 'do-1', 'ph-1', 'wt-1', 'tu-1',
                    'am+1', 'co+1', 'do+1', 'ph+1', 'wt+1', 'tu+1',
                    'am+2', 'co+2', 'do+2', 'ph+2', 'wt+2', 'tu+2',
                    'am+3', 'co+3', 'do+3', 'ph+3', 'wt+3', 'tu+3',
                    'am+4', 'co+4', 'do+4', 'ph+4', 'wt+4', 'tu+4',
                    'am+5', 'co+5', 'do+5', 'ph+5', 'wt+5', 'tu+5',
                    'am+6', 'co+6', 'do+6', 'ph+6', 'wt+6', 'tu+6',
                    'am+7', 'co+7', 'do+7', 'ph+7', 'wt+7', 'tu+7',
                    'am+8', 'co+8', 'do+8', 'ph+8', 'wt+8', 'tu+8',
                    'am+9', 'co+9', 'do+9', 'ph+9', 'wt+9', 'tu+9',
                    'am+10', 'co+10', 'do+10', 'ph+10', 'wt+10', 'tu+10',
                    'am+11', 'co+11', 'do+11', 'ph+11', 'wt+11', 'tu+11',
                    'am+12', 'co+12', 'do+12', 'ph+12', 'wt+12', 'tu+12',
                    'am+13', 'co+13', 'do+13', 'ph+13', 'wt+13', 'tu+13',
                    'am+14', 'co+14', 'do+14', 'ph+14', 'wt+14', 'tu+14',
                    'am+15', 'co+15', 'do+15', 'ph+15', 'wt+15', 'tu+15',
                    'am+16', 'co+16', 'do+16', 'ph+16', 'wt+16', 'tu+16'
                    ]

    feature_names_med = [
                    'am-8', 'co-8', 'do-8', 'ph-8', 'wt-8', 'tu-8',
                    'am-7', 'co-7', 'do-7', 'ph-7', 'wt-7', 'tu-7',
                    'am-6', 'co-6', 'do-6', 'ph-6', 'wt-6', 'tu-6',
                    'am-5', 'co-5', 'do-5', 'ph-5', 'wt-5', 'tu-5',
                    'am-4', 'co-4', 'do-4', 'ph-4', 'wt-4', 'tu-4',
                    'am-3', 'co-3', 'do-3', 'ph-3', 'wt-3', 'tu-3',
                    'am-2', 'co-2', 'do-2', 'ph-2', 'wt-2', 'tu-2',
                    'am-1', 'co-1', 'do-1', 'ph-1', 'wt-1', 'tu-1',
                    'am+1', 'co+1', 'do+1', 'ph+1', 'wt+1', 'tu+1',
                    'am+2', 'co+2', 'do+2', 'ph+2', 'wt+2', 'tu+2',
                    'am+3', 'co+3', 'do+3', 'ph+3', 'wt+3', 'tu+3',
                    'am+4', 'co+4', 'do+4', 'ph+4', 'wt+4', 'tu+4',
                    'am+5', 'co+5', 'do+5', 'ph+5', 'wt+5', 'tu+5',
                    'am+6', 'co+6', 'do+6', 'ph+6', 'wt+6', 'tu+6',
                    'am+7', 'co+7', 'do+7', 'ph+7', 'wt+7', 'tu+7',
                    'am+8', 'co+8', 'do+8', 'ph+8', 'wt+8', 'tu+8'
                    ]

    feature_names_low = [
                    'am-4', 'co-4', 'do-4', 'ph-4', 'wt-4', 'tu-4',
                    'am-3', 'co-3', 'do-3', 'ph-3', 'wt-3', 'tu-3',
                    'am-2', 'co-2', 'do-2', 'ph-2', 'wt-2', 'tu-2',
                    'am-1', 'co-1', 'do-1', 'ph-1', 'wt-1', 'tu-1',
                    'am+1', 'co+1', 'do+1', 'ph+1', 'wt+1', 'tu+1',
                    'am+2', 'co+2', 'do+2', 'ph+2', 'wt+2', 'tu+2',
                    'am+3', 'co+3', 'do+3', 'ph+3', 'wt+3', 'tu+3',
                    'am+4', 'co+4', 'do+4', 'ph+4', 'wt+4', 'tu+4'
                    ]

    # Select the resolution of the feature names
    feature_names = feature_names_high if resolution == 'high' else feature_names_med if resolution == 'med' else feature_names_low

    # The wt+7 has to be removed at the end of each element because it corresponds to the -2 index of the leaves
    subset_feature_names = []
    for i in tree_feature_indices:
        subset_feature_names.append([feature_names[j] for j in i[:-1].tolist()])

    subset_feature_thresholds = []
    for i in tree_feature_thresholds:
        subset_feature_thresholds.append(i[:-1].tolist())
    print(subset_feature_names)
    # Variable-position plot
    # Extract variable names and their positions
    variables = {}
    for sublist in subset_feature_names:
        for i, item in enumerate(sublist):
            var = item.split('-')[0].split('+')[0]
            if var not in variables:
                variables[var] = []
            variables[var].append(i)

    # Create a 2D array for the heatmap
    max_len = max([len(sublist) for sublist in subset_feature_names])
    heatmap_data = np.zeros((len(variables), max_len))

    for i, var in enumerate(variables.keys()):
        for pos in variables[var]:
            heatmap_data[i, pos] += 1

    # Create the heatmap
    heatmap_data = heatmap_data.astype(int) # Convert data to int

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, xticklabels=range(max_len), yticklabels=list(variables.keys()), cmap='viridis', annot=True, fmt="d")
    plt.xlabel('Position')
    plt.ylabel('Variable')
    plt.title(f'Variable importance window {window_to_explain}')
    plt.show()

    # Variable-threshold plot
    variables_dict = {}
    for sublist, sublist_thresholds in zip(subset_feature_names, subset_feature_thresholds):
        for var, threshold in zip(sublist, sublist_thresholds):
            var_type = var.split('-')[0].split('+')[0]  # Extract variable type
            if var_type not in variables_dict:
                variables_dict[var_type] = {}
            if var not in variables_dict[var_type]:
                variables_dict[var_type][var] = []
            variables_dict[var_type][var].append(threshold)

    # Define a color mapping for the variable types
    color_mapping = {
        'am': '#ff6961',
        'co': '#ffb347',
        'do': '#aec6cf',
        'ph': '#b39eb5',
        'tu': '#fdfd96',
        'wt': '#77dd77'
    }

    # Create a violin plot for each variable type
    for var_type, var_dict in variables_dict.items():
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame([(key, var) for key, values in var_dict.items() for var in values], columns=['Variable', 'Threshold'])
        
        # Extract the numeric part of 'Variable' for sorting
        df['SortKey'] = df['Variable'].apply(lambda x: int(x.split(var_type)[1]) if var_type in x else 0)

        # Sort the DataFrame by 'SortKey'
        df = df.sort_values('SortKey', ascending=False)

        # Drop the 'SortKey' as it's no longer needed
        df = df.drop('SortKey', axis=1)

        # Create the violin plot with the color for the current variable type
        plt.figure(figsize=(10, 8))
        sns.violinplot(x='Variable', y='Threshold', data=df, order=df['Variable'].unique(), color=color_mapping.get(var_type, 'black'))
        plt.title(f'Violin plot for {var_type} variables')
        plt.show()

def tree_plotter(model, resolution):

    """This function plots a tree of a Random Forest model.
    ---------
    Arguments:
    model: The Random Forest model to plot.
    resolution: The resolution of the model.

    Returns:
    None.
    """

    # Plot an estimator (tree) of the Random Forest model
    from sklearn.tree import export_graphviz

    # Define feature names for all resolution levels
    feature_names_high = [
                    'am-16', 'co-16', 'do-16', 'ph-16', 'wt-16', 'tu-16',
                    'am-15', 'co-15', 'do-15', 'ph-15', 'wt-15', 'tu-15',
                    'am-14', 'co-14', 'do-14', 'ph-14', 'wt-14', 'tu-14',
                    'am-13', 'co-13', 'do-13', 'ph-13', 'wt-13', 'tu-13',
                    'am-12', 'co-12', 'do-12', 'ph-12', 'wt-12', 'tu-12',
                    'am-11', 'co-11', 'do-11', 'ph-11', 'wt-11', 'tu-11',
                    'am-10', 'co-10', 'do-10', 'ph-10', 'wt-10', 'tu-10',
                    'am-9', 'co-9', 'do-9', 'ph-9', 'wt-9', 'tu-9',
                    'am-8', 'co-8', 'do-8', 'ph-8', 'wt-8', 'tu-8',
                    'am-7', 'co-7', 'do-7', 'ph-7', 'wt-7', 'tu-7',
                    'am-6', 'co-6', 'do-6', 'ph-6', 'wt-6', 'tu-6',
                    'am-5', 'co-5', 'do-5', 'ph-5', 'wt-5', 'tu-5',
                    'am-4', 'co-4', 'do-4', 'ph-4', 'wt-4', 'tu-4',
                    'am-3', 'co-3', 'do-3', 'ph-3', 'wt-3', 'tu-3',
                    'am-2', 'co-2', 'do-2', 'ph-2', 'wt-2', 'tu-2',
                    'am-1', 'co-1', 'do-1', 'ph-1', 'wt-1', 'tu-1',
                    'am+1', 'co+1', 'do+1', 'ph+1', 'wt+1', 'tu+1',
                    'am+2', 'co+2', 'do+2', 'ph+2', 'wt+2', 'tu+2',
                    'am+3', 'co+3', 'do+3', 'ph+3', 'wt+3', 'tu+3',
                    'am+4', 'co+4', 'do+4', 'ph+4', 'wt+4', 'tu+4',
                    'am+5', 'co+5', 'do+5', 'ph+5', 'wt+5', 'tu+5',
                    'am+6', 'co+6', 'do+6', 'ph+6', 'wt+6', 'tu+6',
                    'am+7', 'co+7', 'do+7', 'ph+7', 'wt+7', 'tu+7',
                    'am+8', 'co+8', 'do+8', 'ph+8', 'wt+8', 'tu+8',
                    'am+9', 'co+9', 'do+9', 'ph+9', 'wt+9', 'tu+9',
                    'am+10', 'co+10', 'do+10', 'ph+10', 'wt+10', 'tu+10',
                    'am+11', 'co+11', 'do+11', 'ph+11', 'wt+11', 'tu+11',
                    'am+12', 'co+12', 'do+12', 'ph+12', 'wt+12', 'tu+12',
                    'am+13', 'co+13', 'do+13', 'ph+13', 'wt+13', 'tu+13',
                    'am+14', 'co+14', 'do+14', 'ph+14', 'wt+14', 'tu+14',
                    'am+15', 'co+15', 'do+15', 'ph+15', 'wt+15', 'tu+15',
                    'am+16', 'co+16', 'do+16', 'ph+16', 'wt+16', 'tu+16'
                    ]

    feature_names_med = [
                    'am-8', 'co-8', 'do-8', 'ph-8', 'wt-8', 'tu-8',
                    'am-7', 'co-7', 'do-7', 'ph-7', 'wt-7', 'tu-7',
                    'am-6', 'co-6', 'do-6', 'ph-6', 'wt-6', 'tu-6',
                    'am-5', 'co-5', 'do-5', 'ph-5', 'wt-5', 'tu-5',
                    'am-4', 'co-4', 'do-4', 'ph-4', 'wt-4', 'tu-4',
                    'am-3', 'co-3', 'do-3', 'ph-3', 'wt-3', 'tu-3',
                    'am-2', 'co-2', 'do-2', 'ph-2', 'wt-2', 'tu-2',
                    'am-1', 'co-1', 'do-1', 'ph-1', 'wt-1', 'tu-1',
                    'am+1', 'co+1', 'do+1', 'ph+1', 'wt+1', 'tu+1',
                    'am+2', 'co+2', 'do+2', 'ph+2', 'wt+2', 'tu+2',
                    'am+3', 'co+3', 'do+3', 'ph+3', 'wt+3', 'tu+3',
                    'am+4', 'co+4', 'do+4', 'ph+4', 'wt+4', 'tu+4',
                    'am+5', 'co+5', 'do+5', 'ph+5', 'wt+5', 'tu+5',
                    'am+6', 'co+6', 'do+6', 'ph+6', 'wt+6', 'tu+6',
                    'am+7', 'co+7', 'do+7', 'ph+7', 'wt+7', 'tu+7',
                    'am+8', 'co+8', 'do+8', 'ph+8', 'wt+8', 'tu+8'
                    ]

    feature_names_low = [
                    'am-4', 'co-4', 'do-4', 'ph-4', 'wt-4', 'tu-4',
                    'am-3', 'co-3', 'do-3', 'ph-3', 'wt-3', 'tu-3',
                    'am-2', 'co-2', 'do-2', 'ph-2', 'wt-2', 'tu-2',
                    'am-1', 'co-1', 'do-1', 'ph-1', 'wt-1', 'tu-1',
                    'am+1', 'co+1', 'do+1', 'ph+1', 'wt+1', 'tu+1',
                    'am+2', 'co+2', 'do+2', 'ph+2', 'wt+2', 'tu+2',
                    'am+3', 'co+3', 'do+3', 'ph+3', 'wt+3', 'tu+3',
                    'am+4', 'co+4', 'do+4', 'ph+4', 'wt+4', 'tu+4'
                    ]

    # Select the resolution of the feature names
    feature_names = feature_names_high if resolution == 'high' else feature_names_med if resolution == 'med' else feature_names_low

    # Plot the tree ([0] for the first tree in the RF)
    export_graphviz(model.estimators_[0], out_file='tree.dot',
                    feature_names=feature_names, # Number of data points in a window
                    rounded=True,
                    proportion=False,
                    precision=2,
                    filled=True)

    # Convert to png using system command (requires Graphviz)
    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree_0.png', '-Gdpi=600'])