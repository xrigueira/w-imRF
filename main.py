import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from tictoc import tictoc

"""This file contains the main class imRF which implements
iterative multiresolution Random Forest."""

class imRF():
    
    def __init__(self, station, trim_percentage, ratio, num_variables, window_size, stride, seed) -> None:
        
        self.station = station
        self.trim_percentage = trim_percentage
        self.ratio = ratio
        self.num_variables = num_variables
        self.window_size = window_size
        self.stride = stride
        self.seed = seed
    
    def windower(self, data):
        
        """
        Takes a 2D array with multivariate time series data
        and creates multiresolution sliding windows. The size
        of the slinding windows gets halved each time. The
        resulting windows store different variables in a 
        consecutive manner. E.g. [first 6 variables, next 6 
        variables, and so on].
        ----------
        Arguments:
        data (pickle): file with the time-series data to turn 
        into windows.
        num_variables (int): the number of variables in the data.
        window_size (int): the size of the windows.
        stride (int): the stride of the windows.
        
        Returns:
        windows (list): time series data grouped in windows"""
        
        windows = []
        if self.window_size > 4: # This way the maximum window would be 8 data points
            
            for i in data:
                
                # Get the number of windows
                num_windows = (len(i) - self.window_size * self.num_variables) // (self.stride * self.num_variables) + 1
                
                # Create the windows
                for j in range(0, num_windows, self.stride):
                    window = i[j * self.num_variables: (j * self.num_variables) + (self.window_size * self.num_variables)]
                    windows.append(window)
            
            self.window_size = self.window_size // 2 # Halve window size
            
            return [windows] + self.windower(data)  # Recursive call
        
        else:
            
            # Restore the window size after recursion
            self.window_size = window_size
            return []
    
    def majority_vote(self, *args):
        total = sum(args)
        return total / len(args)
    
    def anomalies(self):
        
        """Extracts the anomalies from the database and
        saves them to a pickle file.
        ----------
        Arguments:
        self.
        
        Stores:
        anomaly_data (pickle): file with the multivariate data.
        from each anomaly.
        
        Returns:
        trimmed_anomalies_indexes (list): start and end indexes of the extracted
        anomaly data.
        """
        
        # Load the data
        data = pd.read_csv(f'data/labeled_{self.station}_smo.csv', sep=',', encoding='utf-8', parse_dates=['date'])
        
        # Filter the data to select only rows where the label column has a value of 1
        data_anomalies = data[data["label"] == 1]
        
        # Create a new column with the difference between consecutive dates
        date_diff = (data_anomalies['date'] - data_anomalies['date'].shift()).fillna(pd.Timedelta(minutes=15))

        # Create groups of consecutive dates
        date_group = (date_diff != pd.Timedelta(minutes=15)).cumsum()

        # Get the starting and ending indexes of each group of consecutive dates
        grouped = data.groupby(date_group)
        consecutive_dates_indexes = [(group.index[0], group.index[-1]) for _, group in grouped]
        
        # Trim the start and end of the anomalies to remove the onset and the offset
        trimmed_anomalies_indexes = []
        for start, end in consecutive_dates_indexes:
            anomaly_length = end - start
            trim_amount = int(anomaly_length * self.trim_percentage / 100)
            trimmed_start = start + trim_amount
            trimmed_end = end - trim_amount
            trimmed_anomalies_indexes.append((trimmed_start, trimmed_end))
        
        # Extract the data
        anomaly_data = []
        for start, end in trimmed_anomalies_indexes:
            subset_rows = data.iloc[start:end + 1, 1:-2].values.flatten()  # Extract rows within the subset
            anomaly_data.append(subset_rows)

        # Group the data in windows before saving
        anomaly_data = self.windower(anomaly_data)
        
        # Save anomaly_data to disk as pickle object
        with open('pickels/anomaly_data_0.pkl', 'wb') as file:
            pickle.dump(anomaly_data, file)
        
        return trimmed_anomalies_indexes
    
    def init_background(self, anomalies_indexes):
        
        """Creates the initial background file by extracting
        'ratio' times more non anomalous data than the anomaly method.
        The data is saved to a pickle file.
        -----------
        Arguments:
        self.
        anomalies_indexes (list): start and end indexes of the extracted
        anomaly data.
        
        Saves:
        background_data (pickle): file with 'ratio' times more 
        nonanomalous data, also know as background, compared to the
        total legth of the anomalies in the dataset.
        
        Returns:
        background_indexes (list): start and end indexes of the extracted
        background data.
        """
        
        # Define random seed
        random.seed(self.seed)
        
        # Load the DataFrame from your dataset
        data = pd.read_csv(f'data/labeled_{self.station}_smo.csv', sep=',', encoding='utf-8', parse_dates=['date'])
        
        # Filter the data to select only rows where the label column has a value of 0
        data_background = data[data["label"] == 0]
        
        # Filter the dataset to include only days that meet the ammonium level the condition
        mean_ammonium = np.mean(data_background.ammonium_901)
        data_background = data_background.groupby(data_background['date'].dt.date).filter(lambda x: x[f'ammonium_{self.station}'].max() <= mean_ammonium)
        
        # Extract the length of the anomalies
        len_anomalies = [end - start for start, end in anomalies_indexes]
        
        # Define background data indexes
        background_indexes = []
        for anomaly_length in len_anomalies:
            if anomaly_length != 0:
                start = random.randint(0, len(data_background) - 1)
                end = start + (anomaly_length * self.ratio)
                background_indexes.append((start, end))
        
        # Extract the data
        background_data = []
        for start, end in background_indexes:
            
            subset_rows = data_background.iloc[start:end + 1, 1:-2].values.flatten() # Extarct rows withing the subset
            background_data.append(subset_rows)
        
        # Group data into windows before saving
        background_data = self.windower(background_data)
        
        # Save anomalies_data to disk as numpy object
        with open(f'pickels/background_data_0.pkl', 'wb') as file:
            pickle.dump(background_data, file)
            
        return background_indexes
    
    def background(self, anomalies_indexes, background_indexes, iteration):
        
        """Creates the background file for each iteration by extracting
        'ratio' times more non anomalous data than the anomaly method and
        adding that that to the previous background file. The data is saved
        to a pickle file. It makes sure that the new non anomalous data 
        extracted has not been selected before.
        ----------
        Arguments:
        self.
        anomalies_indexes (list): start and end indexes of the extracted
        anomaly data.
        background_indexes (list): start and end indexes of the previously
        extracted background data.
        iteration (int): the current iteration number.
        
        Saves:
        background_data_i (pickle): file with 'ratio' times more 
        nonanomalous data, compared to the total legth of the 
        anomalies in the dataset.
        
        Returns:
        background_indexes (list): updated start and end indexes of the 
        extracted background data.
        """
        
        # Define random seed
        random.seed(self.seed)
    
        # Load the DataFrame from your dataset
        data = pd.read_csv(f'data/labeled_{self.station}_smo.csv', sep=',', encoding='utf-8', parse_dates=['date'])
        
        # Filter the data to select only rows where the label column has a value of 0
        data_background = data[data["label"] == 0]
        
        # Filter the dataset to include only days that meet the ammonium level the condition
        mean_ammonium = np.mean(data_background.ammonium_901)
        data_background = data_background.groupby(data_background['date'].dt.date).filter(lambda x: x[f'ammonium_{self.station}'].max() <= mean_ammonium)
        
        # Extract the length of the anomalies
        len_anomalies = [end - start for start, end in anomalies_indexes]

        # Define new background data indexes
        new_background_indexes = []
        for anomaly_length in len_anomalies:
            if anomaly_length != 0:
                new_start = random.randint(0, len(data_background) - 1)
                new_end = new_start + (anomaly_length * self.ratio)
                
                # Check for overlap
                overlaps = any(start <= new_end and end >= new_start for start, end in background_indexes)
                
                # If there is an overlap, generate a new index
                max_retries = 10  # Set a maximum number of retries
                retry_count = 0
                while overlaps:
                    new_start = random.randint(0, len(data_background) - 1)
                    new_end = new_start + (anomaly_length * self.ratio)
                    overlaps = any(start <= new_end and end >= new_start for start, end in background_indexes)
                    retry_count += 1
                
                    if retry_count == max_retries:
                        break
                
                # Append the nonoverlaping indexes to the new list and the old one
                new_background_indexes.append((new_start, new_end))
                background_indexes.append((new_start, new_end))
        
        # Extract the data
        background_data = []
        for start, end in new_background_indexes:
            
            subset_rows = data_background.iloc[start:end + 1, 1:-2].values.flatten() # Extarct rows withing the subset
            background_data.append(subset_rows)
        
        # Group data into windows before saving
        background_data = self.windower(background_data)
        
        # Save anomalies_data to disk as pickle object
        with open(f'pickels/background_data_{iteration}.pkl', 'wb') as file:
            pickle.dump(background_data, file)
            
        return background_indexes
    
    @tictoc
    def init_RandomForest(self):
        
        """Initiates a set of three Random Forest classifiers
        in the first iteration. The models gets trained different
        window sizes on the first batch of anomalies and background 
        data extrated previously.
        ----------
        Arguments:
        self.
        
        Saves:
        rf_model_0 (sav): file with the model saved to disk.
        
        Returns:
        """
        
        # Read the windowed anomalous data
        file_anomalies = open('pickels/anomaly_data_0.pkl', 'rb')
        anomalies_windows = pickle.load(file_anomalies)
        file_anomalies.close()

        # Read the windowed background data
        file_background = open('pickels/background_data_0.pkl', 'rb')
        background_windows = pickle.load(file_background)
        file_background.close()
        
        # Generate labels for each window
        anomalies_labels = []
        for i in range(len(anomalies_windows)):
            anomalies_labels.append(np.array([1 for j in anomalies_windows[i]]))
        
        background_labels = []
        for i in range(len(background_windows)):
            background_labels.append(np.array([0 for j in background_windows[i]]))
        
        # Concatenate array
        X = []
        for i in range(len(anomalies_windows)):
            X.append(np.concatenate((anomalies_windows[i], background_windows[i])))
        
        y = []
        for i in range(len(anomalies_windows)):
            y.append(np.concatenate((anomalies_labels[i], background_labels[i])))
        
        # Shuffle data
        randomized = []
        for i in range(len(anomalies_windows)):
            combined = np.column_stack((X[i], y[i]))
            np.random.seed(self.seed)
            np.random.shuffle(combined)
            randomized.append(combined)
            
        # Split the shuffled array back into data and labels
        for i in range(len(anomalies_windows)):    
            X[i], y[i] = randomized[i][:, :-1], randomized[i][:, -1]
        
        # Train the Random Forest classifiers
        model_high = RandomForestClassifier(random_state=self.seed)
        model_med = RandomForestClassifier(random_state=self.seed)
        model_low = RandomForestClassifier(random_state=self.seed)
        
        # Split the shuffled data into the training and testing set
        X_train, y_train, X_test, y_test = [], [], [], []
        for i in range(len(anomalies_windows)):
            X_train.append(X[i][:int(len(X[i]) * 0.75)])
            y_train.append(y[i][:int(len(X[i]) * 0.75)])
            X_test.append(X[i][int(len(X[i]) * 0.75):])
            y_test.append(y[i][int(len(X[i]) * 0.75):])

        # Fit the model to the training data
        model_high.fit(X_train[0], y_train[0]) # Long length data windows
        model_med.fit(X_train[1], y_train[1]) # Medium legth data windows
        model_low.fit(X_train[2], y_train[2]) # Short length data windows

        from sklearn.metrics import confusion_matrix as cm
        confusion_matrix_high = cm(y_test[0], model_high.predict(X_test[0]))
        print(confusion_matrix_high)
        confusion_matrix_med = cm(y_test[1], model_med.predict(X_test[1]))
        print(confusion_matrix_med)
        confusion_matrix_low = cm(y_test[2], model_low.predict(X_test[2]))
        print(confusion_matrix_low)
        
        # Get the number of rows labeled as anomalies in y_test
        num_anomalies = len([i for i in y_test[0] if i==1])
        print('Number of anomalies in test set:', num_anomalies)
        num_anomalies = len([i for i in y_test[1] if i==1])
        print('Number of anomalies in test set:', num_anomalies)
        num_anomalies = len([i for i in y_test[2] if i==1])
        print('Number of anomalies in test set:', num_anomalies)
        
        # Save the model to disk
        filename = 'models/rf_model_high_0.sav'
        pickle.dump(model_high, open(filename, 'wb'))
        filename = 'models/rf_model_med_0.sav'
        pickle.dump(model_med, open(filename, 'wb'))
        filename = 'models/rf_model_low_0.sav'
        pickle.dump(model_low, open(filename, 'wb'))
    
    @tictoc
    def RandomForest(self, iteration):
        
        """Updates the Random Forest models on each iterations.
        The older models performs prediction on new background data.
        Those windows classified as anomalies get added to previous
        anomaly data and those which are background get included in the
        previous background data. The older models gets retrained and
        saved as new version.
        ----------
        Arguments:
        self.
        iteration (int): the current iteration number.
        
        Saves:
        rf_model_{iteration} (sav): with the updated model saved to disk.
        
        Returns:
        """
        
        # Read the current windowed background
        file_background = open(f'pickels/background_data_{iteration}.pkl', 'rb')
        background_windows = pickle.load(file_background)
        file_background.close()
        
        # Variable name change to follow best practives in ML
        X = background_windows

        # # AVOID for now. Shuffle the data and variable name change to follow best practives in ML
        # X = []
        # for i in range(len(background_windows)):
        #     np.random.seed(self.seed)
        #     np.random.shuffle(background_windows[i])
        #     X.append(background_windows[i])
        
        # Load the previous models
        filename = f'models/rf_model_high_{iteration - 1}.sav'
        loaded_model_high = pickle.load(open(filename, 'rb'))
        filename = f'models/rf_model_med_{iteration - 1}.sav'
        loaded_model_med = pickle.load(open(filename, 'rb'))
        filename = f'models/rf_model_low_{iteration - 1}.sav'
        loaded_model_low = pickle.load(open(filename, 'rb'))
        
        # Get the results from each tree
        trees_high = loaded_model_high.estimators_
        trees_med = loaded_model_med.estimators_
        trees_low = loaded_model_low.estimators_

        # Continue here adapting the code to having three models
        tree_classifications_high = [tree.predict(X[0]) for tree in trees_high]
        tree_classifications_med = [tree.predict(X[1]) for tree in trees_med]
        tree_classifications_low = [tree.predict(X[2]) for tree in trees_low]

        # Get the average score for each window across all estimators
        score_Xs_high = np.mean(tree_classifications_high, axis=0)
        score_Xs_med = np.mean(tree_classifications_med, axis=0)
        score_Xs_low = np.mean(tree_classifications_low, axis=0)
        
        variables = [(score_Xs_high, 'score_Xs_high'), (score_Xs_med, 'score_Xs_med'), (score_Xs_low, 'score_Xs_low')]

        for variable, name in variables:
            plt.plot(variable)
            # plt.show()
            plt.savefig(f'images/{name}_{iteration}.png', dpi=300)
        plt.close()
        
        # Get the indexes of those windows considered anomalies or background
        stride = 1
        num_variables = 6
        med_subwindow_span = len(X[1][0]) // (num_variables * stride)
        low_subwindow_span = (len(X[0][0])- len(X[2][0])) // (num_variables * stride)

        index_high = 0
        start_index_med, end_index_med = 0, med_subwindow_span + 1
        start_index_low, end_index_low = 0, low_subwindow_span + 1

        indexes_anomalies_windows_high, indexes_background_windows_high = [], []
        indexes_anomalies_windows_med, indexes_background_windows_med = [], []
        indexes_anomalies_windows_low, indexes_background_windows_low = [], []
        for i in range(len(score_Xs_high)):
    
            scores_high = score_Xs_high[index_high]
            scores_med = score_Xs_med[start_index_med:end_index_med]
            scores_low = score_Xs_low[start_index_low:end_index_low]
            
            # Combine the float result with the majority voting of the lists
            multiresolution_vote = self.majority_vote(scores_high, *scores_med, *scores_low)
        
            if multiresolution_vote >= 0.51:
                indexes_anomalies_windows_high.append(index_high)
                indexes_anomalies_windows_med.append((start_index_med, end_index_med))
                indexes_anomalies_windows_low.append((start_index_low, end_index_low))
            elif multiresolution_vote <= 0.10:
                indexes_background_windows_high.append(index_high)
                indexes_background_windows_med.append((start_index_med, end_index_med))
                indexes_background_windows_low.append((start_index_low, end_index_low))
            
            # Update the index values
            index_high = index_high + stride
            start_index_med, end_index_med = start_index_med + stride, end_index_med + stride
            start_index_low, end_index_low = start_index_low + stride, end_index_low + stride
        
        # Extract those new anomaly and background windows
        add_anomalies_windows_high = [X[0][i] for i in indexes_anomalies_windows_high]
        add_background_windows_high = [X[0][i] for i in indexes_background_windows_high]
        add_anomalies_windows_med = []
        [add_anomalies_windows_med.extend(X[1][start:end]) for start, end in indexes_anomalies_windows_med]
        add_background_windows_med = []
        [add_background_windows_med.extend(X[1][start:end]) for start, end in indexes_background_windows_med]
        add_anomalies_windows_low = []
        [add_anomalies_windows_low.extend(X[2][start:end]) for start, end in indexes_anomalies_windows_low]
        add_background_windows_low = []
        [add_background_windows_low.extend(X[2][start:end]) for start, end in indexes_background_windows_low]
        # print(f'Percentage of anomalies {round(len(add_anomalies_windows) / len(background_windows) * 100, 2)}%')

        # Read the previous windowed anomalous data
        file_anomalies = open(f'pickels/anomaly_data_{iteration - 1}.pkl', 'rb')
        prev_anomalies_windows = pickle.load(file_anomalies)
        file_anomalies.close()

        # Read the previous windows background
        file_background = open(f'pickels/background_data_{iteration - 1}.pkl', 'rb')
        prev_background_windows = pickle.load(file_background)
        file_background.close()
        
        # Conactenate new data with old data
        anomalies_windows = [np.vstack((prev_anomalies_windows[0], add_anomalies_windows_high)),
                            np.vstack((prev_anomalies_windows[1], add_anomalies_windows_med)),
                            np.vstack((prev_anomalies_windows[2], add_anomalies_windows_low))]
        
        background_windows = [np.vstack((prev_background_windows[0], add_background_windows_high)),
                            np.vstack((prev_background_windows[1], add_background_windows_med)),
                            np.vstack((prev_background_windows[2], add_background_windows_low))]

        # Save anomalies_data to disk as pickle object
        with open(f'pickels/anomaly_data_{iteration}.pkl', 'wb') as file:
            pickle.dump(anomalies_windows, file)
        
        # Save background data as a pickle object
        with open(f'pickels/background_data_{iteration}.pkl', 'wb') as file:
            pickle.dump(background_windows, file)
        
        # Retrain the model with the updated anomaly and background data
        anomalies_labels = []
        for i in range(len(anomalies_windows)):
            anomalies_labels.append(np.array([1 for j in anomalies_windows[i]]))
        
        background_labels = []
        for i in range(len(background_windows)):
            background_labels.append(np.array([0 for j in background_windows[i]]))
        
        # Concatenate array
        X = []
        for i in range(len(anomalies_windows)):
            X.append(np.concatenate((anomalies_windows[i], background_windows[i])))
        
        y = []
        for i in range(len(anomalies_windows)):
            y.append(np.concatenate((anomalies_labels[i], background_labels[i])))

        # Shuffle data
        randomized = []
        for i in range(len(anomalies_windows)):
            combined = np.column_stack((X[i], y[i]))
            np.random.seed(self.seed)
            np.random.shuffle(combined)
            randomized.append(combined)
            
        # Split the shuffled array back into data and labels
        for i in range(len(anomalies_windows)):    
            X[i], y[i] = randomized[i][:, :-1], randomized[i][:, -1]

        # Load the models
        filename = f'models/rf_model_high_{iteration - 1}.sav'
        model_high = pickle.load(open(filename, 'rb'))
        filename = f'models/rf_model_med_{iteration - 1}.sav'
        model_med = pickle.load(open(filename, 'rb'))
        filename = f'models/rf_model_low_{iteration - 1}.sav'
        model_low = pickle.load(open(filename, 'rb'))

        # Increase estimators and set warm_start to True
        model_high.n_estimators += 10
        model_high.warm_start = True
        model_med.n_estimators += 10
        model_med.warm_start = True
        model_low.n_estimators += 10
        model_low.warm_start = True

        # Split the shuffled data into the training and testing set
        X_train, y_train, X_test, y_test = [], [], [], []
        for i in range(len(anomalies_windows)):
            X_train.append(X[i][:int(len(X[i]) * 0.75)])
            y_train.append(y[i][:int(len(X[i]) * 0.75)])
            X_test.append(X[i][int(len(X[i]) * 0.75):])
            y_test.append(y[i][int(len(X[i]) * 0.75):])

        # Fit the model to the training data
        model_high.fit(X_train[0], y_train[0]) # Long length data windows
        model_med.fit(X_train[1], y_train[1]) # Medium legth data windows
        model_low.fit(X_train[2], y_train[2]) # Short length data windows

        from sklearn.metrics import confusion_matrix as cm
        confusion_matrix_high = cm(y_test[0], model_high.predict(X_test[0]))
        print(confusion_matrix_high)
        confusion_matrix_med = cm(y_test[1], model_med.predict(X_test[1]))
        print(confusion_matrix_med)
        confusion_matrix_low = cm(y_test[2], model_low.predict(X_test[2]))
        print(confusion_matrix_low)
        
        # Get the number of rows labeled as anomalies in y_test
        num_anomalies = len([i for i in y_test[0] if i==1])
        print('Number of anomalies in test set:', num_anomalies)
        num_anomalies = len([i for i in y_test[1] if i==1])
        print('Number of anomalies in test set:', num_anomalies)
        num_anomalies = len([i for i in y_test[2] if i==1])
        print('Number of anomalies in test set:', num_anomalies)
        
        # Save the model to disk
        filename = f'models/rf_model_high_{iteration}.sav'
        pickle.dump(model_high, open(filename, 'wb'))
        filename = f'models/rf_model_med_{iteration}.sav'
        pickle.dump(model_med, open(filename, 'wb'))
        filename = f'models/rf_model_low_{iteration}.sav'
        pickle.dump(model_low, open(filename, 'wb'))

if __name__ == '__main__':
    
    # Create an instance of the model
    window_size = 32
    imRF = imRF(station=901, trim_percentage=10, ratio=5, num_variables=6, 
                window_size=window_size, stride=1, seed=0)
    
    # Implement iterative process
    for i in range(0, 5):
        
        if i == 0:
            print(f'[INFO] Iteration {i}')
            # Extract the anomalies and first batch of background
            anomalies_indexes = imRF.anomalies()
            
            background_indexes = imRF.init_background(anomalies_indexes)
            
            # Train the first version of the model
            imRF.init_RandomForest()

        else:
            print(f'[INFO] Iteration {i}')
            # Extract new background data
            background_indexes = imRF.background(anomalies_indexes, background_indexes, iteration=i)
            
            # Iteratively predict on the new background data and update the model
            imRF.RandomForest(iteration=i)
