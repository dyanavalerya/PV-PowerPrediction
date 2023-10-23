import sys, os
import pandas as pd
import numpy as np
from scipy.stats import zscore


def loadFile(file_name, path=None):
    if path == None:
        print(f"Path of current program:\n", os.path.abspath(os.path.dirname(__file__)))
        datafolder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset/'))
    else:
        datafolder_path = path
    
    # check if folder exists if not then error
    if (os.path.isdir(datafolder_path)):
        print(f"Path of dataset folder:\n", datafolder_path)
    else:
        print("Data folder path does not exist")
        sys.exit()
    
    file_path = os.path.join(datafolder_path, file_name)
    file_data=None
    # assign data in file
    if (os.path.isfile(file_path)):
        file_data = pd.read_csv(file_path,header=0)
        if not(file_name=="metadata.csv"):
            file_data.index = pd.DatetimeIndex(file_data["date_time"])
    else:
        print("File name does not exist. Remember to include file type in file name")
        sys.exit()

    print("\n*** File succesfully loaded ***")
    print("\nFile preview:")
    print(file_data.head())
    return file_data

def fileInfo(file):
    time_start = file["date_time"][0]
    print(f"First date_time in dataset is: {time_start}")


def sliceData(name,start_time,end_time):
    print('data sliced from ',start_time,' to ',end_time)
    sliced = name[start_time:end_time]
    return sliced

def checkDate(name):
    datetime_object = pd.to_datetime(name['date_time'])
    
    date_diff = datetime_object.diff()

    mistake_indices = np.where(date_diff != pd.Timedelta(minutes=15))[0]
    mistake_indices = mistake_indices[mistake_indices != 0]
    date_fails = len(mistake_indices)

    if date_fails > 0:
        print(f'Mistakes found at indices:')
        for idx in mistake_indices:
            print(f'Index {idx-1}:{idx} Between {datetime_object.iloc[idx-1]} and {datetime_object.iloc[idx ]}\n')
    else:
        print('No mistakes in time found.')

    print(f'date check done. {date_fails} mistakes found\n')
    return date_fails
        
def checkParam(name, threshold_outlier):
    print('check for outliers and empty cells with outlier z score threshold=', threshold_outlier, '\n')
    
    outlier_counter = 0
    empty_counter = 0

    empty_cells = name.isna()

    for j in range(1,15):  # Loop from 0 to 13 (inclusive)
        print('Checking column', j+1)
        data = name.iloc[:, j]  # Extract the column as a NumPy array
        z_scores = zscore(data)

        outliers = np.abs(z_scores) > threshold_outlier
        empty = empty_cells.iloc[:, j].to_numpy()

        outlier_indices = np.where(outliers)[0]
        empty_indices = np.where(empty)[0]

        for i in outlier_indices:
            print(name.columns[j])
            print('outlier found at (', j+1, i, ')')
            outlier_counter += 1
        
        for i in empty_indices:
            print(name.columns[j])
            print('empty cell at', j+1, i)
            empty_counter += 1
        
        print('Done checking column', j+1, '\n')
    
    print(f'check for outliers and empty cells with outlier z score threshold={threshold_outlier} finished. \n Found {outlier_counter} outliers and {empty_counter} empty cells')
    fails=[outlier_counter,empty_counter]
    return fails

def calculate_average_and_variance(dataframes):
    """
    Calculate the average and variance of each column (except the first column) 
    for each DataFrame individually.
    
    Parameters:
        dataframes (list of pd.DataFrame): A list of input DataFrames.
        
    Returns:
        pd.DataFrame: A DataFrame with columns 'DataFrame', 'Column', 'Average', and 'Variance'.
    """
    stats = []

    for i, df in enumerate(dataframes):
        column_stats = df.iloc[:, 1:].agg(['mean', 'var']).T
        column_stats.columns = ['Average', 'Variance']
        column_stats['Column'] = column_stats.index
        column_stats['DataFrame'] = f'DataFrame{i}'
        stats.append(column_stats)

    return pd.concat(stats, ignore_index=True)




















