import sys, os
import pandas as pd


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

    # assign data in file
    if (os.path.isfile(file_path)):
        file_data = pd.read_csv(file_path)
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
    name2=name.set_index('date_time', inplace=False)
    #add the date_time column to the name2 dataframe
    name2['date_time']=name['date_time']
    #set the date_time column as the index
    sliced = name2.loc[start_time:end_time]
    return sliced












