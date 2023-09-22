import sys, os
import pandas as pd


def loadFile(file_name, path=None):
    if path == None:
        print(os.path.abspath(os.path.dirname(__file__)))
        datafolder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset/'))
    else:
        datafolder_path = path
    print(datafolder_path)
    # list files in folder
    # check if folder exists if not error
    if not (os.path.isdir(datafolder_path)):
        print("Data folder path does not exist")
        sys.exit()
    files_list = os.listdir(datafolder_path)
    for i in files_list:
        print(i)
    
    file_path = os.path.join(datafolder_path, file_name)

    # assigns data in file, if the file name exists
    if (os.path.isfile(file_path)):
        file_data = pd.read_csv(file_path)
        print("File succesfully loaded")
        print(file_data.keys())
    else:
        print("File name does not exist. Remember to include file type in file name")
        sys.exit()

    return file_data

# station02 = loadFile("station02.csv", r'C:\Users\andre\OneDrive - Aalborg Universitet\_Universitet\ES7\_ES7 project\literatureAndDataset\dataset')

# print(station02.keys())


def sliceData(name,start_time,end_time):
    name2=name.set_index('date_time', inplace=False)
    #add the date_time column to the name2 dataframe
    name2['date_time']=name['date_time']
    #set the date_time column as the index
    sliced = name2.loc[start_time:end_time]
    return sliced












