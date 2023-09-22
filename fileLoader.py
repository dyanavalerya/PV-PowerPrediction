import sys, os
import pandas as pd


def loadFile(file_name):
    datafolder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset/'))

    # list files in folder
    files_list = os.listdir(datafolder_path)
    for i in files_list:
        print(i)
    
    file_path = os.path.join(datafolder_path, file_name)

    # assigns data in file, if the file name exists
    if (os.path.isfile(file_path)):
        file_data = pd.read_csv(file_path)
    else:
        print("File name does not exist. Remember to include file type in file name")
        sys.exit()

    return file_data















