import os, sys
# if file is inside a folder then append base folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# load custom functions
from PV_PredictLib import fileLoader as fl
from PV_PredictLib import plotFunctions as pf
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import pysolar.solar as sun
from datetime import datetime,timezone
# load metadata 
st_meta=fl.loadFile("metadata.csv")
def calculateIrradiances(data,metadata,stationNum):
    st_longitude = metadata["Longitude"][stationNum]
    st_latitude = metadata["Latitude"][stationNum]
    date_format = '%Y-%m-%d %H:%M:%S%z'

    # Function to be used in same way as an alpha parameter in apply function
    def get_zenith_angle(row):
        altitude = sun.get_altitude(st_latitude, st_longitude, datetime.strptime(row["date_time"]+"+00:00", date_format), elevation=0, temperature=row["nwp_temperature"], pressure=row["nwp_pressure"])
        return np.radians(float(90) - altitude)
    def calculate_dhi(row):
        return row["nwp_globalirrad"] - row["nwp_directirrad"] * np.cos(get_zenith_angle(row))
    def calculate_lmd(row):
        return ((row["lmd_totalirrad"] - row["lmd_diffuseirrad"])/np.cos(get_zenith_angle(row)))

    data["hmd_diffuseirrad"] = data.apply(calculate_dhi, axis=1)
    data["hmd_directirrad"] = data.apply(calculate_lmd, axis=1)

    # clean the data
    data["hmd_directirrad"]=data["hmd_directirrad"].rolling(10).median()
    # if value is negative set to last value
    while(len(data.loc[data["hmd_directirrad"]<0])):
        data.loc[data["hmd_directirrad"]<0,"hmd_directirrad"]=0

    return data


def main():
    for i in range(0,8):
        # Loading station
        print("Loading station",i)
        st_data=fl.loadPkl(f"station0{i}.pkl")
        #if hmd_directirrad key is not in the dataset then calculate it
        if "hmd_directirrad" not in st_data.keys():
            print("Calculating hmd_directirrad")
            st_data=calculateIrradiances(st_data,st_meta,i)
            # Saving station
            print("Saving station")
            # path relative to the project folder
            path=os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset'))
            file_path=os.path.join(path,f"station0{i}.pkl")
            temp = open(file_path, 'wb+')
            pickle.dump(st_data,temp)
            temp.close()
            print("File saved as: ", file_path)
        print("Allready calculated hmd_directirrad for station",i)
if __name__ == "__main__":
    main()  
        