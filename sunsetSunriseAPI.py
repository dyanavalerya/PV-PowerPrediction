# ###################################
# This program uses the Sunset and sunrise times API, found on https://sunrise-sunset.org/api
# ###################################
import requests
import json
import fileLoader as fl
from datetime import *


# Defining functions
def load_all_datasets(meta, i):
    """
    Load data per station. Add a column with the station number.

    Returns:
    loaded_data (pandas.DataFrame): A pandas dataframe containing data for one station.
    """

    # i = station number
    name = f"station0{i}"
    loaded_data = fl.loadFile(f"station0{i}.csv")
    loaded_data["station"] = i
    for row in meta.iterrows():
        if row[1]["Station_ID"] == name:
            loaded_data["power"] = loaded_data["power"] / meta["Capacity"][row[0]]

    return loaded_data


def sunrisesunset(f, params):
    a = requests.get(f, params)
    a = json.loads(a.text)
    a = a["results"]
    # Convert from 12 hr format (that is given by the API) to 24 hr format (present in the data)
    # remove redundant random date assigned and keep only the time using split() function
    # The time is in Universal time (UTC) (i.e. when 21:33 in Britain, it is 5:33 in China)
    sunrise = str(datetime.strptime(a["sunrise"], '%I:%M:%S %p')).split(' ', 1)[1]
    sunset = str(datetime.strptime(a["sunset"], '%I:%M:%S %p')).split(' ', 1)[1]
    return sunrise, sunset


# Main function
def main():
    # Importing data
    meta = fl.loadFile("metadata.csv")
    # _________________________________ CHANGE STATION NUMBER HERE ! _____________________________________________
    station_nr = 0
    station_data = load_all_datasets(meta, station_nr)

    f = r"https://api.sunrise-sunset.org/json?"

    count = 0
    # Initialize an non existing date
    previous_date = '0000-00-00'
    sunrise_sunset = 0.0, 0.0
    # Loop through the entire data from all stations
    for i, data in enumerate(station_data.values):
        # Find latitude and longitude from metadata variable based on station number stored in data[15]
        longitude = meta.values[data[15]][10]
        latitude = meta.values[data[15]][11]
        # Get the date of the measurement
        current_date = data[0].split(' ', 1)[0]

        # Call the API per day and not per each measurement, to save computations time
        if previous_date != current_date:
            previous_date = current_date

            params = {"lat": latitude, "lng": longitude, "date": current_date}
            # Use the API to return sunrise and sunset times based on the three parameters
            sunrise_sunset = sunrisesunset(f, params)

        # keep only data from sunrise til sunset, i.e. remove night data
        # use both hour and minutes
        current_time = float(data[0].split(' ', 1)[1].split(':')[0]+'.'+data[0].split(' ', 1)[1].split(':')[1])
        sunrise_time = float(sunrise_sunset[0].split(':')[0] + '.' + sunrise_sunset[0].split(':')[1])
        sunset_time = float(sunrise_sunset[1].split(':')[0] + '.' + sunrise_sunset[1].split(':')[1])
        # Check if current time is not in between sunrise (PM) and sunset (AM) because of UTC
        if sunrise_time > current_time > sunset_time:
            # remove row i based on the Timestamp index
            # Make sure to run this only one station at a time, since there are the same
            # timestamps for each station, but located at different latitudes & longitudes
            # resulting in different sunset & sunrise
            station_data.drop(station_data.index[i-count], inplace=True)
            count = count + 1

    station_data.to_pickle(f"station0{station_nr}.pkl")

    print("test")


# Executing main function
if __name__ == "__main__":
    main()
