import numpy as np
import pysolar.solar as sun
from datetime import datetime
import re

def calcIrradiances(z_deg):
    # Precompute constants
    pi_over_180 = np.pi / 180
    exp_factor = np.exp(-0.075 * (90 - z_deg))

    # Calculate DIFF
    DIFF = np.where(z_deg > 128.9142913, 0, 14.29 + 21.04 * ((np.pi / 2) - z_deg * pi_over_180))

    # Calculate DNI
    DNI = np.where(z_deg > 90, 0, 950.2 * (1 - exp_factor))

    # Calculate GHI
    GHI = DNI * np.cos(z_deg * pi_over_180) + DIFF

    return [DNI, DIFF, GHI]

def lmd_clear_sky(meta_data, data, station_number, time):
    date_format = '%Y-%m-%d %H:%M:%S%z'

    st_latitude = meta_data["Latitude"][station_number]
    st_longitude = meta_data["Longitude"][station_number]
    
    datetime_obj = datetime.strptime(time + "+00:00", date_format)
    temp = data.loc[time, 'lmd_temperature']
    pres = data.loc[time, 'lmd_pressure']

    alt = sun.get_altitude(st_latitude, st_longitude, datetime_obj, elevation=0, temperature=temp, pressure=pres)
    alt_rad = np.deg2rad(alt)

    zenith = float(90) - alt

    azi = sun.get_azimuth(st_latitude, st_longitude, datetime_obj, elevation=0)
    azi_rad = np.deg2rad(azi)

    array_tilt = int(re.findall(r'\d+', meta_data.loc[station_number, 'Array_Tilt'])[0])
    array_rad = np.deg2rad(array_tilt)
    array_dir_rad = np.deg2rad(180)

    csm_irradiation = calcIrradiances(zenith)

    panel_irrad = csm_irradiation[2] * (np.cos(alt_rad) * np.sin(array_rad) * np.cos(array_dir_rad - azi_rad) + np.sin(alt_rad) * np.cos(array_rad))
    panel_irrad = max(0, panel_irrad)

    panel_size = meta_data.loc[station_number, 'Panel_Size']
    watt_pr_panel = panel_size * panel_irrad

    panel_number = meta_data.loc[station_number, 'Panel_Number']
    station_watt_tot = watt_pr_panel * panel_number

    panel_eff = 0.16
    station_panel_watt_tot = station_watt_tot * panel_eff

    return [csm_irradiation, panel_irrad, watt_pr_panel, station_watt_tot, station_panel_watt_tot]

