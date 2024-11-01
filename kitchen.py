import time
from data_science_toolkit.dataframe import DataFrame
from climatefiller import ClimateFiller
from lib import Lib
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
import optuna
from sklearn.metrics import mean_squared_error
from optuna.samplers import CmaEsSampler
from optuna.pruners import HyperbandPruner, MedianPruner

# Custom aggregation function
def custom_sum(values):
    if values.isna().any():
        return None
    else:
        return values.sum()

def et0_hargreaves_samani(x, C, a, b):
    return C * (x[0] + a) * (((x[1] - x[2]) ** b) * 0.408 * x[3])



def et0_makkink(x, c1, c2):
    """
    Calculate the reference evapotranspiration using the Priestley-Taylor method.

    Parameters:
    alpha (float): Empirical coefficient, typically around 1.26.
    Delta (float): Slope of the saturation vapor pressure curve at air temperature (kPa/°C).
    gamma (float): Psychrometric constant (kPa/°C).
    Rn (float): Net radiation at the crop surface (MJ/m²/day).
    G (float): Soil heat flux (MJ/m²/day), often assumed to be zero for daily calculations.

    Returns:
    float: Estimated ET0 in mm/day.
    """
    
    
    ta_mean = x[0]
    rs_mean = x[1]
    elevation = x[2]
    
    delta = Lib.slope_saturation_vapor_pressure_curve(ta_mean)
    gama = Lib.psychrometric_constant(elevation, ta_mean)
    lam = Lib.latent_heat_of_vaporization(ta_mean)
    
    # convert units
    rs_mean *= 0.0864  # convert watts per square meter to megajoules per square meter 0.0288 = 60x60x8hours or 0.0864 for 24 hours
    
    et0 = ((c1 * delta * rs_mean) / ((delta + gama) * lam)) - c2
    return et0


def et0_priestley_taylor_daily(x, alpha):
    # input variables
    # T = 25.0  # air temperature in degrees Celsius
    # RH = 60.0  # relative humidity in percent
    # u2 = 2.0  # wind speed at 2 m height in m/s
    # Rs = 15.0  # incoming solar radiation in MJ/m2/day
    # lat = 35.0  # latitude in degrees
    #ta_mean, rs_mean, rh_mean, lat, elevation, doy =  row['ta_mean'], row['rs_mean'], row['rh_mean'], row['lat'], row['elevation'], row['doy']
    G = 0  # Soil heat flux density (MJ/m2/day)
    
    #  [ 18.45178986   1.837304     3.          31.65936     92.88200378  35.92900085 153.58399643 572]
    ta_max, ta_min, doy, lat, rh_max, rh_min, rs_mean, elevation = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
    
    
    rs_mean *= 0.0864  # convert watts per square meter to megajoules per square meter 0.0288 = 60x60x8hours or 0.0864 for 24 hours

    ta_mean = (ta_max + ta_min) / 2
    ta_max_kelvin = ta_max + 273.16  # air temperature in Kelvin
    ta_min_kelvin = ta_min + 273.16  # air temperature in Kelvin
    
    # saturation vapor pressure in kPa
    es_max = 0.6108 * np.exp((17.27 * ta_max) / (ta_max + 237.3))
    es_min = 0.6108 * np.exp((17.27 * ta_min) / (ta_min + 237.3))
    
    # actual vapor pressure in kPa
    ea_max_term = es_max * (rh_min / 100)
    ea_min_term = es_min * (rh_max / 100)
    ea = (ea_max_term + ea_min_term) / 2
    
    delta = Lib.slope_saturation_vapor_pressure_curve(ta_mean) # slope of the vapor pressure curve in kPa/K
    
    # psychrometric constant in kPa/K
    gamma = Lib.psychrometric_constant(elevation, ta_mean)
    
    
    # Calculate extraterrestrial radiation
    ra = Lib.extraterrestrial_radiation_daily(lat, doy)
    
    # Calculate clear sky solar radiation
    rso = (0.75 + (2e-5 * elevation)) * ra
    
    # Calculate net solar shortwave radiation 
    rns = (1 - Lib.ALBEDO) * rs_mean
    
    # Calculate net longwave radiation
    
    rnl = Lib.SIGMA * (((np.power(ta_max_kelvin, 4) + np.power(ta_min_kelvin, 4)) / 2) * (0.34 - (0.14 * np.sqrt(ea))) * ((1.35 * (rs_mean / rso)) - 0.35))
    
    # Calculate net radiation
    rn = rns - rnl
    
    et0 = (alpha * delta * (rn - G) * 0.408) / (delta + gamma)
    
    #print('inside the function: ', et0)
        
    # output result
    return et0


# Define the objective function to be optimized
def objective(trial, data):
    
    # Define the search space for the constants a and b
    # PT
    #alpha = trial.suggest_float('alpha', 0, 10, step=0.0001)
    # HS
    #c = trial.suggest_float('c', 0, 1, step=0.0001)
    #a = trial.suggest_float('a', 0, 50, step=0.1)
    #b = trial.suggest_float('b', 0, 1, step=0.01)
    # MK
    #c1 = trial.suggest_float('c1', 0, 10, step=0.01)
    #c2 = trial.suggest_float('c2', 0, 10, step=0.01)
    # K1
    k1 = trial.suggest_float('k1', 0, 10, step=0.0001)
    
    
    #data.add_column_based_on_function('et0_pt', lambda row: Lib.et0_priestley_taylor_daily(row, alpha))
    #data.add_column_based_on_function('et0_hs', lambda row: Lib.et0_hargreaves_samani(row, c, a, b))
    #data.add_column_based_on_function('et0_mk', lambda row: Lib.et0_makkink(row, c1, c2))
    data.add_column_based_on_function('et0_ab', lambda row: Lib.et0_abtew(row, k1))
    rmse = data.similarity_measure('et0_pm', 'et0_ab', 'ts')['RMSE']
    return rmse

def main():
    ti = time.time()
    
    #data = DataFrame("data/oukaimeden_full_p_et0_bc.csv")
    cf = ClimateFiller(r"data\r3_full_p_et0_bc.csv")
    #cf = ClimateFiller(r"C:\Users\elhac\OneDrive\Desktop\kitchen\projects\pythonsnippets\data\california\cimis_data.csv")
    
    print(cf.climate_zones_classification())

    
   
    print(time.time() - ti)


if __name__ == '__main__':
    main()


