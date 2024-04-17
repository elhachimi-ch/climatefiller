import csv
import pickle
import re
from os import listdir
from os.path import isfile, join
from time import sleep
from math import sqrt
import numpy as np 
#from stringdist.pystringdist.levenshtein import levenshtein as ed
import calendar
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import math
import requests
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, mean_squared_log_error, mean_absolute_error, classification_report

class Lib:
    
    def __init__(self, *args, **kwargs):
        # Stephan Boltzmann constant (W m-2 K-4)
        self.SB = 5.670373e-8
        # heat capacity of dry air at constant pressure (J kg-1 K-1)
        self.C_PD = 1003.5
        # heat capacity of water vapour at constant pressure (J kg-1 K-1)
        self.C_PV = 1865
        # ratio of the molecular weight of water vapor to dry air
        self.epsilon = 0.622
        # Psicrometric Constant kPa K-1
        self.PSIRC = 0.0658
        # gas constant for dry air, J/(kg*degK)
        self.R_D = 287.04
        # acceleration of gravity (m s-2)
        self.G = 9.8
    
    @staticmethod
    def et0_penman_monteith(row):
        # input variables
        # T = 25.0  # air temperature in degrees Celsius
        # RH = 60.0  # relative humidity in percent
        # u2 = 2.0  # wind speed at 2 m height in m/s
        # Rs = 15.0  # incoming solar radiation in MJ/m2/day
        # lat = 35.0  # latitude in degrees
        
        ta_max, ta_min, rh_max, rh_min, u2_mean, rs_mean, lat, elevation, doy =  row['ta_max'], row['ta_min'], row['rh_max'], row['rh_min'], row['u2_mean'], row['rg_mean'], row['lat'], row['elevation'], row['doy']
        
        # constants
        ALBEDO = 0.23  # Albedo coefficient for grass reference surface
        GSC = 0.082  # solar constant in MJ/m2/min
        SIGMA = 0.000000004903  # Stefan-Boltzmann constant in MJ/K4/m2/day
        G = 0  # Soil heat flux density (MJ/m2/day)
        z = 2 # Convert wind speed measured at different heights above the soil surface to wind speed at 2 m above the surface, assuming a short grass surface.

        # convert units
        rs_mean *= 0.0864  # convert watts per square meter to megajoules per square meter 0.0288 = 60x60x8hours or 0.0864 for 24 hours
        ta_mean = (ta_max + ta_min) / 2
        ta_max_kelvin = ta_max + 273.16  # air temperature in Kelvin
        ta_min_kelvin = ta_min + 273.16  # air temperature in Kelvin
        
        # saturation vapor pressure in kPa
        es_max = 0.6108 * math.exp((17.27 * ta_max) / (ta_max + 237.3))
        es_min = 0.6108 * math.exp((17.27 * ta_min) / (ta_min + 237.3))
        es = (es_max + es_min) / 2
        
        # actual vapor pressure in kPa
        ea_max_term = es_max * (rh_min / 100)
        ea_min_term = es_min * (rh_max / 100)
        ea = (ea_max_term + ea_min_term) / 2
        
        # in the absence of rh_max and rh_min
        #ea = (rh_mean / 100) * es
        
        # when using equipement where errors in estimation rh min can be large or when rh data integrity are in doubt use only rh_max term
        #ea = ea_min_term  
        
        delta = (4098 * (0.6108 * math.exp((17.27 * ta_mean) / (ta_mean + 237.3)))) / math.pow((ta_mean + 237.3), 2) # slope of the vapor pressure curve in kPa/K
        
        
        atm_pressure = math.pow(((293.0 - (0.0065 * elevation)) / 293.0), 5.26) * 101.3
        # psychrometric constant in kPa/K
        gamma = 0.000665 * atm_pressure
        
        # Calculate u2
        u2 = u2_mean * (4.87 / math.log((67.8 * z) - 5.42))
        
        # Calculate extraterrestrial radiation
        dr = 1 + 0.033 * math.cos(2 * math.pi / 365 * doy)
        d = 0.409*math.sin( (((2*math.pi) / 365) * doy) - 1.39)
        
        
        # sunset hour angle
        phi = math.radians(lat)
        omega = math.acos(-math.tan(phi)*math.tan(d))
        ra = ((24 * 60)/math.pi) * GSC * dr * ((omega * math.sin(phi) * math.sin(d)) + (math.cos(phi) * math.cos(d) * math.sin(omega)))
        
        # Calculate clear sky solar radiation
        rso = (0.75 + (2e-5 * elevation)) * ra
        
        # Calculate net solar shortwave radiation 
        rns = (1 - ALBEDO) * rs_mean
        
        # Calculate net longwave radiation
        rnl = SIGMA * (((math.pow(ta_max_kelvin, 4) + math.pow(ta_min_kelvin, 4)) / 2) * (0.34 - (0.14 * math.sqrt(ea))) * ((1.35 * (rs_mean / rso)) - 0.35))
        
        # Calculate net radiation
        rn = rns - rnl
        
        
        # decompose et0 to two terms to facilitate the calculation
        """rng = 0.408 * rn
        radiation_term = ((delta) / (delta + (gamma * (1 + 0.34 * u2)))) * rng
        pt = (gamma) / (delta + gamma * (1 + (0.34 * u2)))
        tt = ((900) / (ta_mean + 273) ) * u2
        wind_term = pt * tt * (es - ea)
        et0 = radiation_term + wind_term """
        
        et0 = ((0.408 * delta * (rn - G)) + gamma * ((900 / (ta_mean + 273)) * u2 * (es - ea))) / (delta + (gamma * (1 + 0.34 * u2)))

        # output result
        return et0
    
    @staticmethod
    def et0_hargreaves(row):
        ta_mean, ta_max, ta_min, lat, doy =  row['ta_mean'], row['ta_max'], row['ta_min'], row['lat'], row['doy']
        
        # constants
        GSC = 0.082  # solar constant in MJ/m2/min

        # Calculate extraterrestrial radiation
        dr = 1 + 0.033 * math.cos(2 * math.pi / 365 * doy)
        d = 0.409*math.sin( (((2*math.pi) / 365) * doy) - 1.39)

        # sunset hour angle
        phi = math.radians(lat)
        omega = math.acos(-math.tan(phi)*math.tan(d))
        ra = ((24 * 60)/math.pi) * GSC * dr * ((omega * math.sin(phi) * math.sin(d)) + (math.cos(phi) * math.cos(d) * math.sin(omega)))
        
        et0 = 0.0023 * (ta_mean + 17.8) * (ta_max - ta_min) ** 0.5 * 0.408 * ra

        return et0

    @staticmethod
    def get_elevation(lat, lon):
        """
        Returns the elevation (in meters) and latitude (in degrees) for a given set of coordinates.
        Uses the Open Elevation API (https://open-elevation.com/) to obtain the elevation information.
        """
        # 'https://api.open-elevation.com/api/v1/lookup?locations=10,10|20,20|41.161758,-8.583933'
        url = f'https://api.opentopodata.org/v1/aster30m?locations={lat},{lon}'
        response = requests.get(url)
        print(response.json())
        data = response.json()
        elevation = data['results'][0]['elevation']
        #latitude = data['results'][0]['latitude']
        return elevation
    
    @staticmethod
    def get_2m_wind_speed(row):
        uz, vz, z = row['u10'], row['v10'], 10
        
        # calculate 10m wind speed magnitude
        wsz = math.sqrt(math.pow(uz, 2) + math.pow(vz, 2))
        
        # calculate 2m wind speed using logarithmic wind profile model
        ws = wsz * (4.87 / math.log((67.8 * z) - 5.42))
        
        return ws
    
    @staticmethod
    def energy2evap(energy):
        """
        Convert energy (e.g. radiation energy) in MJ m-2 day-1 to the equivalent
        evaporation, assuming a grass reference crop.

        Energy is converted to equivalent evaporation using a conversion
        factor equal to the inverse of the latent heat of vapourisation
        (1 / lambda = 0.408).

        Based on FAO equation 20 in Allen et al (1998).

        :param energy: Energy e.g. radiation or heat flux [MJ m-2 day-1].
        :return: Equivalent evaporation [mm day-1].
        :rtype: float
        """
        return 0.408 * energy
    
    @staticmethod
    def estimate_pressure(z):
        ''' Calculates the barometric pressure above sea level.

        Parameters
        ----------
        z: float
            height above sea level (m).

        Returns
        -------
        p: float
            air pressure (mb).'''

        p = 1013.25 * (1.0 - 2.225577e-5 * z)**5.25588
        return np.asarray(p)
    

    def calc_rho(self, p, ea, T_A_K):
        '''Calculates the density of air.

        Parameters
        ----------
        p : float
            total air pressure (dry air + water vapour) (mb).
        ea : float
            water vapor pressure at reference height above canopy (mb).
        T_A_K : float
            air temperature at reference height (Kelvin).

        Returns
        -------
        rho : float
            density of air (kg m-3).

        References
        ----------
        based on equation (2.6) from Brutsaert (2005): Hydrology - An Introduction (pp 25).'''

        # p is multiplied by 100 to convert from mb to Pascals
        rho = ((p * 100.0) / (self.R_D * T_A_K)) * (1.0 - (1.0 - self.epsilon) * ea / p)
        return np.asarray(rho)
    
    def estimate_vapor_pressure(self, Ta, Rh=None):
        """
        Calculate actual vapor pressure (e) given temperature (T) in Celsius
        and relative humidity (RH) as a percentage.
        """
        # Calculate saturation vapor pressure (es) using Magnus-Tetens formula
        es = 6.112 * math.exp((17.67 * Ta) / (Ta + 243.5))
        
        if Rh is None:
            return es
        e = (Rh / 100) * es
        return e