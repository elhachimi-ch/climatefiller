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
    
    # Stephan Boltzmann constant (W m-2 K-4)
    SB = 5.670373e-8
    # heat capacity of dry air at constant pressure (J kg-1 K-1)
    C_PD = 1003.5
    # heat capacity of water vapour at constant pressure (J kg-1 K-1)
    C_PV = 1865
    # gas constant for dry air (rd), J/(kg*degK)
    GAS_CONSTANT_FOR_DRY_AIR = 287.04
    # acceleration of gravity (m s-2)
    G = 9.8
    # the density of water kg m-3
    DENSITY_OF_WATER = 1000 
    SOLAR_CONSTANT = 0.08202 # Solar constant (G_sc) in MJ/m²/min 
    SOLAR_CONSTANT_W_PER_M2 = 1367 # Solar constant (G_sc) in W/m² 
    ALBEDO = 0.23  # Albedo coefficient for grass reference surface
    CP = 1.013e-3  # Specific heat of air at constant pressure (MJ/kg°C)
    EPSILON = 0.622  # Ratio molecular weight of water vapor/dry air
    PI = math.pi
    
    def __init__(self, *args, **kwargs):
        pass
         
    
    @staticmethod
    def et0_penman_monteith(
        row,
        ta_column_name,
        rs_column_name,
        rh_column_name,
        u_column_name):
        # input variables
        # T = 25.0  # air temperature in degrees Celsius
        # RH = 60.0  # relative humidity in percent
        # u2 = 2.0  # wind speed at 2 m height in m/s
        # Rs = 15.0  # incoming solar radiation in MJ/m2/day
        # lat = 35.0  # latitude in degrees
        
        ta_max, ta_min, rh_max, rh_min, u2_mean, rs_mean, lat, elevation, doy =  row['ta_max'], row['ta_min'], row['rh_max'], row['rh_min'], row['u_mean'], row['rs_mean'], row['lat'], row['elevation'], row['doy']
        
        # constants
        ALBEDO = 0.23  # Albedo coefficient for grass reference surface
        GSC = 0.082  # solar constant in MJ/m2/min
        SIGMA = 0.000000004903  # Stefan-Boltzmann constant in MJ/K4/m2/day
        G = 0  # Soil heat flux density (MJ/m2/day)
        z = 2 # Convert wind speed measured at different heights above the soil surface to wind speed at 2 m above the surface, assuming a short grass surface.

        # convert units 12 assume hours of day 4.32e-2 
        rs_mean *= 4.32e-2  # convert watts per square meter to megajoules per square meter 2.88e-2 = 60x60x8hours or 0.0864 for 24 hours
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
        
        delta = Lib.slope_vapor_pressure_curve(ta_mean) # slope of the vapor pressure curve in kPa/K
        
        # psychrometric constant in kPa/K
        gamma = Lib.psychrometric_constant(elevation, ta_mean)
        
        # Calculate u2
        u2 = u2_mean
        
        # Calculate extraterrestrial radiation
        ra = Lib.extraterrestrial_radiation(lat, doy)
        
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
    def et0_penman_monteith_hourly(
        row,
        ta_column_name,
        rs_column_name,
        rh_column_name,
        u_column_name,
        standard_meridian,
        reference_crop,
        ):
        # input variables
        # T = 25.0  # air temperature in degrees Celsius
        # RH = 60.0  # relative humidity in percent
        # u2 = 2.0  # wind speed at 2 m height in m/s
        # Rs = 15.0  # incoming solar radiation in MJ/m2/day
        # lat = 35.0  # latitude in degrees
        
        ta_c, rs, rh, u2, lat, elevation, doy, lon, hod =  row[ta_column_name], row[rs_column_name], row[rh_column_name], row[u_column_name], row['lat'], row['elevation'], row['doy'], row['lon'], row['hod']
        
        # convert units
        rs *= 3.6e-3  # convert watts per square meter to megajoules per square meter 0.0288 = 60x60x8hours or 0.0864 for 24 hours
        ta_k = ta_c + 273.16  # air temperature in Kelvin
        
        lambda_heat = Lib.latent_heat_of_vaporization(ta_c)
        
        # saturation vapor pressure in kPa
        es = Lib.saturation_vapor_pressure(ta_c)
        
        # actual vapor pressure in kPa
        ea = Lib.actual_vapor_pressure(es, rh)
        
        epsilon_net = 0.34 - (0.14 * math.sqrt(ea))
        
        # slope of the vapor pressure curve in kPa/K
        delta = Lib.slope_vapor_pressure_curve(ta_c)
        
        # psychrometric constant in kPa/K
        gamma = Lib.psychrometric_constant(elevation, ta_c)
        
        # Calculate extraterrestrial radiation
        ra = Lib.extraterrestrial_radiation_hourly(doy, lat, lon, hod, standard_meridian)
        
        # Calculate net solar shortwave radiation 
        rns = (1 - Lib.ALBEDO) * rs
        
        # Rso
        rso = Lib.rso(ra, elevation)
        
        # cloudness factor
        f = Lib.cloudness_factor(rs, rso)
        
        rnl = f * epsilon_net * Lib.stephan_boltzmann(ta_c)
      
        rn = (0.77 * rns) - rnl
        
        if reference_crop == 'grass':
            crop_dependent_factor = 37
            if rn > 0:
                g = 0.1 * rn
                # Bulk surface resistance and aerodynamic resistance coefficient
                CD = 0.24 
            else:
                g = 0.5 * rn
                # Bulk surface resistance and aerodynamic resistance coefficient
                CD = 0.96 
        elif reference_crop == 'alfalfa':
            crop_dependent_factor = 66
            if rn > 0:
                g = 0.01 * rn
                # Bulk surface resistance and aerodynamic resistance coefficient
                CD = 0.25 
            else:
                g = 0.2 * rn
                # Bulk surface resistance and aerodynamic resistance coefficient
                CD = 1.7
            
            
        # decompose et0 to two terms to facilitate the calculation
        
        radiation = delta * (rn - g)
        
        denominator = delta + (gamma * (1 + (CD * u2)))
        
        radiation_term =(radiation) / (denominator * lambda_heat)
        
        # Aerodynamic Term
        aerodynamic_term = (gamma * ((crop_dependent_factor) / (ta_k)) * u2 * (es - ea)) / denominator
        
        
        et0 = radiation_term + aerodynamic_term
            
        # output result
        return et0
    
    @staticmethod
    def psychrometric_constant(elevation, ta_c):
        """
        Calculate the psychrometric constant for a given altitude.

        Parameters:
        altitude (float): Altitude above sea level in meters.

        Returns:
        float: Psychrometric constant in kPa/°C.
        """
        
        lambda_v = Lib.latent_heat_of_vaporization(ta_c)  # Latent heat of vaporization (MJ/kg)
        
        # Calculate atmospheric pressure based on altitude
        p = Lib.pressure(elevation)
        
        # Calculate psychrometric constant
        gamma = (Lib.CP * p) / (Lib.EPSILON * lambda_v)
        
        return gamma
    
    @staticmethod
    def solar_altitude_angle(latitude, doy, hod):
        """
        Calculate the solar altitude angle based on latitude, day of the year, and local solar time.

        Parameters:
        - latitude: float, latitude of the observer in degrees
        - day_of_year: int, day of the year (1 through 365 or 366)
        - local_solar_time: float, local solar time in hours (solar noon is 12.0)

        Returns:
        - solar_altitude: float, solar altitude angle in degrees
        """
        # Convert latitude to radians
        latitude_rad = math.radians(latitude)

        # Calculate solar declination
        declination = 23.45 * math.sin(math.radians((360 / 365) * (doy - 81)))

        # Convert declination to radians
        declination_rad = math.radians(declination)

        # Calculate the hour angle
        hour_angle = 15 * (hod - 12)  # degrees from solar noon
        hour_angle_rad = math.radians(hour_angle)

        # Calculate the solar altitude angle
        sin_alpha = math.sin(latitude_rad) * math.sin(declination_rad) + \
                    math.cos(latitude_rad) * math.cos(declination_rad) * math.cos(hour_angle_rad)

        # Arcsine to get the angle in radians and then convert to degrees
        solar_altitude_rad = math.asin(sin_alpha)
        solar_altitude = math.degrees(solar_altitude_rad)

        return solar_altitude
    
    @staticmethod
    def pressure(z):
        ''' Calculates the barometric pressure above sea level.

        Parameters
        ----------
        z: float
            height above sea level (m).

        Returns
        -------
        p: float
            air pressure (Kpa).'''
            
        P0 = 101.325  # Standard atmospheric pressure at sea level in kPa
        L = 0.0065    # Standard lapse rate in °C/m
        p = P0 * (((293 - (L * z)) / (293)) ** 5.26)

        return p
    
    def latent_heat_of_vaporization(ta_c):
        """
        Estimate the latent heat of vaporization of water as a function of temperature, in MJ/kg.
        
        Parameters:
        - temp_celsius: float, temperature in degrees Celsius
        
        Returns:
        - lambda_v: float, latent heat of vaporization in MJ/kg
        """
        # Constants for water vaporization (values in J/g)
        # This approximation assumes a linear decrease from 2501.3 J/g at 0°C to 2264.7 J/g at 100°C
        
        # Adjust latent heat based on temperature
        lambda_v = 2.501 - (0.002361 * ta_c)
        return lambda_v
    
    @staticmethod
    def density_of_water(t_c):
        """
        density of air-free water ata pressure of 101.325kPa
        :param t_c: temperature in cellsius
        :return:
        density of water (kg m-3)
        """
        rho_w = (999.83952 + 16.945176 * t_c - 7.9870401e-3 * t_c**2
                - 46.170461e-6 * t_c**3 + 105.56302e-9 * t_c**4
                - 280.54253e-12 * t_c**5) / (1 + 16.897850e-3 * t_c)

        return rho_w
    
    @staticmethod
    def flux_2_evapotranspiration(flux, t_c=20, time_domain=1):
        '''Converts heat flux units (W m-2) to evaporation rates (mm time-1) to a given temporal window

        Parameters
        ----------
        flux : float or numpy array
            heat flux value to be converted,
            usually refers to latent heat flux LE to be converted to ET
        t_c : float or numpy array
            environmental temperature in Kelvin. Default=20 Celsius
        time_domain : float
            Temporal window in hours. Default 1 hour (mm h-1)

        Returns
        -------
        et : float or numpy array
            evaporation rate at the time_domain. Default mm h-1
        '''
        # Calculate latent heat of vaporization
        lambda_ = Lib.latent_heat_of_vaporization(t_c) * 10e6  # J kg-1
        # Density of water
        rho_w = Lib.density_of_water(t_c)  # kg m-3
        et = flux / (rho_w * lambda_)  # m s-1
        # Convert instantaneous rate to the time_domain rate
        et = et * 1e3 * time_domain * 3600.  # mm
        return et
    
    @staticmethod
    def saturation_vapor_pressure(ta_c):
        """
        Calculate saturation vapor pressure (es) in kPa given temperature (T) in Celsius
        
        """
        # Calculate saturation vapor pressure (es) using Magnus-Tetens formula
        es = 0.6108 * math.exp((17.27 * ta_c) / (ta_c + 237.3))
        return es
    
    @staticmethod
    def actual_vapor_pressure(ta_c, rh):
        """
        Calculate actual vapor pressure (ea) in kPa given temperature (T) in Celsius
        and relative humidity (RH) as a percentage.
        """
        
        ea = (rh / 100) * Lib.saturation_vapor_pressure(ta_c)
        return ea
    
    @staticmethod
    def stephan_boltzmann(t_c, freq='h'):
        '''Calculates the total energy radiated by a blackbody.

        Parameters
        ----------
        t_c: float
            body temperature (Celsius)

        Returns
        -------
         : float
            Emitted radiance (W m-2)'''
        
        t_k = t_c + 273.15
            
        if freq == 'h':
            # Stephan Boltzmann constant (W m-2 K-4)
            SB = 2.04e-10
        elif freq == 'd':
            SB = 4.903e-9 # Stefan-Boltzmann constant in MJ/K4/m2/day
            
        return SB * (t_k ** 4)
    
    def estimate_r_li(ta_c, ts_c):
        """
        Calculate actual vapor pressure (e) given temperature (T) in Celsius
        and relative humidity (RH) as a percentage.
        """
        
        r_li = f * self.stephan_boltzmann(ts_c)
        
        return r_li
    
    @staticmethod
    def cloudness_factor(rso, rs):
        """
        Calculate actual vapor pressure (e) given temperature (T) in Celsius
        and relative humidity (RH) as a percentage.
        """
        kt = rs / rso if rso > 0 else 0  # Avoid division by zero
        f = (1.35 * kt) - 0.35
        
        if f < 0:
            f = 0.595
        elif f > 1:
            f = 1
        
        return f
    
    @staticmethod
    def rso(ra, elevation):
        """
        Calculate clear-sky solar radiation based on extraterrestrial radiation and altitude.
        
        Parameters:
        - Ra: float, extraterrestrial radiation in MJ/m²/day
        - altitude: float, elevation above sea level in meters
        
        Returns:
        - Rso: float, clear-sky solar radiation in MJ/m²/day
        """
        rso = (0.75 + (2e-5 * elevation)) * ra
        return rso
    
    @staticmethod
    def inverse_relative_distance_factor(doy):
        """
        Calculate the inverse relative distance factor (dr) for Earth-Sun based on the day of the year.
        
        Parameters:
        - day_of_year: int, the day of the year (1 to 365 or 366 for a leap year)
        
        Returns:
        - dr: float, the inverse relative distance factor
        
        Description:
        This function uses the cosine function to calculate the Earth-Sun distance variation effect.
        """
        return 1 + (0.033 * math.cos((2 * math.pi * doy) / 365))
    
    @staticmethod
    def net_longwave_radiation(ta_c, rh, rs, rso, epsilon=0.95):
        """
        Calculate the net longwave radiation, adjusting sky temperature based on empirical methods.
        
        Parameters:
        Ta (float): Air temperature in degrees Celsius.
        rh (float): Relative humidity in percent.
        Rs (float): Actual solar radiation in W/m2.
        Rso (float): Clear sky solar radiation in W/m2.
        epsilon (float): Emissivity of the surface (default 0.95).

        Returns:
        float: Net longwave radiation in W/m2.
        """
        sigma = 5.67e-8  # Stefan-Boltzmann constant in W/m2/K4
        ta_k = ta_c + 273.15  # Convert air temperature from Celsius to Kelvin

        # Calculate actual vapor pressure
        ea = 0.6108 * math.exp((17.27 * ta_c) / (ta_c + 237.3)) * (rh / 100)

        # Adjust sky temperature based on emissivity and air temperature
        sky_emissivity = 0.787 + 0.764 * math.log10(ea)
        Tsky_k = sky_emissivity * ta_k

        # Surface temperature approximated by air temperature (adjust if necessary)
        ts_k = ta_k
        
        # Estimate effective sky temperature
        # Clearness index
        kt = rs / rso if rso > 0 else 0  # Avoid division by zero
        # Calculate the net longwave radiation
        rn_l = sigma * (Tsky_k**4 - ts_k**4) * epsilon * ((1.35 * kt) - 0.35)

        return rn_l
    
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
        
        et0 = 0.0023 * (ta_mean + 17.8) * ((ta_max - ta_min) ** 0.5) * 0.408 * ra

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
    
<<<<<<< HEAD
    @staticmethod    
    def relative_humidity_magnus(ta_c, dew_point_c):
        """
        Estimate relative humidity given dew point and air temperature using the Magnus formula.
        
        :param ta_c: Air temperature in Celsius
        :param dew_point_c: Dew point temperature in Celsius
        :return: Relative humidity in percentage
        """
        # Constants for the formula
        a = 17.62
        b = 243.12
        
        # Calculate alpha for dew point
        alpha_dp = (a * dew_point_c) / (b + dew_point_c)
        
        # Calculate alpha for temperature
        alpha_t = (a * ta_c) / (b + ta_c)
        
        # Calculate relative humidity
        rh = 100 * (math.exp(alpha_dp) / math.exp(alpha_t))
        
        return rh
=======
    @staticmethod
    def slope_vapor_pressure_curve(t_c, method='standard'):
        """
        Calculate the slope of the saturation vapor pressure curve at a given temperature.
        
        Parameters:
        - temp_celsius: float, temperature in degrees Celsius
        
        Returns:
        - delta: float, slope of the vapor pressure curve in kPa/°C
        """
        # Calculate saturation vapor pressure at the current temperature
        es = Lib.saturation_vapor_pressure(t_c)
        
        # Calculate the slope of the vapor pressure curve
        delta = (4098 * es) / ((t_c + 237.3) ** 2)
        return delta
    
    @staticmethod
    def calculate_slope_of_saturation_vapor_pressure(ta_c):
        # Constants
        L = 2260 * 1000  # Latent heat of vaporization for water (J/kg)
        R_v = 461.5      # Specific gas constant for water vapor (J/(kg*K))
        ta_k = ta_c + 273.15  # Convert Celsius to Kelvin
        
        # Tetens formula to approximate saturation vapor pressure in hPa
        e_s = 6.112 * math.exp((17.67 * ta_c) / (ta_c + 243.5))
        
        # Clausius-Clapeyron equation to calculate the slope (d es/dT)
        slope = (L * e_s) / (R_v * ta_k**2)
        
        # Convert slope from Pa/K to hPa/K for practical purposes
        slope_hPa_per_K = slope / 100
        
        return slope_hPa_per_K
    
    @staticmethod
    def extraterrestrial_radiation_hourly(doy, latitude, longitude, hod, standard_meridian):
        """
        Calculate hourly extraterrestrial radiation (Ra) using Duffie and Beckman's approach and G_sc in MJ/m²/min.
        
        Parameters:
        - day_of_year: int, day of the year (1-365 or 366)
        - latitude: float, latitude in degrees
        - longitude: float, local longitude in degrees
        - standard_meridian: float, longitude of the standard time meridian for the time zone
        - local_time: float, local standard time hour (24-hour format)
        
        Returns:
        - Ra_h: float, hourly extraterrestrial radiation in MJ/m^2
        """
        # Convert latitude and longitude from degrees to radians
        latitude_rad = math.radians(latitude)
        
        # Solar declination in radians
        delta = Lib.solar_declination(doy)
        
        # Equation of Time in minutes
        B = ((2 * Lib.PI) / 364) * (doy - 81)
        EOT = 0.1645 * math.sin(2 * B) - 0.1255 * math.cos(B) - 0.025 * math.sin(B)
        
        # Adjust local time to solar time
        omega = (Lib.PI/12) * (((hod - 0.5) - ((4/60) * (longitude - standard_meridian) + EOT)) - 12 )
        
        # Calculate solar time angles at the start and end of the hour
        omega_1 = omega - (0.5 * (Lib.PI/2))
        omega_2 = omega + (0.5 * (Lib.PI/2))
        
        # Hourly extraterrestrial radiation calculation
        Ra_h = ((12 * 60) / Lib.PI) * Lib.SOLAR_CONSTANT * Lib.inverse_relative_distance_factor(doy) * ((omega_2 - omega_1) * math.sin(latitude_rad) * math.sin(delta) + math.cos(latitude_rad) * math.cos(delta) * (math.sin(omega_2) - math.sin(omega_1)))
        
        return Ra_h
    
    @staticmethod
    def solar_declination(doy):
        """
        Calculate the solar declination in radians for a given day of the year using a precise trigonometric model.

        Parameters:
        - doy: int, day of the year (1 through 365 or 366)

        Returns:
        - declination: float, solar declination in radians
        """
        # Convert day of the year to radians within the sine function
        declination_radians = 0.409 * math.sin(((2 * math.pi * doy) / 365) - 1.39)

        return declination_radians
  
    @staticmethod
    def extraterrestrial_radiation(lat, doy):
        """
        Calculate extraterrestrial radiation (Ra) for a given latitude and day of the year.

        Parameters:
        lat (float): Latitude in degrees. Positive for the northern hemisphere, negative for southern.
        doy (int): Day of the year (1 through 365 or 366).

        Returns:
        float: Extraterrestrial radiation in MJ/m^2/day.
        """
        
        # Convert latitude to radians
        latitude = math.radians(lat)
        
        # Calculate the inverse relative distance Earth-Sun (dr)
        dr = 1 + (0.033 * math.cos((2 * math.pi * doy) / 365))
        
        # Calculate the solar declination (δ):
        delta = 0.409 * math.sin(((2 * math.pi * doy) / 365) - 1.39)
        
        # Calculate the sunset hour angle
        ws = math.acos(-math.tan(latitude) * math.tan(delta))
        
        # Calculate the extraterrestrial radiation
        ra = ((24 * 60) / math.pi) * Lib.SOLAR_CONSTANT * dr * ((ws * math.sin(latitude) * math.sin(delta)) + (math.cos(latitude) * math.cos(delta) * math.sin(ws)))
        
        return ra
    
    @staticmethod
    def et0_priestley_taylor(row):
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
        
        ta_c = row['ta_mean']
        
        seconds_per_day = 43200 # 43200 for 12 hours number of seconds in a day 86400 for 24 hours
        
        # Conversion from W/m² to kJ/m²/day
        rs_kj_per_m2 = (row['rs_mean'] * seconds_per_day) / 1000  # Convert joules to kilojoules
        
        #elevation = row['elevation']
        GHO = Lib.DENSITY_OF_WATER
        lambda_v = 2266
        DELTA = 4.95e-4
        
        
        if ta_c < 0:
            slope = 0.3405 * (math.exp(0.06642 * ta_c))
        else:
            slope = 0.3221 * (math.exp(0.0803 * (ta_c ** 0.8876)))
        
        et0 = ((1.3) / (lambda_v * GHO)) * ((slope)/(slope + DELTA)) * rs_kj_per_m2
        return et0 * 1000
    
    @staticmethod
    def et0_priestley_taylor_hourly(row, ta_column_name, rs_column_name):
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
        
        ta_c = row[ta_column_name]
        
        seconds_per_day = 3600 # 43200 for 12 hours number of seconds in a day 86400 for 24 hours
        
        # Conversion from W/m² to kJ/m²/day
        rs_kj_per_m2 = (row[rs_column_name] * seconds_per_day) / 1000  # Convert joules to kilojoules
        
        #elevation = row['elevation']
        GHO = Lib.DENSITY_OF_WATER
        lambda_v = 2266
        DELTA = 4.95e-4
        
        
        if ta_c < 0:
            slope = 0.3405 * (math.exp(0.06642 * ta_c))
        else:
            slope = 0.3221 * (math.exp(0.0803 * (ta_c ** 0.8876)))
        
        et0 = ((1.3) / (lambda_v * GHO)) * ((slope)/(slope + DELTA)) * rs_kj_per_m2
        return et0 * 1000
>>>>>>> d1504203b8a2b4007b8b84ba1326e20861c6b47e
