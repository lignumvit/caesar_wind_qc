from scipy.optimize import curve_fit
import math
import copy
import os
import netCDF4
import pandas as pd
from datetime import datetime, timedelta
from fnmatch import fnmatch
import numpy as np
from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import row
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show
from bokeh.models import Title, CustomJS, Select, TextInput, Button, LinearAxis, Range1d, FuncTickFormatter, HoverTool, ColumnDataSource
from bokeh.models.formatters import DatetimeTickFormatter
from bokeh.models.tickers import DatetimeTicker
from bokeh.palettes import Category10
import itertools
from scipy import signal

# Thresholds for determining what data can be considered in AoA coef determination (e.g. straight-and-level).
# Used by mask_straight_and_level.
#max_vspd = 0.3 # could use 1 m/s
max_vspd = 8 # m/s
max_roll = 1 # deg
min_tas = 100 # m/s
# min vspd used to isolate climb situations
min_vspd = 2.5 

# constants from Nimbus
mol_wgt_dry_air = 28.9637 # kg/kmol
R0 = 8314.462618 # J/kmol/K
Rd = R0/mol_wgt_dry_air
Cpd = 7.0/2.0*Rd
Cvd = 5.0/2.0*Rd

read_vars = ['ADIFR', 'BDIFR', 'ADIFRTEMP', 'BDIFRTEMP', # adifr, bdifr
             'QCF', 'QCFR', 'QCR', 'QC_A', 'QC_A2', 'QCFRC',      # raw, dynamic pressures (q)
             'GGLAT','GGLON','GGNSAT','GGQUAL','GGSPD','GGTRK', # GPS variables
             'PSFD', 'PSFRD', 'PSXC', # static pressures
             'VEW', 'VNS', 'VSPD', 'GGVEW', 'GGVNS', 'GGVSPD', 'VEWC', 'VNSC', # aircraft velocities, raw and blended
             'UI', 'UIC', 'VI', 'VIC', 'WI', 'WIC', # winds, both uncorrected and GPS-corrected
             'PALTF', 'PALT', 'GGALT', 'ALT', # altitudes
             'TASF', 'TASFR', 'TASR', 'TAS_A', 'TAS_A2', 'MACHX', # speeds
             'PITCH', 'ROLL', 'THDG', # attitude
             'AKRD', 'SSLIP', # flow angles
             'RHUM', 'RICE', 'ATX', 'BNORMA', 'BLATA', 'BLONGA', 'WDC',
            ]

def hms_to_sfm(hms_str: str):
    hh = int(hms_str[0:2])
    mm = int(hms_str[3:5])
    ss = int(hms_str[6:8])
    return hh*3600 + mm*60 + ss

def to_sfm(beg_end_times: list):
    sfm_list = []
    for pair in beg_end_times:
        sfm_list.append([hms_to_sfm(pair[0]), hms_to_sfm(pair[1])])
    return sfm_list

def hms_dict_to_sfm(beg_end_times: dict):
    sfm_dict = {}
    for flight, pairs in beg_end_times.items():
        sfm_dict[flight] = to_sfm(pairs)
        #print(f"flight: {flight}, pairs: {pairs}, list_sfm: {list_sfm}")
    return sfm_dict
         
        

def mask_out_times(df, beg_time=-1, end_time=-1):
    if beg_time != -1 and end_time != -1:
        mask = np.logical_or(df["Time"].to_numpy() <= beg_time,
                             df["Time"].to_numpy() >= end_time)
    else:
        n_pts = len(nc["Time"].to_numpy().squeeze())
        mask = np.ones(n_pts)
    return mask

def mask_in_times(df, beg_time=-1, end_time=-1):
    if beg_time != -1 and end_time != -1:
        mask = np.logical_and(df["Time"].to_numpy() >= beg_time,
                              df["Time"].to_numpy() <= end_time)
    else:
        n_pts = len(nc["Time"].to_numpy().squeeze())
        mask = np.ones(n_pts)
    return mask


def mask_straight_and_level(df):
    roll = df['ROLL']
    vspd = df['GGVSPD']
    tas = df['TASFR']
    mask = np.abs(roll) < max_roll
    mask = np.logical_and(mask, np.abs(vspd) < max_vspd)
    mask = np.logical_and(mask, tas > min_tas)
    nan_check_vars = ['QCF', 'PSFD', 'ADIFR', 'GGVSPD', 'TASFR', 'PITCH', 'AKRD']
    for var in nan_check_vars:
        mask = np.logical_and(mask, np.isfinite(df[var]))
    return mask.to_numpy()

def mask_ascent(df):
    roll = df['ROLL']
    vspd = df['GGVSPD']
    tas = df['TASFR']
    mask = np.abs(roll) < max_roll
    mask = np.logical_and(mask, vspd > min_vspd)
    mask = np.logical_and(mask, tas > min_tas)
    nan_check_vars = ['QCF', 'PSFD', 'ADIFR', 'GGVSPD', 'TASFR', 'PITCH', 'AKRD']
    for var in nan_check_vars:
        mask = np.logical_and(mask, np.isfinite(df[var]))
    return mask.to_numpy()

def mask_descent(df):
    roll = df['ROLL']
    vspd = df['GGVSPD']
    tas = df['TASFR']
    mask = np.abs(roll) < max_roll
    mask = np.logical_and(mask, vspd < -min_vspd)
    mask = np.logical_and(mask, tas > min_tas)
    nan_check_vars = ['QCF', 'PSFD', 'ADIFR', 'GGVSPD', 'TASFR', 'PITCH', 'AKRD']
    for var in nan_check_vars:
        mask = np.logical_and(mask, np.isfinite(df[var]))
    return mask.to_numpy()

def mask_flying(df):
    tas = df['TASFR']
    mask = tas > 80.
    return mask.to_numpy()


def calc_mach(q: np.array, ps: np.array):
    # a function to calculate mach given a dynamic (q) and static (ps) pressure
    return (5*(((ps + q)/ps)**(Rd/Cpd) - 1))**0.5

def fit_func_four(x: np.array, a: float, b: float, c: float, d: float):

    # x: a 2 by N array, where x[0,:] = ADIFR/QCF and x[1,:] = MACH
    # a, b, c: fit coefficients
    ratio = x[0,:]
    mach = x[1,:]
    rho = x[2,:]
    return a + rho*b + ratio*(c + d*mach)

def fit_func(x: np.array, a: float, b: float, c: float):

    # x: a 2 by N array, where x[0,:] = ADIFR/QCF and x[1,:] = MACH
    # a, b, c: fit coefficients
    ratio = x[0,:]
    mach = x[1,:]
    return a + ratio*(b + c*mach)

def simple_fit_func(ratio, a, b):
    # ratio: an N element array from ADIFR/QCF or BDIFR/QCF
    # a, b: fit coefficients
    return a + ratio*b

def open_nc(data_dir):
    # get file names
    # only partial data on return ff's, so those are excluded. Only included the first three
    ffnames = sorted(["CAESARff01.nc", "CAESARff02.nc", "CAESARff03.nc", "CAESARff04.nc", "CAESARff05.nc", "CAESARff06.nc"])
    tfnames = sorted([fname for fname in os.listdir(data_dir) if fnmatch(fname, "*tf??.nc")])
    rfnames = sorted([fname for fname in os.listdir(data_dir) if fnmatch(fname, "*rf??.nc")])
    allfnames = ffnames + tfnames + rfnames
    allfnames = sorted(allfnames)
    
    print(f"Found {len(ffnames)} ferry flights, {len(tfnames)} test flights, and {len(rfnames)} research flights")
    print("Opening all flight NetCDF Files")
    nc_dict = {}
    for fname in allfnames:
        stem = fname.split('.')[0]
        if stem[-1] == "h":
            flname = stem[-5:-1]
    
        else:
            flname = stem[-4:]
    
        try:
            nc_dict[flname] = netCDF4.Dataset(data_dir + "/" + fname)
    
        except Exception as e:
            print(f"Could not read {fname} netcdf.")
            print(e)
    
    # try to get global attributes from the netcdf file if they are present
    # determine preliminary or final status
    try:
        proc_status = nc_dict['ff01'].getncattr('WARNING')
        print(proc_status)
    except:
        proc_status = 'final'
    
    # determine the NIDAS version
    try:
        nidas = nc_dict['ff01'].getncattr('NIDASrevision')
        print('NIDAS version: ' + nidas)
    except Exception as e:
        print(e)
    
    # determine the NIMBUS version
    try:
        nimbus = nc_dict['ff01'].getncattr('RepositoryRevision')
        print('NIMBUS version: ' + nimbus)
    except Exception as e:
        print(e)
    
    # determine the processing date and time
    try:
        proc_date = nc_dict['ff01'].getncattr('date_created')
        print('Processing Date & Time: ' + proc_date)
    except Exception as e:
        print(e)

    return nc_dict

def read_nc(nc: netCDF4._netCDF4.Dataset):
    # sometimes the netcdf4 api produces an issue with big-endian buffer on little-endian compiler
    byte_swap = False
    
    # create empty placeholders for asc, histo_asc and units
    data = {}
    units = {}
    
    # use the netcdf4 api to get the netcdf data into a dataframe
#    try:
        
    # loop over keys in netCDF file and organize
    #for i in nc.variables.keys():
    for i in read_vars:
        try:
            output = nc[i][:]
            data[i] = pd.DataFrame(output)
            units_var = nc.variables[i].getncattr('units')
            units[i] = pd.Series(units_var)
            data[i].columns = pd.MultiIndex.from_tuples(zip(data[i].columns, units[i]))

        except Exception as e:
            print(e)

    # add times
    i = 'Time'
    output = nc[i][:]
    data[i] = pd.DataFrame(output)
    units_var = nc.variables[i].getncattr('units')
    units[i] = pd.Series(units_var)
    data[i].columns = pd.MultiIndex.from_tuples(zip(data[i].columns, units[i]))

    # concatenate the dataframe
    data = pd.concat(data, axis=1, ignore_index=False)
    # clean up the dataframe by dropping some of the multi-index rows
    data.columns = data.columns.droplevel(1)
    data.columns = data.columns.droplevel(1)

    # add a datetime-type time as well
    data['datetime'] = [timedelta(seconds=int(time)) for time in data['Time']]

    return data

def plot_track(df: pd.DataFrame, mask: pd.Series = None, title: str =''):
    if mask is None:
        mask = np.ones(len(df))
    # get latitude and longitude from dataframe
    latitude = df["GGLAT"].to_numpy().squeeze()
    longitude = df["GGLON"].to_numpy().squeeze()
    
    # update to mercator projection
    k = 6378137
    longitude = longitude * (k * np.pi/180.0)
    latitude = np.log(np.tan((90 + latitude) * np.pi/360.0)) * k
    
    # create the plot layout and add axis labels
    try:
        plot = figure(width=600, height=600, title=title, x_axis_type="mercator", y_axis_type="mercator") 
        plot.add_layout(Title(text="Longitude [Degrees]", align="center"), "below")
        plot.add_layout(Title(text="Latitude [Degrees]", align="center"), "left")
        
        # add the flight track in yellow and add Esri World Imagery as the basemap
        if sum(mask) != len(latitude):
            lat = df["GGLAT"][mask].to_numpy().squeeze()
            lon = df["GGLON"][mask].to_numpy().squeeze()
            lon = lon * (k * np.pi/180.0)
            lat = np.log(np.tan((90 + lat) * np.pi/360.0)) * k
            plot.multi_line([longitude,lon],[latitude,lat], color=["yellow","red"])
        else:
            plot.line(longitude,latitude, color="yellow")
    
        plot.add_tile("Esri World Imagery", retina=True)
        show(plot)
    except Exception as e:
        print(e)

# function definition for creating generic timeseries plot
def format_ticks(plot):
    plot.xaxis.formatter=DatetimeTickFormatter(days =['%h:%m'], hours="%h:%m", minutes="%h:%m",hourmin = ['%h:%m'])             

def plot_time_series_aoa(df: pd.DataFrame, mask=None, title=''):
    if mask is None:
        mask = [True for i in range(len(df))]
    # generate the altitude, heading and gps quality plots
    # altitude plot
    height = 150
    width = 1000
    colors = itertools.cycle(Category10[8])
    p1 = figure(width=width, height=height, title=title)
    p1.add_layout(Title(text="Altitude [m]", align="center"), "left")
    p1.line(df['datetime'][mask], df['GGALT'][mask], color=next(colors), legend_label='GGALT')    
    format_ticks(p1)

    colors = itertools.cycle(Category10[8])
    p2 = figure(width=width, height=height, x_range=p1.x_range, title=title)
    p2.add_layout(Title(text="Altitude [m]", align="center"), "left")
    p2.line(df['datetime'][mask], df['PALT'][mask], color=next(colors), legend_label='PALT')
    p2.line(df['datetime'][mask], df['ALT'][mask], color=next(colors), legend_label='ALT')
    format_ticks(p2)

    colors = itertools.cycle(Category10[8])
    p3 = figure(width=width, height=height, x_range=p1.x_range)
    p3.add_layout(Title(text="Static Pres. [hPa]", align="center"), "left")
    p3.line(df['datetime'][mask], df['PSFD'][mask], color=next(colors), legend_label='PSFD')
    p3.line(df['datetime'][mask], df['PSFRD'][mask], color=next(colors), legend_label='PSFRD')
    p3.line(df['datetime'][mask], df['PSXC'][mask], color=next(colors), legend_label='PSXC')
    format_ticks(p3)

    
    colors = itertools.cycle(Category10[8])
    p4 = figure(width=width, height=height, x_range=p1.x_range)
    p4.add_layout(Title(text="True Airspeed [m/s]", align="center"), "left")
    p4.line(df['datetime'][mask], df['TASF'][mask], color=next(colors), legend_label='TASF')
    p4.line(df['datetime'][mask], df['TASFR'][mask], color=next(colors), legend_label='TASFR')
    p4.line(df['datetime'][mask], df['TASR'][mask], color=next(colors), legend_label='TASR')
    #p4.line(df['datetime'][mask], df['TAS_A'][mask], color=next(colors), legend_label='TAS_A')
    #p4.line(df['datetime'][mask], df['TAS_A2'][mask], color=next(colors), legend_label='TAS_A2')
    format_ticks(p4)

    colors = itertools.cycle(Category10[8])
    p5 = figure(width=width, height=height, x_range=p1.x_range)
    p5.add_layout(Title(text="Diff. Pressure [hPa]", align="center"), "left")
    p5.line(df['datetime'][mask], df['ADIFR'][mask], color=next(colors), legend_label='ADIFR')
    #p5.line(df['datetime'][mask], df['BDIFR'][mask], color=next(colors), legend_label='BDIFR')
    format_ticks(p5)

    colors = itertools.cycle(Category10[8])
    p6 = figure(width=width, height=height, x_range=p1.x_range)
    p6.add_layout(Title(text="Diff. Pressure [hPa]", align="center"), "left")
    p6.line(df['datetime'][mask], df['QCF'][mask], color=next(colors), legend_label='QCF')
    p6.line(df['datetime'][mask], df['QCFR'][mask], color=next(colors), legend_label='QCFR')
    p6.line(df['datetime'][mask], df['QCR'][mask], color=next(colors), legend_label='QCR')
    p6.line(df['datetime'][mask], df['QCFRC'][mask], color=next(colors), legend_label='QCFRC')
    format_ticks(p6)

    colors = itertools.cycle(Category10[8])
    p7 = figure(width=width, height=height, x_range=p1.x_range)
    p7.add_layout(Title(text="Roll [deg]", align="center"), "left")
    p7.line(df['datetime'][mask], df['ROLL'][mask], color=next(colors), legend_label='ROLL')
    format_ticks(p7)

    colors = itertools.cycle(Category10[8])
    p8 = figure(width=width, height=height, x_range=p1.x_range)
    p8.add_layout(Title(text="Time [UTC]", align="center"), "below")
    p8.add_layout(Title(text="Wind Spd. [m/s]", align="center"), "left")
    p8.line(df['datetime'][mask], df['WIC'][mask], color=next(colors), legend_label='WIC')
    format_ticks(p8)

    colors = itertools.cycle(Category10[8])
    p9 = figure(width=width, height=height, x_range=p1.x_range)
    p9.add_layout(Title(text="Vert. Spd. [m/s]", align="center"), "left")
    p9.line(df['datetime'][mask], df['GGVSPD'][mask], color=next(colors), legend_label='GGVSPD')
    p9.line(df['datetime'][mask], df['VSPD'][mask], color=next(colors), legend_label='VSPD')
    format_ticks(p9)


    p = gridplot([[p2], [p4], [p5], [p7], [p9], [p8]])
    show(p)

def plot_time_series_pitch(df_v0: pd.DataFrame, df_v1: pd.DataFrame, mask_v0=None, mask_v1=None, title=''):
    if mask_v0 is None:
        mask_v0 = [True for i in range(len(df_v0))]
    if mask_v1 is None:
        mask_v1 = [True for i in range(len(df_v1))]
    # generate the altitude, heading and gps quality plots
    # altitude plot
    height = 200
    width = 1000

    ht = HoverTool(tooltips=[('time', '@x{%H:%M:%S}'), ('y', '@y')], formatters={'@x': 'datetime'})

    colors = itertools.cycle(Category10[8])
    p1 = figure(width=width, height=height, title=title)
    p1.add_layout(Title(text="Vert. Spd. [m/s]", align="center"), "left")
    p1.line(df_v0['datetime'][mask_v0], df_v0['GGVSPD'][mask_v0], color=next(colors), legend_label='GGVSPD_v0')
    p1.line(df_v1['datetime'][mask_v1], df_v1['GGVSPD'][mask_v1], color=next(colors), legend_label='GGVSPD_v1')
    p1.line(df_v0['datetime'][mask_v0], df_v0['WIC'][mask_v0], color=next(colors), legend_label='W_v0')
    p1.line(df_v1['datetime'][mask_v1], df_v1['WIC'][mask_v1], color=next(colors), legend_label='W_v1')
    p1.add_tools(ht)
    format_ticks(p1)

    aoa_corr_v0 = -np.arcsin(df_v0['GGVSPD']/df_v0['TASFR'])*180./math.pi
    aoa_ref_v0 = df_v0['PITCH'] + aoa_corr_v0
    aoa_corr_v1 = -np.arcsin(df_v1['GGVSPD']/df_v1['TASFR'])*180./math.pi
    aoa_ref_v1 = df_v1['PITCH'] + aoa_corr_v1

    colors = itertools.cycle(Category10[8])
    p2 = figure(width=width, height=height, title=title)
    p2.add_layout(Title(text="AKRD [deg]", align="center"), "left")
    p2.line(df_v0['datetime'][mask_v0], aoa_ref_v0[mask_v0], color=next(colors), legend_label='AoARef')
    p2.line(df_v0['datetime'][mask_v0], df_v0['AKRD'][mask_v0], color=next(colors), legend_label='AKRD_v0')
    p2.line(df_v1['datetime'][mask_v1], df_v1['AKRD'][mask_v1], color=next(colors), legend_label='AKRD_v1')
    p2.add_tools(ht)
    format_ticks(p2)

    colors = itertools.cycle(Category10[8])
    p3 = figure(width=width, height=height, title=title)
    p3.add_layout(Title(text="Altitude [m]", align="center"), "left")
    p3.line(df_v0['datetime'][mask_v0], df_v0['PALTF'][mask_v0], color=next(colors))
    p3.add_tools(ht)
    format_ticks(p3)

    colors = itertools.cycle(Category10[8])
    p4 = figure(width=width, height=height, title=title)
    p4.add_layout(Title(text="Wind [m/s]", align="center"), "left")
    p4.line(df_v1['datetime'][mask_v1], df_v1['UIC'][mask_v0], color=next(colors), legend_label='U')
    p4.line(df_v1['datetime'][mask_v1], df_v1['VIC'][mask_v0], color=next(colors), legend_label='V')
    p4.add_tools(ht)
    format_ticks(p4)



    p = gridplot([[p3], [p1], [p2], [p4] ])
    show(p)


class aoa_fit:
    # a fancy dataclass that computes quite a few things in the init
    def __init__(self, df: pd.DataFrame, mask: np.array, flight: str, leg: int = None, order=5, sampling_freq=2*math.pi/1, cutoff_freq=2*math.pi/30.):
        self.df = df
        self.mask = mask
        self.flight = flight
        self.leg = leg
        self.prev_coefs_three = [4.7532, 9.7908, 6.0781]
        self.prev_coefs_two = [5.661,14.826]
        self.varnamemap = {'datetime': 'dt', 'QCF': 'q', 'PSFD': 'ps', 'ADIFR': 'adifr', 'BDIFR': 'bdifr', 'SSLIP': 'sslip',
                           'GGVSPD': 'vspd', 'TASFR': 'tas', 'PITCH': 'pitch', 'AKRD': 'akrd', 'PALTF': 'paltf', 'THDG': 'hdg',
                           'GGALT': 'alt', 'ATX': 'tc', 'PSXC': 'p', 'UIC': 'u', 'VIC': 'v', 'WIC': 'w', 'GGVEW': 'up', 'GGVNS': 'vp', 'GGTRK': 'trk'}
        self.r2d = 180./math.pi
        # get basic vars needed

        for key, val in self.varnamemap.items():
            setattr(self, val, df[key][mask].to_numpy())
        self.mach = calc_mach(self.q, self.ps)
        self.aoa_corr = -np.arcsin(self.vspd/self.tas)*self.r2d
        self.aoa_ref = self.pitch + self.aoa_corr
        self.tk = self.tc + 273.15
        self.p = self.p*100 # hPa to Pa
        self.rho = self.p/Rd/self.tk
        
        # now, calculate reference AoS
        # do filtering
        self.order = order
        self.sampling_freq = sampling_freq
        self.cutoff_freq = cutoff_freq
        self.ratio = self.bdifr/self.q
        # design the filter
        sos = signal.butter(self.order, self.cutoff_freq, btype='lowpass', fs=self.sampling_freq, output='sos', analog=False)
        # compute response function
        self.f, self.h = signal.sosfreqz(sos,fs=self.sampling_freq)
        # now, actually do the filtering
        all_u = df['UIC'].to_numpy()
        all_v = df['VIC'].to_numpy()
        mean_u = np.nanmean(all_u)
        mean_v = np.nanmean(all_v)
        all_u = np.where(np.isfinite(all_u), all_u, mean_u)
        all_v = np.where(np.isfinite(all_v), all_v, mean_u)

        self.hdg_sm = signal.sosfiltfilt(sos, df['THDG'].to_numpy())[mask]
        self.u_sm = signal.sosfiltfilt(sos, all_u)[mask]
        self.v_sm = signal.sosfiltfilt(sos, all_v)[mask]
        self.up_sm = signal.sosfiltfilt(sos, df['GGVEW'].to_numpy())[mask]
        self.vp_sm = signal.sosfiltfilt(sos, df['GGVNS'].to_numpy())[mask]
        self.trk_sm = signal.sosfiltfilt(sos, df['GGTRK'].to_numpy())[mask]
        self.hdg_sm = signal.sosfiltfilt(sos, df['THDG'].to_numpy())[mask]
        self.wd_sm = signal.sosfiltfilt(sos, df['WDC'].to_numpy())[mask]
        self.ratio_sm = signal.sosfiltfilt(sos, df['BDIFR'].to_numpy()/df['QCF'].to_numpy())[mask]
        # now, compute aos_ref
        self.aos_corr = np.arctan2((self.up - self.u_sm),(self.vp-self.v_sm))*180/math.pi
        self.aos_corr = np.where(self.aos_corr < 0, self.aos_corr + 360, self.aos_corr)
        self.aos_ref = -self.hdg + self.aos_corr
        self.aos_ref = np.where(self.aos_ref < -200, self.aos_ref+360, self.aos_ref)

        self.calc_coefs()

    def calc_coefs(self):
        # first, fit two predictor, three parameter function
        ratio = self.adifr/self.q
        finite_mask = np.logical_and(np.logical_and(np.isfinite(ratio), np.isfinite(self.mach)),np.isfinite(self.rho))
        x = np.array([ratio[finite_mask], self.mach[finite_mask], self.rho[finite_mask]])
        self.coefs_four, pcov_four, odict_four, msg_four, ierr_four = curve_fit(fit_func_four, x, self.aoa_ref[finite_mask], full_output=True, method='trf')
        self.cond_no_four = np.linalg.cond(pcov_four)
        self.converged_four = not ierr_four == 0

        x = np.array([ratio[finite_mask], self.mach[finite_mask]])
        self.coefs_three, pcov_three, odict_three, msg_three, ierr_three = curve_fit(fit_func, x, self.aoa_ref[finite_mask], 
                                                                                     p0=self.prev_coefs_three, full_output=True, method='trf')
        self.cond_no_three = np.linalg.cond(pcov_three)
        self.converged_three = not ierr_three == 0

        # second, fit single predictor, two parameter function
        self.coefs_two, pcov_two, odict_two, msg_two, ierr_two = curve_fit(simple_fit_func, ratio[finite_mask], self.aoa_ref[finite_mask], 
                                                                           p0=self.prev_coefs_two, full_output=True, method='trf')
        self.cond_no_two = np.linalg.cond(pcov_two)
        self.converged_two = not ierr_two == 0

        # now with fitting done, predict aoa_ref
        self.akrd_four = fit_func_four(np.array([ratio, self.mach, self.rho]), *self.coefs_four)
        self.akrd_three = fit_func(np.array([ratio, self.mach]), *self.coefs_three)
        self.akrd_two = simple_fit_func(ratio, *self.coefs_two)

    def append(self, obj):
        obj_copy = copy.deepcopy(self)
        vars_to_append = list(self.varnamemap.values()) + ['mach', 'aoa_corr', 'aoa_ref', 'rho', 'tk']
        # append vars to the copied object
        for var in vars_to_append:
            setattr(obj_copy, var, np.concatenate((getattr(obj_copy,var), getattr(obj,var))))

        # now, recompute coefficients, now that the the constituent data has been added to
        obj_copy.calc_coefs()
        setattr(obj_copy, 'flight', 'appended')
        return obj_copy

    def print_coefs(self):
        if self.flight == "appended":
            print(f"Flight: {self.flight}, "
                  f"mach coefs: {self.coefs_three[0]:8.4f}; {self.coefs_three[1]:8.4f}; {self.coefs_three[2]:8.4f}, "
                  f"simple coefs: {self.coefs_two[0]:8.4f}; {self.coefs_two[1]:8.4f}")
            #print(f"    four coefs: {self.coefs_four[0]:8.4f}; {self.coefs_four[1]:8.4f}; {self.coefs_four[2]:8.4f}; {self.coefs_four[3]:8.4f}")
        else:            
            print(f"Flight: {self.flight}, Leg: {self.leg}: "
                  f"mach coefs: {self.coefs_three[0]:8.4f}; {self.coefs_three[1]:8.4f}; {self.coefs_three[2]:8.4f}, "
                  f"simple coefs: {self.coefs_two[0]:8.4f}; {self.coefs_two[1]:8.4f}")
            #print(f"    four coefs: {self.coefs_four[0]:8.4f}; {self.coefs_four[1]:8.4f}; {self.coefs_four[2]:8.4f}; {self.coefs_four[3]:8.4f}")


class aos_fit:
    # a fancy dataclass that computes quite a few things in the init
    def __init__(self, df: pd.DataFrame, mask: np.array, flight: str, leg: int = None, sampling_freq: float = 2*math.pi/1., cutoff_freq: float = 2.*math.pi/30, order: int = 5):
        # sampling_freq has units of 1./(samples per second)
        self.df = df
        self.mask = mask
        self.flight = flight
        self.leg = leg
        self.prev_coefs_three = [4.7532, 9.7908, 6.0781]
        self.prev_coefs_two = [5.661,14.826]
        self.varnamemap = {'datetime': 'dt', 'QCF': 'q', 'PSFD': 'ps', 'ADIFR': 'adifr', 'BDIFR': 'bdifr',
                           'GGVSPD': 'vspd', 'TASFR': 'tas', 'UIC': 'u', 'VIC': 'v',
                           'PITCH': 'pitch', 'ROLL': 'roll', 'THDG': 'hdg', 'AKRD': 'akrd', 'SSLIP': 'sslip', 'PALTF': 'paltf',
                           'GGALT': 'alt', 'GGVEW': 'up', 'GGVNS': 'vp', 'GGTRK': 'trk', 'ATX': 'tc', 'PSXC': 'p',
                           'UIC': 'u', 'VIC': 'v', 'WIC': 'w', 'WDC': 'wd'}
        self.r2d = 180./math.pi
        # get basic vars needed

        for key, val in self.varnamemap.items():
            setattr(self, val, df[key][mask].to_numpy())
        self.mach = calc_mach(self.q, self.ps)
        self.tk = self.tc + 273.15
        self.p = self.p*100 # hPa to Pa
        self.rho = self.p/Rd/self.tk

        # now, calculate reference AoS
        # do filtering
        self.order = order
        self.sampling_freq = sampling_freq
        self.cutoff_freq = cutoff_freq
        self.ratio = self.bdifr/self.q
        # design the filter
        sos = signal.butter(self.order, self.cutoff_freq, btype='lowpass', fs=self.sampling_freq, output='sos', analog=False)
        # compute response function
        self.f, self.h = signal.sosfreqz(sos,fs=self.sampling_freq)
        # now, actually do the filtering
        self.hdg_sm = signal.sosfiltfilt(sos, df['THDG'].to_numpy())[mask]
        self.u_sm = signal.sosfiltfilt(sos, df['UIC'].to_numpy())[mask]
        self.v_sm = signal.sosfiltfilt(sos, df['VIC'].to_numpy())[mask]
        self.up_sm = signal.sosfiltfilt(sos, df['GGVEW'].to_numpy())[mask]
        self.vp_sm = signal.sosfiltfilt(sos, df['GGVNS'].to_numpy())[mask]
        self.trk_sm = signal.sosfiltfilt(sos, df['GGTRK'].to_numpy())[mask]
        self.hdg_sm = signal.sosfiltfilt(sos, df['THDG'].to_numpy())[mask]
        self.wd_sm = signal.sosfiltfilt(sos, df['WDC'].to_numpy())[mask]
        self.ratio_sm = signal.sosfiltfilt(sos, df['BDIFR'].to_numpy()/df['QCF'].to_numpy())[mask]
        # now, compute aos_ref
        self.aos_corr = np.arctan2((self.up - self.u_sm),(self.vp-self.v_sm))*180/math.pi
        self.aos_corr = np.where(self.aos_corr < 0, self.aos_corr + 360, self.aos_corr)
        self.aos_ref = -self.hdg + self.aos_corr
        # simply the difference between where the aircraft is pointing and where it is going, not accounting for BDIFR
        self.aos_ref2 = self.trk - self.hdg

        #self.lat_spd = self.tas*np.sin(self.sslip*math.pi/180)
        self.lat_spd = self.tas*np.sin(self.aos_ref*math.pi/180)

        self.calc_coefs()

    def calc_coefs(self):
        # first, fit two predictor, three parameter function
        ratio = self.bdifr/self.q

        # fit single predictor, two parameter function
        self.coefs_two, pcov_two, odict_two, msg_two, ierr_two = curve_fit(simple_fit_func, ratio, self.aos_ref, 
                                                                           full_output=True, method='trf')
        self.cond_no_two = np.linalg.cond(pcov_two)
        self.converged_two = not ierr_two == 0

        # now with fitting done, predict aoa_ref
        self.sslip_fit = simple_fit_func(ratio, *self.coefs_two)

    def append(self, obj):
        obj_copy = copy.deepcopy(self)
        vars_to_append = list(self.varnamemap.values()) + ['mach', 'hdg_sm', 'u_sm', 'v_sm', 'aos_corr', 'aos_ref', 'aos_ref2', 'lat_spd', 'ratio']
        # append vars to the copied object
        for var in vars_to_append:
            setattr(obj_copy, var, np.concatenate((getattr(obj_copy,var), getattr(obj,var))))

        # now, recompute coefficients, now that the the constituent data has been added to
        obj_copy.calc_coefs()
        setattr(obj_copy, 'flight', 'appended')
        return obj_copy

    def print_coefs(self):
        if self.flight == "appended":
            print(f"Flight: {self.flight}, coefs: {self.coefs_two[0]:8.4f}; {self.coefs_two[1]:8.4f}")
        else:            
            print(f"Flight: {self.flight}, Leg: {self.leg}: simple coefs: {self.coefs_two[0]:8.4f}; {self.coefs_two[1]:8.4f}")

def plot_filter_response(aos_obj: aos_fit):
    height=300
    width=500

    cutoff = aos_obj.cutoff_freq/2./math.pi # in physical frequency, not angular
    cutoff = 1./cutoff # cutoff in period
    y = abs(aos_obj.h)
    f = aos_obj.f/2./math.pi
    period = 1./f

    colors = itertools.cycle(Category10[8])
    p = figure(width=width, height=height, title='Filter Reponse', x_range=(0,2*cutoff))
    p.add_layout(Title(text="Amplitude [dB]", align="center"), "left")
    p.add_layout(Title(text="Sampling Frequency [Hz]", align="center"), "below")
    p.line(period, y, color=next(colors), width=2, legend_label='Butterworth')
    p.line([cutoff, cutoff], [np.min(y),np.max(y)], color=next(colors), width=2, legend_label='Cutoff')
    p.legend.location = 'top_left'
    show(p)




def plot_aoa_obj(aoa_obj: aoa_fit, coefs: list[float]):
    # set up hover tool
    ht = HoverTool(tooltips=[('time', '@x{%H:%M:%S}'), ('y', '@y')], formatters={'@x': 'datetime'})

    x = np.array([aoa_obj.adifr/aoa_obj.q, aoa_obj.mach])
    cal_aoa = fit_func(x, *coefs)

    # generate the altitude, heading and gps quality plots
    height = 150
    width = 1000
    colors = itertools.cycle(Category10[8])
    p1 = figure(width=width, height=int(height*1.5))
    p1.add_layout(Title(text="AoA [deg]", align="center"), "left")
    #p1.line(aoa_obj.dt, aoa_obj.akrd, color=next(colors), legend_label='AOAPrev')
    p1.line(aoa_obj.dt, aoa_obj.aoa_ref, color=next(colors), legend_label='AOAREF')
    #p1.line(aoa_obj.dt, aoa_obj.akrd_three, color=next(colors), legend_label='AOAFit')
    p1.line(aoa_obj.dt, cal_aoa, color=next(colors), legend_label='AOACal')
    #p1.line(aoa_obj.dt, aoa_obj.akrd, color=next(colors), legend_label='AKRD')
    p1.line(aoa_obj.dt, cal_aoa-aoa_obj.aoa_ref, color=next(colors), legend_label='Diff')
    #p1.line(aoa_obj.dt, aoa_obj.akrd_two, color=next(colors), legend_label='AOAFitSimple')
    p1.legend.location = 'top_left'
    p1.add_tools(ht)
    format_ticks(p1)

    colors = itertools.cycle(Category10[8])
    p2 = figure(width=width, height=height, x_range=p1.x_range)
    p2.add_layout(Title(text="Diff. Pres. [hPa]", align="center"), "left")
    p2.line(aoa_obj.df['datetime'], aoa_obj.df['BDIFR'], color=next(colors), legend_label='BDIFR')
    p2.line(aoa_obj.df['datetime'], aoa_obj.df['ADIFR'], color=next(colors), legend_label='ADIFR')
    p2.add_tools(ht)
    format_ticks(p2)

    colors = itertools.cycle(Category10[8])
    p3 = figure(width=width, height=height, x_range=p1.x_range)
    p3.add_layout(Title(text="RICE", align="center"), "left")
    p3.line(aoa_obj.df['datetime'], aoa_obj.df['RICE'], color=next(colors))
    p3.add_tools(ht)
    format_ticks(p3)
  
    colors = itertools.cycle(Category10[8])
    p7 = figure(width=width, height=int(height), x_range=p1.x_range)
    p7.add_layout(Title(text="Diff. Pres. [hPa]", align="center"), "left")
    p7.line(aoa_obj.df['datetime'], aoa_obj.df['QCR'], color=next(colors), legend_label='QCR')
    p7.line(aoa_obj.df['datetime'], aoa_obj.df['QCF'], color=next(colors), legend_label='QCF')
    p7.legend.location = 'top_left'
    p7.add_tools(ht)
    format_ticks(p7)

    colors = itertools.cycle(Category10[8])
    p8 = figure(width=width, height=int(height), x_range=p1.x_range, title=aoa_obj.flight)
    p8.add_layout(Title(text="Pres. Alt. [ft]", align="center"), "left")
    #p8.line(aoa_obj.df['datetime'], aoa_obj.df['PALTF'], color=next(colors), legend_label='PSFD')
    p8.line(aoa_obj.df['datetime'], aoa_obj.df['GGALT'], color=next(colors), legend_label='GGALT')
    p8.legend.location = 'bottom_right'
    p8.add_tools(ht)
    format_ticks(p8)

    colors = itertools.cycle(Category10[8])
    p9 = figure(width=width, height=int(height), x_range=p1.x_range)
    p9.add_layout(Title(text="Vert. Spd. [m/s]", align="center"), "left")
    p9.line(aoa_obj.df['datetime'], aoa_obj.df['GGVSPD'], color=next(colors))
    p9.line(aoa_obj.dt, aoa_obj.vspd, color=next(colors))
    p9.legend.location = 'bottom_right'
    p9.add_tools(ht)
    format_ticks(p9)

    colors = itertools.cycle(Category10[8])
    p4 = figure(width=width, height=int(height), x_range=p1.x_range)
    p4.add_layout(Title(text="Altitude [m]", align="center"), "left")
    p4.line(aoa_obj.df['datetime'], aoa_obj.df['GGALT'], color=next(colors))
    p4.legend.location = 'bottom_right'
    p4.add_tools(ht)
    format_ticks(p4)

    colors = itertools.cycle(Category10[8])
    p5 = figure(width=width, height=int(height), x_range=p1.x_range)
    p5.add_layout(Title(text="Wind Spd. [m/s]", align="center"), "left")
    #p5.line(aoa_obj.df['datetime'], aoa_obj.df['UIC'], color=next(colors), legend_label='UIC')
    #p5.line(aoa_obj.df['datetime'], aoa_obj.df['VIC'], color=next(colors), legend_label='VIC')
    #p5.line(aoa_obj.df['datetime'], aoa_obj.df['WIC'], color=next(colors), legend_label='WIC')
    p5.line(aoa_obj.dt, aoa_obj.w, color=next(colors), legend_label='WIC')
    p5.legend.location = 'bottom_right'
    p5.add_tools(ht)
    format_ticks(p5)

    colors = itertools.cycle(Category10[8])
    p6 = figure(width=width, height=int(height), x_range=p1.x_range)
    p6.add_layout(Title(text="z-acc [m/s/s]", align="center"), "left")
    p6.line(aoa_obj.df['datetime'], aoa_obj.df['BNORMA'], color=next(colors), legend_label='BNORMA')
    p6.line(aoa_obj.df['datetime'], aoa_obj.df['BLATA'], color=next(colors), legend_label='BLATA')
    p6.line(aoa_obj.df['datetime'], aoa_obj.df['BLONGA'], color=next(colors), legend_label='BLONGA')
    p6.legend.location = 'bottom_right'
    p6.add_tools(ht)
    format_ticks(p6)

    colors = itertools.cycle(Category10[8])
    p10 = figure(width=width, height=int(height*1.5), x_range=p1.x_range)
    p10.add_layout(Title(text="AoS [deg]", align="center"), "left")
    p10.line(aoa_obj.dt, aoa_obj.aos_ref, color=next(colors), legend_label='AOSREF')
    p10.line(aoa_obj.dt, aoa_obj.sslip, color=next(colors), legend_label='AOS ARISTO')
    p10.line(aoa_obj.dt, aoa_obj.sslip-aoa_obj.aos_ref, color=next(colors), legend_label='Diff')
    p10.legend.location = 'top_left'
    p10.add_tools(ht)
    format_ticks(p10)


    #p = gridplot([[p8], [p2], [p1], [p3], [p7], [p9], [p5], [p6],])
    p = gridplot([[p8], [p2], [p1], [p3], [p5],])
    show(p)

def plot_maneuv_for_aoa(aoa_obj: aoa_fit):
    # set up hover tool
    ht = HoverTool(tooltips=[('time', '@x{%H:%M:%S}'), ('y', '@y')], formatters={'@x': 'datetime'})

    # generate the altitude, heading and gps quality plots
    height = 150
    width = 1000
    colors = itertools.cycle(Category10[8])
    p1 = figure(width=width, height=int(height*1.5))
    p1.add_layout(Title(text="Pitch [deg]", align="center"), "left")
    p1.line(aoa_obj.dt, aoa_obj.akrd, color=next(colors), legend_label='AOAPrev')
    p1.line(aoa_obj.dt, aoa_obj.aoa_ref, color=next(colors), legend_label='AOAREF')
    p1.line(aoa_obj.dt, aoa_obj.akrd_three, color=next(colors), legend_label='AOAFit')
    p1.line(aoa_obj.dt, aoa_obj.akrd_two, color=next(colors), legend_label='AOAFitSimple')
    p1.legend.location = 'top_left'
    p1.add_tools(ht)
    format_ticks(p1)

    colors = itertools.cycle(Category10[8])
    p2 = figure(width=width, height=height, x_range=p1.x_range)
    p2.add_layout(Title(text="Diff. Pres. [hPa]", align="center"), "left")
    p2.line(aoa_obj.df['datetime'], aoa_obj.df['BDIFR'], color=next(colors), legend_label='BDIFR')
    p2.line(aoa_obj.df['datetime'], aoa_obj.df['ADIFR'], color=next(colors), legend_label='ADIFR')
    p2.add_tools(ht)
    format_ticks(p2)

    colors = itertools.cycle(Category10[8])
    p3 = figure(width=width, height=height, x_range=p1.x_range)
    p3.add_layout(Title(text="TAS [m/s]", align="center"), "left")
    p3.line(aoa_obj.df['datetime'], aoa_obj.df['TASFR'], color=next(colors))
    p3.add_tools(ht)
    format_ticks(p3)
  
    colors = itertools.cycle(Category10[8])
    p7 = figure(width=width, height=int(height), x_range=p1.x_range)
    p7.add_layout(Title(text="Diff. Pres. [hPa]", align="center"), "left")
    p7.line(aoa_obj.df['datetime'], aoa_obj.df['QCR'], color=next(colors), legend_label='QCR')
    p7.line(aoa_obj.df['datetime'], aoa_obj.df['QCF'], color=next(colors), legend_label='QCF')
    p7.legend.location = 'top_left'
    p7.add_tools(ht)
    format_ticks(p7)

    colors = itertools.cycle(Category10[8])
    p8 = figure(width=width, height=int(height), x_range=p1.x_range)
    p8.add_layout(Title(text="Pres. Alt. [ft]", align="center"), "left")
    p8.line(aoa_obj.df['datetime'], aoa_obj.df['PALTF'], color=next(colors), legend_label='PSFD')
    p8.legend.location = 'bottom_right'
    p8.add_tools(ht)
    format_ticks(p8)

    colors = itertools.cycle(Category10[8])
    p9 = figure(width=width, height=int(height), x_range=p1.x_range)
    p9.add_layout(Title(text="Vert. Spd. [m/s]", align="center"), "left")
    p9.line(aoa_obj.df['datetime'], aoa_obj.df['GGVSPD'], color=next(colors))
    p9.line(aoa_obj.dt, aoa_obj.vspd, color=next(colors))
    p9.legend.location = 'bottom_right'
    p9.add_tools(ht)
    format_ticks(p9)

    colors = itertools.cycle(Category10[8])
    p4 = figure(width=width, height=int(height), x_range=p1.x_range)
    p4.add_layout(Title(text="Altitude [m]", align="center"), "left")
    p4.line(aoa_obj.df['datetime'], aoa_obj.df['GGALT'], color=next(colors))
    p4.legend.location = 'bottom_right'
    p4.add_tools(ht)
    format_ticks(p4)

    colors = itertools.cycle(Category10[8])
    p5 = figure(width=width, height=int(height), x_range=p1.x_range)
    p5.add_layout(Title(text="Wind Spd. [m/s]", align="center"), "left")
    #p5.line(aoa_obj.df['datetime'], aoa_obj.df['UIC'], color=next(colors), legend_label='UIC')
    #p5.line(aoa_obj.df['datetime'], aoa_obj.df['VIC'], color=next(colors), legend_label='VIC')
    p5.line(aoa_obj.df['datetime'], aoa_obj.df['WIC'], color=next(colors), legend_label='WIC')
    p5.legend.location = 'bottom_right'
    p5.add_tools(ht)
    format_ticks(p5)

    colors = itertools.cycle(Category10[8])
    p6 = figure(width=width, height=height, x_range=p1.x_range)
    p6.add_layout(Title(text="Diff. Pres. [None]", align="center"), "left")
    #p6.line(aoa_obj.df['datetime'], aoa_obj.df['BDIFR']/aoa_obj.df['QCF'], color=next(colors), legend_label='BDIFR/QCF')
    #p6.line(aoa_obj.df['datetime'], aoa_obj.df['ADIFR']/aoa_obj.df['QCF'], color=next(colors), legend_label='ADIFR/QCF')
    p6.line(aoa_obj.dt, aoa_obj.bdifr/aoa_obj.q, color=next(colors), legend_label='BDIFR/QCF')
    p6.line(aoa_obj.dt, aoa_obj.adifr/aoa_obj.q, color=next(colors), legend_label='ADIFR/QCF')
    p6.add_tools(ht)
    format_ticks(p6)


    p = gridplot([[p8], [p2], [p1], [p3], [p7], [p9], [p5]])
    show(p)

def plot_maneuv_for_aos(aos_obj: aos_fit):
    # set up hover tool
    ht = HoverTool(tooltips=[('time', '@x{%H:%M:%S}'), ('y', '@y')], formatters={'@x': 'datetime'})

    # generate the altitude, heading and gps quality plots
    height = 150
    width = 1000
    colors = itertools.cycle(Category10[8])
    p1 = figure(width=width, height=int(height*1.5))
    p1.add_layout(Title(text="AoS [deg]", align="center"), "left")
    p1.line(aos_obj.dt, aos_obj.sslip, color=next(colors), legend_label='AOSPrev')
    p1.line(aos_obj.dt, aos_obj.aos_ref, color=next(colors), legend_label='AOSRef')
    p1.line(aos_obj.dt, aos_obj.sslip_fit, color=next(colors), legend_label='AOSFit')
    p1.legend.location = 'top_left'
    p1.add_tools(ht)
    format_ticks(p1)

    colors = itertools.cycle(Category10[8])
    p2 = figure(width=width, height=height, x_range=p1.x_range)
    p2.add_layout(Title(text="Diff. Pres. [hPa]", align="center"), "left")
    #p2.line(aos_obj.df['datetime'], aos_obj.df['BDIFR'], color=next(colors), legend_label='BDIFR')
    #p2.line(aos_obj.df['datetime'], aos_obj.df['ADIFR'], color=next(colors), legend_label='ADIFR')
    p2.line(aos_obj.dt, aos_obj.bdifr, color=next(colors), legend_label='BDIFR')
    p2.line(aos_obj.dt, aos_obj.adifr, color=next(colors), legend_label='ADIFR')
    p2.add_tools(ht)
    format_ticks(p2)

    colors = itertools.cycle(Category10[8])
    p3 = figure(width=width, height=height, x_range=p1.x_range)
    p3.add_layout(Title(text="Heading [deg]", align="center"), "left")
    p3.line(aos_obj.dt, aos_obj.hdg, color=next(colors), legend_label='THDG')
    p3.line(aos_obj.dt, aos_obj.hdg_sm, color=next(colors), legend_label='THDG_Smoothed')
    p3.line(aos_obj.dt, aos_obj.trk, color=next(colors), legend_label='TRK')
    p3.line(aos_obj.dt, aos_obj.aos_corr, color=next(colors), legend_label='AoS Corr.')
    p3.add_tools(ht)
    format_ticks(p3)

    colors = itertools.cycle(Category10[8])
    p4 = figure(width=width, height=height, x_range=p1.x_range)
    p4.add_layout(Title(text="U [m/s]", align="center"), "left")
    color = next(colors)
    p4.line(aos_obj.dt, aos_obj.u, color=color, legend_label='U')
    p4.line(aos_obj.dt, aos_obj.u_sm, color=color, legend_label='U_SM', line_dash='dashed')
    p4.add_tools(ht)
    format_ticks(p4)

    colors = itertools.cycle(Category10[8])
    p5 = figure(width=width, height=height, x_range=p1.x_range)
    p5.add_layout(Title(text="Up [m/s]", align="center"), "left")
    color = next(colors)
    p5.line(aos_obj.dt, aos_obj.up, color=color, legend_label='Up')
    #p5.line(aos_obj.dt, aos_obj.u_sm, color=color, legend_label='U_SM', line_dash='dashed')
    p5.add_tools(ht)
    format_ticks(p5)



    colors = itertools.cycle(Category10[8])
    p6 = figure(width=width, height=height, x_range=p1.x_range)
    p6.add_layout(Title(text="Roll [deg]", align="center"), "left")
    p6.line(aos_obj.dt, aos_obj.roll, color=next(colors))
    p6.add_tools(ht)
    format_ticks(p6)

    colors = itertools.cycle(Category10[8])
    p7 = figure(width=width, height=height, x_range=p1.x_range)
    p7.add_layout(Title(text="Pres. Alt. [m]", align="center"), "left")
    p7.line(aos_obj.dt, aos_obj.paltf, color=next(colors))
    p7.add_tools(ht)
    format_ticks(p7)
  
    p = gridplot([[p1], [p2], [p3], [p4], [p5], [p6], [p7], ])
    show(p)

def plot_aos_validation(prev: aos_fit, curr: aos_fit):
    # set up hover tool
    ht = HoverTool(tooltips=[('time', '@x{%H:%M:%S}'), ('y', '@y')], formatters={'@x': 'datetime'})

    # generate the altitude, heading and gps quality plots
    height = 150
    width = 1000

    colors = itertools.cycle(Category10[8])
    p1 = figure(width=width, height=int(height))
    p1.add_layout(Title(text="AoS [deg]", align="center"), "left")
    p1.line(curr.dt, curr.aos_ref, color=next(colors), legend_label='Reference AoS')
    #p1.line(curr.dt, curr.aos_ref2, color=next(colors), legend_label='Reference AoS Test')
    p1.line(prev.dt, prev.sslip, color=next(colors), legend_label='Prev AoS')
    p1.line(curr.dt, curr.sslip, color=next(colors), legend_label='Current AoS')
    p1.legend.location = 'top_left'
    p1.add_tools(ht)
    format_ticks(p1)

    colors = itertools.cycle(Category10[8])
    p2 = figure(width=width, height=int(height))
    p2.add_layout(Title(text="U [m/s]", align="center"), "left")
    p2.line(prev.dt, prev.up - prev.up_sm, color=next(colors), legend_label='Prev HP Up')
    p2.line(prev.dt, prev.u - prev.u_sm, color=next(colors), legend_label='Prev HP U')
    p2.line(curr.dt, curr.u - curr.u_sm, color=next(colors), legend_label='Current HP U')
    p2.legend.location = 'top_left'
    p2.add_tools(ht)
    format_ticks(p2)

    colors = itertools.cycle(Category10[8])
    p3 = figure(width=width, height=int(height))
    p3.add_layout(Title(text="V [m/s]", align="center"), "left")
    p3.line(prev.dt, prev.vp - prev.vp_sm, color=next(colors), legend_label='Prev HP Vp')
    p3.line(prev.dt, prev.v - prev.v_sm, color=next(colors), legend_label='Prev HP V')
    p3.line(curr.dt, curr.v - curr.v_sm, color=next(colors), legend_label='Current HP V')
    p3.legend.location = 'top_left'
    p3.add_tools(ht)
    format_ticks(p3)

    colors = itertools.cycle(Category10[8])
    p4 = figure(width=width, height=int(height))
    p4.add_layout(Title(text="Direction [deg]", align="center"), "left")
    p4.line(prev.dt, prev.hdg - np.mean(prev.hdg), color=next(colors), legend_label='Hdg', width=2)
    p4.line(prev.dt, prev.trk - np.mean(prev.trk), color=next(colors), legend_label='Trk')
    p4.line(prev.dt, prev.wd - np.mean(prev.wd), color=next(colors), legend_label='Prev WD')
    p4.line(prev.dt, curr.wd - np.mean(curr.wd), color=next(colors), legend_label='Current WD', width=2)
    #p4.line(prev.dt, prev.wd - prev.wd_sm, color=next(colors), legend_label='Prev HP WD')
    #p4.line(prev.dt, prev.wd, color=next(colors), legend_label='Prev HP WD')
    #p4.line(prev.dt, prev.wd_sm, color=next(colors), legend_label='Prev HP WD')
    #print(prev.wd)
    #print(prev.wd_sm)
    p4.legend.location = 'top_left'
    p4.add_tools(ht)
    format_ticks(p4)

    colors = itertools.cycle(Category10[8])
    p5 = figure(width=width, height=int(height))
    p5.add_layout(Title(text="BDIFR/QCF", align="center"), "left")
    p5.line(prev.dt, curr.ratio, color=next(colors))
    p5.line(prev.dt, curr.ratio_sm, color=next(colors))
    p5.legend.location = 'top_left'
    p5.add_tools(ht)
    format_ticks(p5)

    colors = itertools.cycle(Category10[8])
    p6 = figure(width=width, height=int(height))
    p6.add_layout(Title(text="Lat. Spd. [m/s]", align="center"), "left")
    p6.line(curr.dt, curr.lat_spd, color=next(colors))
    p6.legend.location = 'top_left'
    p6.add_tools(ht)
    format_ticks(p6)


    p = gridplot([[p5], [p1], [p4], [p2], [p3], [p6], ])
    show(p)




def plot_aoa_scatter(aoa_obj: aoa_fit, coefs: list[float], title='', aoa_range=(0,6.5), aos_range=(-6.5,6.5)):

    x = np.array([aoa_obj.adifr/aoa_obj.q,aoa_obj.mach])
    aoa_cal = fit_func(x, *coefs)

    # generate the altitude, heading and gps quality plots
    height = 400
    width = 400
    colors = itertools.cycle(Category10[8])
    p1 = figure(width=width, height=height, title=title+'AOARef vs Ratio')
    p1.add_layout(Title(text="AoA Ref [deg]", align="center"), "left")
    p1.add_layout(Title(text="ADIFR/Q", align="center"), "below")
    p1.dot(aoa_obj.adifr/aoa_obj.q, aoa_obj.aoa_ref, color=next(colors), size=10)
    p1.line(aoa_obj.adifr/aoa_obj.q, aoa_obj.akrd_two, color=next(colors), legend_label='1-predictor fit', width=2)
    p1.line(aoa_obj.adifr/aoa_obj.q, aoa_obj.akrd, color=next(colors), legend_label='ARISTO fit')
    p1.legend.location = 'top_left'

    colors = itertools.cycle(Category10[8])
    p2 = figure(width=width, height=height, title=title+'Predicted AoA vs Ref AoA', x_range=aoa_range, y_range=aoa_range)
    p2.add_layout(Title(text="Calibrated AoA [deg]", align="center"), "left")
    p2.add_layout(Title(text="AoA Ref [def]", align="center"), "below")
    p2.dot(aoa_obj.aoa_ref, aoa_cal, color=next(colors), size=10)
    #p2.dot(aoa_obj.aoa_ref, aoa_obj.akrd_two, color=next(colors), size=10, legend_label='1-predictor')
    #p2.dot(aoa_obj.aoa_ref, aoa_obj.akrd_three, color=next(colors), size=10, legend_label='2-predictors')
    #p2.dot(aoa_obj.aoa_ref, aoa_obj.akrd, color=next(colors), size=10, legend_label='ARISTO', fill_alpha=0.5, line_alpha=0.5)
    p2.line(aoa_range, aoa_range, color='black', legend_label='1-to-1')
    p2.legend.location = 'top_left'
    #fit_three = fit_func(np.array([ratio, self.mach]), *self.coefs_three)

    colors = itertools.cycle(Category10[8])
    p3 = figure(width=width, height=height, title=title+' Reference AoS vs BDIFR/Q')
    p3.add_layout(Title(text="AoA Ref [deg]", align="center"), "left")
    p3.add_layout(Title(text="BDIFR/Q", align="center"), "below")
    p3.dot(aoa_obj.bdifr/aoa_obj.q, aoa_obj.aos_ref, color=next(colors), size=10)
    p3.line(aoa_obj.bdifr/aoa_obj.q, aoa_obj.sslip, color=next(colors), legend_label='ARISTO fit', width=2)
    p3.legend.location = 'top_left'

    colors = itertools.cycle(Category10[8])
    p4 = figure(width=width, height=height, title=title+' ARISTO AoS vs Ref AoS', x_range=aos_range, y_range=aos_range)
    p4.add_layout(Title(text="ARISTO AoA [deg]", align="center"), "left")
    p4.add_layout(Title(text="AoS Ref [def]", align="center"), "below")
    p4.dot(aoa_obj.aos_ref, aoa_obj.sslip, color=next(colors), size=10)
    p4.line(aos_range, aos_range, color='black', legend_label='1-to-1')
    p4.legend.location = 'top_left'
    #fit_three = fit_func(np.array([ratio, self.mach]), *self.coefs_three)

   
    #p1.line(aoa_obj.akrd_three, aoa_obj.aoa_ref, color=next(colors))
    #p = gridplot([[p1, p2], [p3, p4]])
    p = gridplot([[p1, p2],])
    show(p)

def plot_aoa_scatter_for_cal(aoa_obj: aoa_fit, title='', aoa_range=(0,6.5), aos_range=(-6.5,6.5)):

    # generate the altitude, heading and gps quality plots
    height = 400
    width = 400
    colors = itertools.cycle(Category10[8])
    p1 = figure(width=width, height=height, title=title+'AOARef vs Ratio')
    p1.add_layout(Title(text="AoA Ref [deg]", align="center"), "left")
    p1.add_layout(Title(text="ADIFR/Q", align="center"), "below")
    p1.dot(aoa_obj.adifr/aoa_obj.q, aoa_obj.aoa_ref, color=next(colors), size=10)
    p1.line(aoa_obj.adifr/aoa_obj.q, aoa_obj.akrd_two, color=next(colors), legend_label='1-predictor fit', width=2)
    p1.line(aoa_obj.adifr/aoa_obj.q, aoa_obj.akrd, color=next(colors), legend_label='ARISTO fit')
    p1.legend.location = 'top_left'

    colors = itertools.cycle(Category10[8])
    p2 = figure(width=width, height=height, title=title+'Predicted AoA vs Ref AoA', x_range=aoa_range, y_range=aoa_range)
    p2.add_layout(Title(text="Calibrated AoA [deg]", align="center"), "left")
    p2.add_layout(Title(text="AoA Ref [def]", align="center"), "below")
    p2.dot(aoa_obj.aoa_ref, aoa_obj.akrd_two, color=next(colors), size=10, legend_label='1-predictor')
    p2.dot(aoa_obj.aoa_ref, aoa_obj.akrd_three, color=next(colors), size=10, legend_label='2-predictors')
    p2.line(aoa_range, aoa_range, color='black', legend_label='1-to-1')
    p2.legend.location = 'top_left'

    p = gridplot([[p1, p2],])
    show(p)


def plot_aos_scatter(aos_obj: aos_fit, title='', aos_range=(-7,7)):
    # generate the altitude, heading and gps quality plots
    height = 400
    width = 400
    colors = itertools.cycle(Category10[8])
    p1 = figure(width=width, height=height, title=title+'AOSRef vs Ratio')
    p1.add_layout(Title(text="AoS Ref [deg]", align="center"), "left")
    p1.add_layout(Title(text="BDIFR/Q", align="center"), "below")
    p1.dot(aos_obj.bdifr/aos_obj.q, aos_obj.aos_ref, color=next(colors), size=10)
    p1.line(aos_obj.bdifr/aos_obj.q, aos_obj.sslip_fit, color=next(colors), legend_label='CAESAR fit', width=2)
    p1.line(aos_obj.bdifr/aos_obj.q, aos_obj.sslip, color=next(colors), legend_label='ARISTO fit')
    p1.legend.location = 'top_left'

    colors = itertools.cycle(Category10[8])
    p2 = figure(width=width, height=height, title=title+'Predicted AoS vs Ref AoS', x_range=aos_range, y_range=aos_range)
    p2.add_layout(Title(text="Fit AoS [deg]", align="center"), "left")
    p2.add_layout(Title(text="AoA Ref [def]", align="center"), "below")
    p2.dot(aos_obj.aos_ref, aos_obj.sslip_fit, color=next(colors), size=10, legend_label='CAESAR Fit')
    p2.line(aos_range, aos_range, color='black', legend_label='1-to-1')
    p2.legend.location = 'top_left'

    
    p = gridplot([[p1, p2]])
    show(p)


def plot_scatter_before_after(aoa_obj_before: aoa_fit, aoa_obj_after: aoa_fit, title='', aoa_range=(0,6.5)):
    # generate the altitude, heading and gps quality plots
    height = 400
    width = 400
    colors = itertools.cycle(Category10[8])
    p1 = figure(width=width, height=height, title=title+', Raw')
    p1.add_layout(Title(text="AoA Ref [deg]", align="center"), "left")
    p1.add_layout(Title(text="ADIFR/Q", align="center"), "below")
    p1.dot(aoa_obj_before.adifr/aoa_obj_before.q, aoa_obj_before.aoa_ref, color=next(colors), size=10)
    p1.line(aoa_obj_before.adifr/aoa_obj_before.q, aoa_obj_before.akrd_two, color=next(colors))

    colors = itertools.cycle(Category10[8])
    p2 = figure(width=width, height=height, title=title+', Icing Times Removed')
    p2.add_layout(Title(text="AoA Ref [deg]", align="center"), "left")
    p2.add_layout(Title(text="ADIFR/Q", align="center"), "below")
    p2.dot(aoa_obj_after.adifr/aoa_obj_after.q, aoa_obj_after.aoa_ref, color=next(colors), size=10)
    p2.line(aoa_obj_after.adifr/aoa_obj_after.q, aoa_obj_after.akrd_two, color=next(colors))

    #colors = itertools.cycle(Category10[8])
    #p2 = figure(width=width, height=height, title=title+'Predicted AoA vs Ref AoA', x_range=aoa_range, y_range=aoa_range)
    #p2.add_layout(Title(text="Fit AoA [deg]", align="center"), "left")
    #p2.add_layout(Title(text="AoA Ref [def]", align="center"), "below")
    #p2.dot(aoa_obj.aoa_ref, aoa_obj.akrd_two, color=next(colors), size=10, legend_label='1-predictor')
    #p2.dot(aoa_obj.aoa_ref, aoa_obj.akrd_three, color=next(colors), size=10, legend_label='2-predictors')
    #p2.line(aoa_range, aoa_range, color='black', legend_label='1-to-1')
    #p2.legend.location = 'top_left'
    #fit_three = fit_func(np.array([ratio, self.mach]), *self.coefs_three)

    
    #p1.line(aoa_obj.akrd_three, aoa_obj.aoa_ref, color=next(colors))
    p = gridplot([[p1, p2]])
    show(p)

def plot_z_vs_tas(aoa_obj: aoa_fit, xlabel='', ylabel=''):
    # generate the altitude, heading and gps quality plots
    height = 500
    width = 500
    colors = itertools.cycle(Category10[8])
    p1 = figure(width=width, height=height, title='Altitudes of Speed Runs')
    p1.add_layout(Title(text="Altitude [km]", align="center"), "left")
    p1.add_layout(Title(text="TAS [m/s]", align="center"), "below")
    p1.dot(aoa_obj.tas, aoa_obj.alt, color=next(colors), size=10)

    show(p1)

def plot_hist(var, nbins=20, xlabel='', ylabel="Probability"):
    var_min = np.min(var)
    var_max = np.max(var)
    bins = np.linspace(var_min, var_max, nbins)
    hist, edges = np.histogram(var, density=True, bins=bins)

    height = 500
    width = 500
    p = figure(width=width, height=height, title='Histogram')
    p.add_layout(Title(text=xlabel, align="center"), "below")
    p.add_layout(Title(text=ylabel, align="center"), "left")
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color='black', line_color='white')
    show(p)

def plot_hists(obj, nbins=20):
    alt = obj.alt
    mach = obj.mach

    alt_bins = np.linspace(np.min(alt), np.max(alt), nbins)
    alt_hist, alt_edges = np.histogram(alt, density=True, bins=alt_bins)

    mach_bins = np.linspace(np.min(mach), np.max(mach), nbins)
    mach_hist, mach_edges = np.histogram(mach, density=True, bins=mach_bins)

    height = 400
    width = 400

    p1 = figure(width=width, height=height, title='Histogram')
    p1.add_layout(Title(text="Height [m]", align="center"), "below")
    p1.add_layout(Title(text="Prob.", align="center"), "left")
    p1.quad(top=alt_hist, bottom=0, left=alt_edges[:-1], right=alt_edges[1:], fill_color='black', line_color='white')

    p2 = figure(width=width, height=height, title='Histogram')
    p2.add_layout(Title(text="Mach Number", align="center"), "below")
    p2.add_layout(Title(text="Prob.", align="center"), "left")
    p2.quad(top=mach_hist, bottom=0, left=mach_edges[:-1], right=mach_edges[1:], fill_color='black', line_color='white')


    p = gridplot([[p1, p2]])
    show(p)


def plot_flight_for_aoa(aoa_obj: aoa_fit):
    # set up hover tool
    ht = HoverTool(tooltips=[('time', '@x{%H:%M:%S}'), ('y', '@y')], formatters={'@x': 'datetime'})

    # generate the altitude, heading and gps quality plots
    height = 150
    width = 1000
    colors = itertools.cycle(Category10[8])
    p1 = figure(width=width, height=int(height*1.5))
    p1.add_layout(Title(text="Pitch [deg]", align="center"), "left")
    p1.line(aoa_obj.dt, aoa_obj.akrd, color=next(colors), legend_label='AOAPrev')
    p1.line(aoa_obj.dt, aoa_obj.aoa_ref, color=next(colors), legend_label='AOAREF')
    p1.line(aoa_obj.dt, aoa_obj.akrd_three, color=next(colors), legend_label='AOAFit')
    p1.line(aoa_obj.dt, aoa_obj.akrd_two, color=next(colors), legend_label='AOAFitSimple')
    #p1.line(aoa_obj.dt, aoa_obj.pitch, color=next(colors), legend_label='PITCH')
    p1.legend.location = 'top_left'
    p1.add_tools(ht)
    format_ticks(p1)

    colors = itertools.cycle(Category10[8])
    p2 = figure(width=width, height=height, x_range=p1.x_range)
    p2.add_layout(Title(text="Diff. Pres. [hPa]", align="center"), "left")
    p2.line(aoa_obj.df['datetime'], aoa_obj.df['BDIFR'], color=next(colors), legend_label='BDIFR')
    p2.line(aoa_obj.df['datetime'], aoa_obj.df['ADIFR'], color=next(colors), legend_label='ADIFR')
    p2.add_tools(ht)
    format_ticks(p2)

    colors = itertools.cycle(Category10[8])
    p3 = figure(width=width, height=height, x_range=p1.x_range)
    p3.add_layout(Title(text="TAS [m/s]", align="center"), "left")
    p3.line(aoa_obj.df['datetime'], aoa_obj.df['TASFR'], color=next(colors))
    p3.add_tools(ht)
    format_ticks(p3)
  
    colors = itertools.cycle(Category10[8])
    p7 = figure(width=width, height=int(height), x_range=p1.x_range)
    p7.add_layout(Title(text="Diff. Pres. [hPa]", align="center"), "left")
    p7.line(aoa_obj.df['datetime'], aoa_obj.df['QCR'], color=next(colors), legend_label='QCR')
    p7.line(aoa_obj.df['datetime'], aoa_obj.df['QCF'], color=next(colors), legend_label='QCF')
    p7.legend.location = 'top_left'
    p7.add_tools(ht)
    format_ticks(p7)

    colors = itertools.cycle(Category10[8])
    p8 = figure(width=width, height=int(height), x_range=p1.x_range)
    p8.add_layout(Title(text="Pres. Alt. [ft]", align="center"), "left")
    p8.line(aoa_obj.df['datetime'], aoa_obj.df['PALTF'], color=next(colors))
    p8.line(aoa_obj.df['datetime'][aoa_obj.mask], aoa_obj.df['PALTF'][aoa_obj.mask], color=next(colors))
    p8.legend.location = 'bottom_right'
    p8.add_tools(ht)
    format_ticks(p8)

    colors = itertools.cycle(Category10[8])
    p9 = figure(width=width, height=int(height), x_range=p1.x_range)
    p9.add_layout(Title(text="Vert. Spd. [m/s]", align="center"), "left")
    p9.line(aoa_obj.df['datetime'], aoa_obj.df['GGVSPD'], color=next(colors))
    p9.line(aoa_obj.dt, aoa_obj.vspd, color=next(colors))
    p9.legend.location = 'bottom_right'
    p9.add_tools(ht)
    format_ticks(p9)

    colors = itertools.cycle(Category10[8])
    p4 = figure(width=width, height=int(height), x_range=p1.x_range)
    p4.add_layout(Title(text="Altitude [m]", align="center"), "left")
    p4.line(aoa_obj.df['datetime'], aoa_obj.df['GGALT'], color=next(colors))
    p4.legend.location = 'bottom_right'
    p4.add_tools(ht)
    format_ticks(p4)

    colors = itertools.cycle(Category10[8])
    p5 = figure(width=width, height=int(height), x_range=p1.x_range)
    p5.add_layout(Title(text="Wind Spd. [m/s]", align="center"), "left")
    #p5.line(aoa_obj.df['datetime'], aoa_obj.df['UIC'], color=next(colors), legend_label='UIC')
    #p5.line(aoa_obj.df['datetime'], aoa_obj.df['VIC'], color=next(colors), legend_label='VIC')
    p5.line(aoa_obj.df['datetime'], aoa_obj.df['WIC'], color=next(colors), legend_label='WIC')
    p5.legend.location = 'bottom_right'
    p5.add_tools(ht)
    format_ticks(p5)

    colors = itertools.cycle(Category10[8])
    p6 = figure(width=width, height=height, x_range=p1.x_range)
    p6.add_layout(Title(text="Diff. Pres. [None]", align="center"), "left")
    #p6.line(aoa_obj.df['datetime'], aoa_obj.df['BDIFR']/aoa_obj.df['QCF'], color=next(colors), legend_label='BDIFR/QCF')
    #p6.line(aoa_obj.df['datetime'], aoa_obj.df['ADIFR']/aoa_obj.df['QCF'], color=next(colors), legend_label='ADIFR/QCF')
    p6.line(aoa_obj.dt, aoa_obj.bdifr/aoa_obj.q, color=next(colors), legend_label='BDIFR/QCF')
    p6.line(aoa_obj.dt, aoa_obj.adifr/aoa_obj.q, color=next(colors), legend_label='ADIFR/QCF')
    p6.add_tools(ht)
    format_ticks(p6)


    p = gridplot([[p8], ])
    show(p)

def plot_all_scatters(aoa_objs: list[aoa_fit]):
    # generate the altitude, heading and gps quality plots
    height = 300
    width = 300

    plt_per_row = 3
    figs = []
    row_list = []
    for i, aoa_obj in enumerate(aoa_objs):
        colors = itertools.cycle(Category10[8])
        p = figure(width=width, height=height, title=f"{aoa_obj.flight}, AOARef vs Ratio")
        p.add_layout(Title(text="AoA Ref [deg]", align="center"), "left")
        p.add_layout(Title(text="ADIFR/Q", align="center"), "below")
        p.dot(aoa_obj.adifr/aoa_obj.q, aoa_obj.aoa_ref, color=next(colors), size=10)
        p.line(aoa_obj.adifr/aoa_obj.q, aoa_obj.akrd_two, color=next(colors), legend_label='1-predictor fit', width=2)
        p.line(aoa_obj.adifr/aoa_obj.q, aoa_obj.akrd, color=next(colors), legend_label='ARISTO fit')
        p.legend.location = 'top_left'
        if len(row_list) < plt_per_row:
            row_list.append(p)
        else:
            figs.append(row_list)
            row_list = [p]

        if i == len(aoa_objs)-1:
            figs.append(row_list)
        
    p = gridplot(figs)
    show(p)

def plot_all_scatters_vertspd(up: list[aoa_fit], dn: list[aoa_fit], coefs):
    # generate the altitude, heading and gps quality plots
    height = 300
    width = 300

    plt_per_row = 3
    figs = []
    row_list = []
    for i, up_obj in enumerate(up):
        dn_obj = dn[i]

        combined_obj = up_obj.append(dn_obj)
        x = np.array([combined_obj.adifr/combined_obj.q, combined_obj.mach])
        aoa_best = fit_func(x, *coefs)

        colors = itertools.cycle(Category10[8])
        p = figure(width=width, height=height, title=f"{up_obj.flight}, AOARef vs Ratio")
        p.add_layout(Title(text="AoA Ref [deg]", align="center"), "left")
        p.add_layout(Title(text="ADIFR/Q", align="center"), "below")
        p.dot(up_obj.adifr/up_obj.q, up_obj.aoa_ref, color=next(colors), size=10, legend_label="Ascent")
        p.dot(dn_obj.adifr/dn_obj.q, dn_obj.aoa_ref, color=next(colors), size=10, legend_label="Descent")
        #p.line(aoa_obj.adifr/aoa_obj.q, aoa_obj.akrd_two, color=next(colors), legend_label='1-predictor fit', width=2)
        #p.line(up_obj.adifr/up_obj.q, up_obj.akrd, color=next(colors), legend_label='ARISTO fit')
        p.line(x[0,:], aoa_best, color=next(colors), legend_label='CAESAR fit')
        p.legend.location = 'top_left'
        if len(row_list) < plt_per_row:
            row_list.append(p)
        else:
            figs.append(row_list)
            row_list = [p]

        if i == len(up)-1:
            figs.append(row_list)
        
    p = gridplot(figs)
    show(p)

def plot_aoa_adifr_vertspd(up: aoa_fit, dn: aoa_fit, coefs):

    combined_obj = up.append(dn)
    x = np.array([combined_obj.adifr/combined_obj.q, combined_obj.mach])
    aoa_best = fit_func(x, *coefs)

    # generate the altitude, heading and gps quality plots
    height = 600
    width = 600
    colors = itertools.cycle(Category10[8])
    p = figure(width=width, height=height, title='AOARef vs Ratio')
    p.add_layout(Title(text="AoA Ref [deg]", align="center"), "left")
    p.add_layout(Title(text="ADIFR/Q", align="center"), "below")
    p.dot(up.adifr/up.q, up.aoa_ref, color=next(colors), size=10, legend_label='Ascent')
    p.dot(dn.adifr/dn.q, dn.aoa_ref, color=next(colors), size=10, legend_label='Descent')
    p.line(x[0,:], aoa_best, color=next(colors), legend_label='CAESAR fit', width=2)
    #p.line(up.adifr/up.q, up.akrd, color=next(colors), legend_label='ARISTO fit', width=2)
    #p.line(aoa_obj.adifr/aoa_obj.q, aoa_obj.akrd_three, color=next(colors), legend_label='2-predictor fit', width=1)
    #p.line(aoa_obj.adifr/aoa_obj.q, aoa_obj.akrd, color=next(colors), legend_label='ARISTO fit')
    p.legend.location = 'top_left'

    show(p)

def plot_aoa_aoa_vertspd(up: aoa_fit, dn: aoa_fit, coefs: list[float], title='', aoa_range=(0,6.5)):
    # generate the altitude, heading and gps quality plots
    height = 600
    width = 600
    x_up = np.array([up.adifr/up.q, up.mach])
    x_dn = np.array([dn.adifr/dn.q, dn.mach])
    aoa_up = fit_func(x_up, *coefs)
    aoa_dn = fit_func(x_dn, *coefs)
    colors = itertools.cycle(Category10[8])
    p = figure(width=width, height=height, title=title+'AOARef vs Ratio', x_range=aoa_range, y_range=aoa_range)
    p.add_layout(Title(text="AoA Gust [deg]", align="center"), "left")
    p.add_layout(Title(text="AoA Ref [deg]", align="center"), "below")
    p.dot(up.aoa_ref, aoa_up, color=next(colors), size=10, legend_label='Ascent')
    p.dot(dn.aoa_ref, aoa_dn, color=next(colors), size=10, legend_label='Descent')
    #p.dot(dn.adifr/dn.q, dn.aoa_ref, color=next(colors), size=10, legend_label='Descent')
    #p.line(up.adifr/up.q, up.akrd, color=next(colors), legend_label='ARISTO fit', width=2)
    #p.line(aoa_obj.adifr/aoa_obj.q, aoa_obj.akrd_three, color=next(colors), legend_label='2-predictor fit', width=1)
    #p.line(aoa_obj.adifr/aoa_obj.q, aoa_obj.akrd, color=next(colors), legend_label='ARISTO fit')
    p.line(aoa_range, aoa_range, color='black', legend_label='1-to-1', width=1)
    p.legend.location = 'top_left'

    show(p)



def plot_aoa_adifr(aoa_obj: aoa_fit, title='', aoa_range=(0,6.5)):
    # generate the altitude, heading and gps quality plots
    height = 600
    width = 600
    colors = itertools.cycle(Category10[8])
    p = figure(width=width, height=height, title=title+'AOARef vs Ratio')
    p.add_layout(Title(text="AoA Ref [deg]", align="center"), "left")
    p.add_layout(Title(text="ADIFR/Q", align="center"), "below")
    p.dot(aoa_obj.adifr/aoa_obj.q, aoa_obj.aoa_ref, color=next(colors), size=10)
    p.line(aoa_obj.adifr/aoa_obj.q, aoa_obj.akrd_two, color=next(colors), legend_label='1-predictor fit', width=2)
    #p.line(aoa_obj.adifr/aoa_obj.q, aoa_obj.akrd_three, color=next(colors), legend_label='2-predictor fit', width=1)
    #p.line(aoa_obj.adifr/aoa_obj.q, aoa_obj.akrd, color=next(colors), legend_label='ARISTO fit')
    p.legend.location = 'top_left'

    show(p)

def plot_aoa_aoa(aoa_obj: aoa_fit, aoa_range=(0,6)):
    # generate the altitude, heading and gps quality plots
    height = 600
    width = 600
    colors = itertools.cycle(Category10[8])
    p = figure(width=width, height=height, title='AOAFit vs AOARef', x_range=aoa_range, y_range=aoa_range)
    p.add_layout(Title(text="AoA Fit [deg]", align="center"), "left")
    p.add_layout(Title(text="AoA Ref [deg]", align="center"), "below")
    p.dot(aoa_obj.aoa_ref, aoa_obj.akrd_two, color=next(colors), size=10, legend_label='No Mach Fit')
    #p.dot(aoa_obj.aoa_ref, aoa_obj.akrd_three, color=next(colors), size=10, legend_label='Mach Fit', fill_alpha=0.5, line_alpha=0.5)
    p.dot(aoa_obj.aoa_ref, aoa_obj.akrd_three, color=next(colors), size=10, legend_label='Mach Fit')
    #p.dot(aoa_obj.aoa_ref, aoa_obj.akrd_four, color=next(colors), size=10, legend_label='Mach, Rho Fit')
    p.line(aoa_range, aoa_range, color='black', legend_label='1-to-1', width=1)
    p.legend.location = 'top_left'

    show(p)

def plot_aoa_aoa_final(aoa_obj: aoa_fit, coefs_mans, coefs_flights, coefs_final, aoa_range=(0,6)):
    # generate the altitude, heading and gps quality plots
    height = 500
    width = 500
    alpha = 0.1
    x = np.array([aoa_obj.adifr/aoa_obj.q, aoa_obj.mach])
    aoa_mans = fit_func(x, *coefs_mans)
    aoa_flights = fit_func(x, *coefs_flights)
    aoa_final = fit_func(x, *coefs_final)

    colors = itertools.cycle(Category10[8])
    p1 = figure(width=width, height=height, title='Coefficients only from Maneuvers', x_range=aoa_range, y_range=aoa_range)
    p1.add_layout(Title(text="AoA Fit [deg]", align="center"), "left")
    p1.add_layout(Title(text="AoA Ref [deg]", align="center"), "below")
    p1.dot(aoa_obj.aoa_ref, aoa_mans, color=next(colors), size=10, fill_alpha=alpha, line_alpha=alpha)
    p1.line(aoa_range, aoa_range, color='black', legend_label='1-to-1', width=1)
    p1.legend.location = 'top_left'

    colors = itertools.cycle(Category10[8])
    p2 = figure(width=width, height=height, title='Coefficients from All Flight Data', x_range=aoa_range, y_range=aoa_range)
    p2.add_layout(Title(text="AoA Fit [deg]", align="center"), "left")
    p2.add_layout(Title(text="AoA Ref [deg]", align="center"), "below")
    p2.dot(aoa_obj.aoa_ref, aoa_flights, color=next(colors), size=10, fill_alpha=alpha, line_alpha=alpha)
    p2.line(aoa_range, aoa_range, color='black', legend_label='1-to-1', width=1)
    p2.legend.location = 'top_left'

    colors = itertools.cycle(Category10[8])
    p3 = figure(width=width, height=height, title='Coefficients from Maneuvers + RF10 BL Profiling', x_range=aoa_range, y_range=aoa_range)
    p3.add_layout(Title(text="AoA Fit [deg]", align="center"), "left")
    p3.add_layout(Title(text="AoA Ref [deg]", align="center"), "below")
    p3.dot(aoa_obj.aoa_ref, aoa_final, color=next(colors), size=10, fill_alpha=alpha, line_alpha=alpha)
    p3.line(aoa_range, aoa_range, color='black', legend_label='1-to-1', width=1)
    p3.legend.location = 'top_left'

    p = gridplot([[p1, p2,], [p3],])
    show(p)




def plot_scatter_rf04(low: aoa_fit, mid: aoa_fit, high: aoa_fit):
    # generate the altitude, heading and gps quality plots
    height = 600
    width = 600
    ratios = np.array([-0.35, 0])
    colors = itertools.cycle(Category10[8])
    color1=next(colors)
    color2=next(colors)
    color3=next(colors)

    p1 = figure(width=width, height=height, title='AOARef vs Ratio', x_range=list(ratios))
    p1.add_layout(Title(text="AoA Ref [deg]", align="center"), "left")
    p1.add_layout(Title(text="ADIFR/Q", align="center"), "below")
    p1.dot(low.adifr/low.q, low.aoa_ref, color=color1, size=10)
    p1.dot(mid.adifr/mid.q, mid.aoa_ref, color=color2, size=10)
    p1.dot(high.adifr/high.q, high.aoa_ref, color=color3, size=10)
    p1.line(ratios, simple_fit_func(ratios, *low.coefs_two), color=color1, legend_label='Low', width=1.5)
    p1.line(ratios, simple_fit_func(ratios, *mid.coefs_two), color=color2, legend_label='Mid', width=1.5)
    p1.line(ratios, simple_fit_func(ratios, *high.coefs_two), color=color3, legend_label='High', width=1.5)
    p1.legend.location = 'top_left'

    #p2 = figure(width=width, height=height, title='AOARef vs Ratio')
    #p2.add_layout(Title(text="AoA Ref [deg]", align="center"), "left")
    #p2.add_layout(Title(text="ADIFR/Q", align="center"), "below")
    ##p2.dot(low.adifr/low.q, low.aoa_ref, color=color1, size=10)
    ##p2.dot(high.adifr/high.q, high.aoa_ref, color=color3, size=10)
    #p2.dot(mid.adifr/mid.q, mid.aoa_ref, color=color2, size=10)
    #p2.line(ratios, simple_fit_func(ratios, *low.coefs_two), color=color1, legend_label='Low', width=1.5)
    #p2.line(ratios, simple_fit_func(ratios, *mid.coefs_two), color='black', legend_label='Mid', width=1.5)
    #p2.line(ratios, simple_fit_func(ratios, *high.coefs_two), color=color3, legend_label='High', width=1.5)
    #p2.legend.location = 'top_left'

    #p3 = figure(width=width, height=height, title='AOARef vs Ratio')
    #p3.add_layout(Title(text="AoA Ref [deg]", align="center"), "left")
    #p3.add_layout(Title(text="ADIFR/Q", align="center"), "below")
    ##p3.dot(low.adifr/low.q, low.aoa_ref, color=color1, size=10)
    #p3.dot(high.adifr/high.q, high.aoa_ref, color=color3, size=10)
    ##p3.dot(mid.adifr/mid.q, mid.aoa_ref, color=color2, size=10)
    #p3.line(ratios, simple_fit_func(ratios, *low.coefs_two), color=color1, legend_label='Low', width=1.5)
    #p3.line(ratios, simple_fit_func(ratios, *mid.coefs_two), color=color2, legend_label='Mid', width=1.5)
    #p3.line(ratios, simple_fit_func(ratios, *high.coefs_two), color='black', legend_label='High', width=1.5)
    #p3.legend.location = 'top_left'

    #p = gridplot([[p1], [p2], [p3], ])

    #show(p)
    show(p1)


def plot_hists_rf04(low, mid, high , nbins=50):

    mach_low = low.mach
    mach_mid = mid.mach
    mach_high = high.mach

    low_bins = np.linspace(np.min(mach_low), np.max(mach_low), nbins)
    low_hist, low_edges = np.histogram(low.mach, density=True, bins=low_bins)

    mid_bins = np.linspace(np.min(mach_mid), np.max(mach_mid), nbins)
    mid_hist, mid_edges = np.histogram(mid.mach, density=True, bins=mid_bins)

    high_bins = np.linspace(np.min(mach_high), np.max(mach_high), nbins)
    high_hist, high_edges = np.histogram(high.mach, density=True, bins=high_bins)

    height = 600
    width = 600

    p = figure(width=width, height=height, title='Low')
    p.add_layout(Title(text="Mach Number", align="center"), "below")
    p.add_layout(Title(text="Prob.", align="center"), "left")
    p.quad(top=low_hist, bottom=0, left=low_edges[:-1], right=low_edges[1:], fill_color='white', line_color='black', legend_label='Low', fill_alpha=0)
    p.quad(top=mid_hist, bottom=0, left=mid_edges[:-1], right=mid_edges[1:], fill_color='white', line_color='green', legend_label='Mid', fill_alpha=0)
    p.quad(top=high_hist, bottom=0, left=high_edges[:-1], right=high_edges[1:], fill_color='white', line_color='blue', legend_label='High', fill_alpha=0)
    p.legend.location = 'top_center'

    show(p)


def add_7hrs(beg_end: dict):
    beg_end_copy = copy.deepcopy(beg_end)
    for flight, pairs in beg_end.items():
        for i, pair in enumerate(pairs):
            #print(f"{beg_end[flight][i][0]} {beg_end_copy[flight][i][1]}")
            beg_end_copy[flight][i][0] = beg_end[flight][i][0] + 7*3600
            beg_end_copy[flight][i][1] = beg_end[flight][i][1] + 7*3600

    return beg_end_copy

def plot_z_vs_w(aoa_obj: aoa_fit, title='', n_z_bins=20):

    # generate the altitude, heading and gps quality plots
    height = 500
    width = 500
    colors = itertools.cycle(Category10[8])
    p = figure(width=width, height=height, title=title+' Z vs W')
    p.add_layout(Title(text="Z [km]", align="center"), "left")
    p.add_layout(Title(text="w [m/s]", align="center"), "below")
    p.dot(aoa_obj.w, aoa_obj.alt/1000., color=next(colors), size=10)

    max_z = np.max(aoa_obj.alt)
    min_z = np.min(aoa_obj.alt)
    z_step = (max_z-min_z)/n_z_bins
    bin_edges = np.arange(min_z,max_z,z_step)
    bin_edges = np.append(bin_edges, max_z)
    bin_centers = 0.5*(bin_edges[0:-1]+bin_edges[1:])
    mean_w = np.zeros(len(bin_centers))
    for i in range(len(bin_centers)):
        bin_min = bin_edges[i]
        bin_max = bin_edges[i+1]
        bin_mask = np.logical_and(aoa_obj.alt > bin_min, aoa_obj.alt < bin_max)
        mean_w[i] = np.mean(aoa_obj.w[bin_mask])

    p.line(mean_w, bin_centers/1000., color='black', width=2)

    show(p)

def plot_all_z_vs_w(aoa_objs: list[aoa_fit], n_z_bins=20):
    # generate the altitude, heading and gps quality plots
    height = 300
    width = 300

    plt_per_row = 3
    figs = []
    row_list = []
    for i, aoa_obj in enumerate(aoa_objs):
        max_z = np.max(aoa_obj.alt)
        min_z = np.min(aoa_obj.alt)
        z_step = (max_z-min_z)/n_z_bins
        bin_edges = np.arange(min_z,max_z,z_step)
        bin_edges = np.append(bin_edges, max_z)
        bin_centers = 0.5*(bin_edges[0:-1]+bin_edges[1:])
        mean_w = np.zeros(len(bin_centers))
        for i in range(len(bin_centers)):
            bin_min = bin_edges[i]
            bin_max = bin_edges[i+1]
             
            bin_mask = np.logical_and(aoa_obj.alt > bin_min, aoa_obj.alt < bin_max)
            #print(f"bin min: {bin_min:5.0f}, bin max: {bin_max:5.0f}, mean z: {np.mean(aoa_obj.alt[bin_mask])}")
            mean_w[i] = np.mean(aoa_obj.w[bin_mask])
            #print(f"z: {bin_centers[i]:5.0f} m, mean w: {mean_w[i]}")

        #for i in range(len(bin_centers)):
        #    print(f"z: {bin_centers[i]:5.0f} m, mean w: {mean_w[i]}")
         
        colors = itertools.cycle(Category10[8])
        p = figure(width=width, height=height, title=f"{aoa_obj.flight}, AOARef vs Ratio")
        p.add_layout(Title(text="z [km]", align="center"), "left")
        p.add_layout(Title(text="w [m/s]", align="center"), "below")
        p.dot(aoa_obj.w, aoa_obj.alt/1000., color=next(colors), size=10)
        p.line(mean_w, bin_centers/1000., color='black', width=2)
        if len(row_list) < plt_per_row:
            row_list.append(p)
        else:
            figs.append(row_list)
            row_list = [p]

        if i == len(aoa_objs)-1:
            figs.append(row_list)

        
    p = gridplot(figs)
    show(p)

def plot_all_aoa_obj(aoa_objs: list[aoa_fit], coefs: list[float]):
    # set up hover tool
    ht = HoverTool(tooltips=[('time', '@x{%H:%M:%S}'), ('y', '@y')], formatters={'@x': 'datetime'})


    figs = []

    height = 200
    width = 1000

    for aoa_obj in aoa_objs:

        x = np.array([aoa_obj.adifr/aoa_obj.q, aoa_obj.mach])
        cal_aoa = fit_func(x, *coefs)
        # generate the altitude, heading and gps quality plots
        colors = itertools.cycle(Category10[8])
        p = figure(width=width, height=int(height), title=aoa_obj.flight)
        p.add_layout(Title(text="AoA [deg]", align="center"), "left")
        #p.line(aoa_obj.dt, aoa_obj.aoa_ref, color=next(colors), legend_label='AOAREF')
        #p.line(aoa_obj.dt, cal_aoa, color=next(colors), legend_label='AOACal')
        #p.line(aoa_obj.dt, aoa_obj.akrd, color=next(colors), legend_label='AKRD')
        p.line(aoa_obj.dt, cal_aoa-aoa_obj.aoa_ref, color=next(colors), legend_label='Diff')
        p.legend.location = 'top_left'
        p.add_tools(ht)
        format_ticks(p)
        figs.append([p])

    p = gridplot(figs)
    show(p)


