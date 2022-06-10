#%%
# import modules
import datetime as dt
import pandas as pd
import glob
import pickle
import numpy as np

import requests
from bs4 import BeautifulSoup

from energyquantified import EnergyQuantified, time
from razorshell.api_market_data import MarketDataAPI

from plotly.offline import init_notebook_mode
import plotly.graph_objects as go

# import statsmodels.api as sm
import sklearn.metrics as metrics
from patsy import dmatrices

init_notebook_mode(connected=True) 
pd.options.plotting.backend = "plotly"


#%%
#get EQ weather year demand data
# Initialize client
eq = EnergyQuantified(api_key='0168d6-b068b6-f41124-7108d3', timeout=60)

# Free-text search (filtering on attributes is also supported)
curves = [
    'NL Consumption MWh/h 15min Actual',
    'NL Solar Photovoltaic Production MWh/h 15min Actual',
    'NL Wind Power Production Onshore MWh/h 15min Actual',
    'NL Wind Power Production Offshore MWh/h 15min Actual',
    'NL Consumption MWh/h 15min_Backcast',
    'NL Solar Photovoltaic Production MWh/h 15min_Backcast',
    'NL Wind Power Production Onshore MWh/h 15min_Backcast',
    'NL Wind Power Production Offshore MWh/h 15min_Backcast',
    'GB Consumption MWh/h 30min Actual',
    'GB Solar Photovoltaic Production MWh/h 30min Actual',
    'GB Wind Power Production Onshore MWh/h 30min Actual',
    'GB Wind Power Production Offshore MWh/h 30min Actual',
    'GB Consumption MWh/h 15min_Backcast',
    'GB Solar Photovoltaic Production MWh/h 15min_Backcast',
    'GB Wind Power Production Onshore MWh/h 15min_Backcast',
    'GB Wind Power Production Offshore MWh/h 15min_Backcast',
    'ES Consumption MWh/h 15min Actual',
    'ES Solar Photovoltaic Production MWh/h 15min Actual',
    'ES Wind Power Production MWh/h 15min Actual',
    'ES Consumption MWh/h 15min_Backcast',
    'ES Solar Photovoltaic Production MWh/h 15min_Backcast',
    'ES Wind Power Production MWh/h 15min_Backcast',
    'DE Consumption MWh/h 15min Actual',
    'DE Solar Photovoltaic Production MWh/h 15min Actual',
    'DE Wind Power Production Onshore MWh/h 15min Actual',
    'DE Wind Power Production Offshore MWh/h 15min Actual',
    'DE Consumption MWh/h 15min_Backcast',
    'DE Solar Photovoltaic Production MWh/h 15min_Backcast',
    'DE Wind Power Production Onshore MWh/h 15min_Backcast',
    'DE Wind Power Production Offshore MWh/h 15min_Backcast',
    'FR Consumption MWh/h 30min Actual',
    'FR Solar Photovoltaic Production MWh/h 30min Actual',
    'FR Wind Power Production MWh/h 30min Actual',
    'FR Consumption MWh/h 15min_Backcast',
    'FR Solar Photovoltaic Production MWh/h 15min_Backcast',
    'FR Wind Power Production MWh/h 15min_Backcast',
    'BE Consumption MWh/h 15min Actual',
    'BE Solar Photovoltaic Production MWh/h H Actual',
    'BE Wind Power Production Onshore MWh/h H Actual',
    'BE Wind Power Production Offshore MWh/h H Actual',
    'BE Consumption MWh/h 15min_Backcast',
    'BE Solar Photovoltaic Production MWh/h 15min_Backcast',
    'BE Wind Power Production Onshore MWh/h 15min_Backcast',
    'BE Wind Power Production Offshore MWh/h 15min_Backcast',
    'AT Consumption MWh/h 15min Actual',
    'AT Solar Photovoltaic Production MWh/h 15min Actual',
    'AT Wind Power Production MWh/h 15min Actual',
    'AT Consumption MWh/h 15min_Backcast',
    'AT Solar Photovoltaic Production MWh/h 15min_Backcast',
    'AT Wind Power Production MWh/h 15min_Backcast',
    'CZ Consumption MWh/h H Actual',
    'CZ Solar Photovoltaic Production MWh/h H Actual',
    'CZ Wind Power Production MWh/h H Actual',
    'CZ Consumption MWh/h H Backcast',
    'CZ Solar Photovoltaic Production MWh/h H Backcast',
    'CZ Wind Power Production MWh/h H Backcast',
    # 'CH Consumption MWh/h 15min Actual',
    # 'CH Solar Photovoltaic Production MWh/h H Actual',
    # 'CH Wind Power Production MWh/h H Actual',
    # 'CH Consumption MWh/h H Backcast',
    # 'CH Solar Photovoltaic Production MWh/h H Backcast',
    # 'CH Wind Power Production MWh/h H Backcast',
    'DK Consumption MWh/h H Actual',
    'DK Solar Photovoltaic Production MWh/h H Actual',
    'DK Wind Power Production Onshore MWh/h H Actual',
    'DK Wind Power Production Offshore MWh/h H Actual',
    'DK Consumption MWh/h H_Backcast',
    'DK Solar Photovoltaic Production MWh/h H_Backcast',
    'DK Wind Power Production Onshore MWh/h H_Backcast',
    'DK Wind Power Production Offshore MWh/h H_Backcast',
    'PT Consumption MWh/h H Actual',
    'PT Solar Photovoltaic Production MWh/h H Actual',
    'PT Wind Power Production MWh/h H Actual',
    'PT Consumption MWh/h 15min_Backcast',
    'PT Solar Photovoltaic Production MWh/h 15min_Backcast',
    'PT Wind Power Production MWh/h 15min_Backcast',
    'PL Consumption MWh/h 15min Actual',
    'PL Solar Photovoltaic Production MWh/h H Actual',
    'PL Wind Power Production Onshore MWh/h H Actual',
    'PL Wind Power Production Offshore MWh/h H Actual',
    'PL Consumption MWh/h 15min_Backcast',
    'PL Solar Photovoltaic Production MWh/h 15min_Backcast',
    'PL Wind Power Production Onshore MWh/h 15min_Backcast',
    'PL Wind Power Production Offshore MWh/h 15min_Backcast',
]

curves_wx = [
    'NL Consumption MWh/h 15min Climate',
    'NL Solar Photovoltaic Production MWh/h 15min Climate',
    'NL Wind Power Production Onshore MWh/h 15min Climate',
    'NL Wind Power Production Offshore MWh/h 15min Climate',
    'GB Consumption MWh/h 15min Climate',
    'GB Solar Photovoltaic Production MWh/h 15min Climate',
    'GB Wind Power Production Onshore MWh/h 15min Climate',
    'GB Wind Power Production Offshore MWh/h 15min Climate',
    'ES Consumption MWh/h 15min Climate',
    'ES Solar Photovoltaic Production MWh/h 15min Climate',
    'ES Wind Power Production MWh/h 15min Climate',
    'DE Consumption MWh/h 15min Climate',
    'DE Solar Photovoltaic Production MWh/h 15min Climate',
    'DE Wind Power Production Onshore MWh/h 15min Climate',
    'DE Wind Power Production Offshore MWh/h 15min Climate',
    'FR Consumption MWh/h 15min Climate',
    'FR Solar Photovoltaic Production MWh/h 15min Climate',
    'FR Wind Power Production MWh/h 15min Climate',
    'BE Consumption MWh/h 15min Climate',
    'BE Solar Photovoltaic Production MWh/h H Climate',
    'BE Wind Power Production Onshore MWh/h H Climate',
    'BE Wind Power Production Offshore MWh/h H Climate',
    'AT Consumption MWh/h 15min Climate',
    'AT Solar Photovoltaic Production MWh/h 15min Climate',
    'AT Wind Power Production MWh/h 15min Climate',
    'CZ Consumption MWh/h H Climate',
    'CZ Solar Photovoltaic Production MWh/h H Climate',
    'CZ Wind Power Production MWh/h H Climate',
    # 'CH Consumption MWh/h 15min Climate',
    # 'CH Solar Photovoltaic Production MWh/h H Climate',
    # 'CH Wind Power Production MWh/h H Climate',
    'DK Consumption MWh/h H Climate',
    'DK Solar Photovoltaic Production MWh/h H Climate',
    'DK Wind Power Production Onshore MWh/h H Climate',
    'DK Wind Power Production Offshore MWh/h H Climate',
    'PT Consumption MWh/h H Climate',
    'PT Solar Photovoltaic Production MWh/h H Climate',
    'PT Wind Power Production MWh/h H Climate',
]

curves_cap = [
    # offshore
    'GB Wind Power Installed Offshore MW Capacity',
    'DE Wind Power Installed Offshore MW Capacity',
    'NL Wind Power Installed Offshore MW Capacity',
    'BE Wind Power Installed Offshore MW Capacity',
    # 'FR Wind Power Installed Offshore MW Capacity',
    'DK1 Wind Power Installed Offshore MW Capacity',
    'DK2 Wind Power Installed Offshore MW Capacity',
    #onshore
    'GB Wind Power Installed Onshore MW Capacity',
    'DE Wind Power Installed Onshore MW Capacity',
    'NL Wind Power Installed Onshore MW Capacity',
    'BE Wind Power Installed Onshore MW Capacity',
    # 'FR Wind Power Installed Onshore MW Capacity',
    'DK1 Wind Power Installed Onshore MW Capacity',
    'DK2 Wind Power Installed Onshore MW Capacity',
    'AT Wind Power Installed MW Capacity',
    'ES Wind Power Installed MW Capacity',
    'PT Wind Power Installed MW Capacity',
    'CH Wind Power Installed MW Capacity',
    'CZ Wind Power Installed MW Capacity',
    'FR Wind Power Installed MW Capacity',
    #solar
    'PT Solar Photovoltaic Installed MW Capacity',
    'BE Solar Photovoltaic Installed MW Capacity',
    'DK2 Solar Photovoltaic Installed MW Capacity',
    'FR Solar Photovoltaic Installed MW Capacity',
    'CZ Solar Photovoltaic Installed MW Capacity',
    'NL Solar Photovoltaic Installed MW Capacity',
    'ES Solar Photovoltaic Installed MW Capacity',
    'DE Solar Photovoltaic Installed MW Capacity',
    'AT Solar Photovoltaic Installed MW Capacity',
    'CH Solar Photovoltaic Installed MW Capacity',
    'GB Solar Photovoltaic Installed MW Capacity',
    'DK1 Solar Photovoltaic Installed MW Capacity',
]
#%%
# Load time series data
try:
    with open('dict_eq.pickle', 'rb') as handle:
        dict_eq = pickle.load(handle)
except FileNotFoundError:
    dict_eq = {}
    for curve_name in curves:
        curve = eq.metadata.curves(q=curve_name)[0]
        print(curve)
        dict_eq[curve.name] = eq.timeseries.load(
            curve,
            begin=dt.date(2016,1,1),
            end=dt.date(2025,1,2)
        )
        print('done')
    # Convert to Pandas data frame
    for k, v in dict_eq.items():
        dict_eq[k] = v.to_dataframe()        
    # save dict_cty as pickle
    with open('dict_eq.pickle', 'wb') as handle:
        pickle.dump(dict_eq, handle, protocol=pickle.HIGHEST_PROTOCOL)

try:
    with open('dict_eq_wx.pickle', 'rb') as handle:
        dict_eq_wx = pickle.load(handle)
except FileNotFoundError:
    dict_eq_wx = {}
    for curve_name in curves_wx:
        curve = eq.metadata.curves(q=curve_name)[0]
        print(curve)
        dict_eq_wx[curve.name] = eq.timeseries.load(
            curve,
            begin=dt.date(2016,1,1),
            end=dt.date(2025,1,2)
        )
    # Convert to Pandas data frame
    for k, v in dict_eq_wx.items():
        dict_eq_wx[k] = v.to_dataframe() 
    # save dict_cty as pickle
    with open('dict_eq_wx.pickle', 'wb') as handle:
        pickle.dump(dict_eq_wx, handle, protocol=pickle.HIGHEST_PROTOCOL)

try:
    with open('dict_cap.pickle', 'rb') as handle:
        dict_cap = pickle.load(handle)
except FileNotFoundError:
    dict_cap = {}
    for curve_name in curves_cap:
        curve = eq.metadata.curves(q=curve_name)[0]
        print(curve)
        dict_cap[curve.name] = eq.periods.load(
            curve,
            begin=dt.date(2016,1,1),
            end=dt.date(2025,1,2)
        )
    # Convert to Pandas data frame
    for k, v in dict_cap.items():
        dict_cap[k] = v.to_timeseries(time.Frequency.P1M).to_dataframe()    
    # save dict_cty as pickle
    with open('dict_cap.pickle', 'wb') as handle:
        pickle.dump(dict_cap, handle, protocol=pickle.HIGHEST_PROTOCOL)  


#%%
# resample to hourly
df_eq = pd.concat(dict_eq, axis=1)  
df_eq = df_eq.resample('h').mean()
df_eq.columns = df_eq.columns.get_level_values(0)

df_eq_wx = pd.concat(dict_eq_wx, axis=1)  
df_eq_wx = df_eq_wx.resample('h').mean()
# df_eq_wx.columns = df_eq_wx.columns.get_level_values(0)
df_eq_wx = df_eq_wx.groupby(level=[0,-1], axis=1).mean()

df_cap = pd.concat(dict_cap, axis=1)
df_cap = df_cap.resample('h').interpolate()
# df_cap.index = df_cap.index.tz_convert(None)
df_cap = df_cap.groupby(level=0, axis=1).mean()

#%%
# rename columns
rename_dict = {
    'NL Consumption MWh/h 15min Actual':'NL_Consumption',
    'NL Solar Photovoltaic Production MWh/h 15min Actual':'NL_Solar',
    'NL Wind Power Production Onshore MWh/h 15min Actual':'NL_Onshore',
    'NL Wind Power Production Offshore MWh/h 15min Actual':'NL_Offshore',
    'NL Consumption MWh/h 15min Backcast':'NL_Consumption_Backcast',
    'NL Solar Photovoltaic Production MWh/h 15min Backcast':'NL_Solar_Backcast',
    'NL Wind Power Production Onshore MWh/h 15min Backcast':'NL_Onshore_Backcast',
    'NL Wind Power Production Offshore MWh/h 15min Backcast':'NL_Offshore_Backcast',
    'NL Consumption MWh/h 15min Climate':'NL_Consumption_wx',
    'NL Solar Photovoltaic Production MWh/h 15min Climate':'NL_Solar_wx',
    'NL Wind Power Production Onshore MWh/h 15min Climate':'NL_Onshore_wx',
    'NL Wind Power Production Offshore MWh/h 15min Climate':'NL_Offshore_wx',
    'GB Consumption MWh/h 30min Actual':'GB_Consumption',
    'GB Solar Photovoltaic Production MWh/h 30min Actual':'GB_Solar',
    'GB Wind Power Production Onshore MWh/h 30min Actual':'GB_Onshore',
    'GB Wind Power Production Offshore MWh/h 30min Actual':'GB_Offshore',
    'GB Consumption MWh/h 15min Backcast':'GB_Consumption_Backcast',
    'GB Solar Photovoltaic Production MWh/h 15min Backcast':'GB_Solar_Backcast',
    'GB Wind Power Production Onshore MWh/h 15min Backcast':'GB_Onshore_Backcast',
    'GB Wind Power Production Offshore MWh/h 15min Backcast':'GB_Offshore_Backcast',
    'GB Consumption MWh/h 15min Climate':'GB_Consumption_wx',
    'GB Solar Photovoltaic Production MWh/h 15min Climate':'GB_Solar_wx',
    'GB Wind Power Production Onshore MWh/h 15min Climate':'GB_Onshore_wx',
    'GB Wind Power Production Offshore MWh/h 15min Climate':'GB_Offshore_wx',
    'ES Consumption MWh/h 15min Actual':'ES_Consumption',
    'ES Solar Photovoltaic Production MWh/h 15min Actual':'ES_Solar',
    'ES Wind Power Production MWh/h 15min Actual':'ES_Onshore',
    'ES Consumption MWh/h 15min Backcast':'ES_Consumption_Backcast',
    'ES Solar Photovoltaic Production MWh/h 15min Backcast':'ES_Solar_Backcast',
    'ES Wind Power Production MWh/h 15min Backcast':'ES_Onshore_Backcast',
    'ES Consumption MWh/h 15min Climate':'ES_Consumption_wx',
    'ES Solar Photovoltaic Production MWh/h 15min Climate':'ES_Solar_wx',
    'ES Wind Power Production MWh/h 15min Climate':'ES_Onshore_wx',
    'DE Consumption MWh/h 15min Actual':'DE_Consumption',
    'DE Solar Photovoltaic Production MWh/h 15min Actual':'DE_Solar',
    'DE Wind Power Production Onshore MWh/h 15min Actual':'DE_Onshore',
    'DE Wind Power Production Offshore MWh/h 15min Actual':'DE_Offshore',
    'DE Consumption MWh/h 15min Backcast':'DE_Consumption_Backcast',
    'DE Solar Photovoltaic Production MWh/h 15min Backcast':'DE_Solar_Backcast',
    'DE Wind Power Production Onshore MWh/h 15min Backcast':'DE_Onshore_Backcast',
    'DE Wind Power Production Offshore MWh/h 15min Backcast':'DE_Offshore_Backcast',
    'DE Consumption MWh/h 15min Climate':'DE_Consumption_wx',
    'DE Solar Photovoltaic Production MWh/h 15min Climate':'DE_Solar_wx',
    'DE Wind Power Production Onshore MWh/h 15min Climate':'DE_Onshore_wx',
    'DE Wind Power Production Offshore MWh/h 15min Climate':'DE_Offshore_wx',
    'FR Consumption MWh/h 30min Actual':'FR_Consumption',
    'FR Solar Photovoltaic Production MWh/h 30min Actual':'FR_Solar',
    'FR Wind Power Production MWh/h 30min Actual':'FR_Onshore',
    'FR Consumption MWh/h 15min Backcast':'FR_Consumption_Backcast',
    'FR Solar Photovoltaic Production MWh/h 15min Backcast':'FR_Solar_Backcast',
    'FR Wind Power Production MWh/h 15min Backcast':'FR_Onshore_Backcast',
    'FR Consumption MWh/h 15min Climate':'FR_Consumption_wx',
    'FR Solar Photovoltaic Production MWh/h 15min Climate':'FR_Solar_wx',
    'FR Wind Power Production MWh/h 15min Climate':'FR_Onshore_wx',
    'BE Consumption MWh/h 15min Actual':'BE_Consumption',
    'BE Solar Photovoltaic Production MWh/h H Actual':'BE_Solar',
    'BE Wind Power Production Onshore MWh/h H Actual':'BE_Onshore',
    'BE Wind Power Production Offshore MWh/h H Actual':'BE_Offshore',
    'BE Consumption MWh/h 15min Backcast':'BE_Consumption_Backcast',
    'BE Solar Photovoltaic Production MWh/h 15min Backcast':'BE_Solar_Backcast',
    'BE Wind Power Production Onshore MWh/h 15min Backcast':'BE_Onshore_Backcast',
    'BE Wind Power Production Offshore MWh/h 15min Backcast':'BE_Offshore_Backcast',
    'BE Consumption MWh/h 15min Climate':'BE_Consumption_wx',
    'BE Solar Photovoltaic Production MWh/h 15min Climate':'BE_Solar_wx',
    'BE Wind Power Production Onshore MWh/h 15min Climate':'BE_Onshore_wx',
    'BE Wind Power Production Offshore MWh/h 15min Climate':'BE_Offshore_wx',
    'AT Consumption MWh/h 15min Actual':'AT_Consumption',
    'AT Solar Photovoltaic Production MWh/h 15min Actual':'AT_Solar',
    'AT Wind Power Production MWh/h 15min Actual':'AT_Onshore',
    'AT Consumption MWh/h 15min Backcast':'AT_Consumption_Backcast',
    'AT Solar Photovoltaic Production MWh/h 15min Backcast':'AT_Solar_Backcast',
    'AT Wind Power Production MWh/h 15min Backcast':'AT_Onshore_Backcast',
    'AT Consumption MWh/h 15min Climate':'AT_Consumption_wx',
    'AT Solar Photovoltaic Production MWh/h 15min Climate':'AT_Solar_wx',
    'AT Wind Power Production MWh/h 15min Climate':'AT_Onshore_wx',
    'CZ Consumption MWh/h H Actual':'CZ_Consumption',
    'CZ Solar Photovoltaic Production MWh/h H Actual':'CZ_Solar',
    'CZ Wind Power Production MWh/h H Actual':'CZ_Onshore',
    'CZ Consumption MWh/h 15min Backcast':'CZ_Consumption_Backcast',
    'CZ Solar Photovoltaic Production MWh/h 15min Backcast':'CZ_Solar_Backcast',
    'CZ Wind Power Production MWh/h 15min Backcast':'CZ_Onshore_Backcast',
    'CZ Consumption MWh/h 15min Climate':'CZ_Consumption_wx',
    'CZ Solar Photovoltaic Production MWh/h 15min Climate':'CZ_Solar_wx',
    'CZ Wind Power Production MWh/h 15min Climate':'CZ_Onshore_wx',
    # 'CH Consumption MWh/h 15min Actual':'CH_Consumption',
    # 'CH Solar Photovoltaic Production MWh/h H Actual':'CH_Solar',
    # 'CH Wind Power Production MWh/h H Actual':'CH_Onshore',
    # 'CH Consumption MWh/h H Backcast':'CH_Consumption_Backcast',
    # 'CH Solar Photovoltaic Production MWh/h H Backcast':'CH_Solar_Backcast',
    # 'CH Wind Power Production MWh/h H Backcast':'CH_Onshore_Backcast',
    # 'CH Consumption MWh/h H Climate':'CH_Consumption_wx',
    # 'CH Solar Photovoltaic Production MWh/h H Climate':'CH_Solar_wx',
    # 'CH Wind Power Production MWh/h H Climate':'CH_Onshore_wx',
    'DK Consumption MWh/h H Actual':'DK_Consumption',
    'DK Solar Photovoltaic Production MWh/h H Actual':'DK_Solar',
    'DK Wind Power Production Onshore MWh/h H Actual':'DK_Onshore',
    'DK Wind Power Production Offshore MWh/h H Actual':'DK_Offshore',
    'DK Consumption MWh/h 15min Backcast':'DK_Consumption_Backcast',
    'DK Solar Photovoltaic Production MWh/h 15min Backcast':'DK_Solar_Backcast',
    'DK Wind Power Production Onshore MWh/h 15min Backcast':'DK_Onshore_Backcast',
    'DK Wind Power Production Offshore MWh/h 15min Backcast':'DK_Offshore_Backcast',
    'DK Consumption MWh/h 15min Climate':'DK_Consumption_wx',
    'DK Solar Photovoltaic Production MWh/h 15min Climate':'DK_Solar_wx',
    'DK Wind Power Production Onshore MWh/h 15min Climate':'DK_Onshore_wx',
    'DK Wind Power Production Offshore MWh/h 15min Climate':'DK_Offshore_wx',
    'PT Consumption MWh/h H Actual':'PT_Consumption',
    'PT Solar Photovoltaic Production MWh/h H Actual':'PT_Solar',
    'PT Wind Power Production MWh/h H Actual':'PT_Onshore',
    'PT Consumption MWh/h 15min Backcast':'PT_Consumption_Backcast',
    'PT Solar Photovoltaic Production MWh/h 15min Backcast':'PT_Solar_Backcast',
    'PT Wind Power Production MWh/h 15min Backcast':'PT_Onshore_Backcast',
    'PT Consumption MWh/h 15min Climate':'PT_Consumption_wx',
    'PT Solar Photovoltaic Production MWh/h 15min Climate':'PT_Solar_wx',
    'PT Wind Power Production MWh/h 15min Climate':'PT_Onshore_wx',
    'GB Wind Power Installed Offshore MW Capacity':'GB_Offshore',
    'DE Wind Power Installed Offshore MW Capacity':'DE_Offshore',
    'NL Wind Power Installed Offshore MW Capacity':'NL_Offshore',
    'BE Wind Power Installed Offshore MW Capacity':'BE_Offshore',
    'DK1 Wind Power Installed Offshore MW Capacity':'DK_Offshore',
    'DK2 Wind Power Installed Offshore MW Capacity':'DK_Offshore',
    'GB Wind Power Installed Onshore MW Capacity':'GB_Onshore',
    'DE Wind Power Installed Onshore MW Capacity':'DE_Onshore',
    'NL Wind Power Installed Onshore MW Capacity':'NL_Onshore',
    'BE Wind Power Installed Onshore MW Capacity':'BE_Onshore',
    'DK1 Wind Power Installed Onshore MW Capacity':'DK_Onshore',
    'DK2 Wind Power Installed Onshore MW Capacity':'DK_Onshore',
    'AT Wind Power Installed MW Capacity':'AT_Onshore',
    'ES Wind Power Installed MW Capacity':'ES_Onshore',
    'PT Wind Power Installed MW Capacity':'PT_Onshore',
    'CH Wind Power Installed MW Capacity':'CH_Onshore',
    'CZ Wind Power Installed MW Capacity':'CZ_Onshore',
    'FR Wind Power Installed MW Capacity':'FR_Onshore',
    'PT Solar Photovoltaic Installed MW Capacity':'PT_Solar',
    'BE Solar Photovoltaic Installed MW Capacity':'BE_Solar',
    'DK2 Solar Photovoltaic Installed MW Capacity':'DK_Solar',
    'FR Solar Photovoltaic Installed MW Capacity':'FR_Solar',
    'CZ Solar Photovoltaic Installed MW Capacity':'CZ_Solar',
    'NL Solar Photovoltaic Installed MW Capacity':'NL_Solar',
    'ES Solar Photovoltaic Installed MW Capacity':'ES_Solar',
    'DE Solar Photovoltaic Installed MW Capacity':'DE_Solar',
    'AT Solar Photovoltaic Installed MW Capacity':'AT_Solar',
    'CH Solar Photovoltaic Installed MW Capacity':'CH_Solar',
    'GB Solar Photovoltaic Installed MW Capacity':'GB_Solar',
    'DK1 Solar Photovoltaic Installed MW Capacity':'DK_Solar',
}

df_eq = df_eq.rename(columns=rename_dict)
df_eq_wx = df_eq_wx.rename(columns=rename_dict)
df_cap = df_cap.rename(columns=rename_dict)
df_cap = df_cap.groupby(df_cap.columns, axis=1).sum()

#%%
#import entsoe load
try:
    with open('entsoe_load.pickle', 'rb') as handle:
        entsoe_load = pickle.load(handle)
except FileNotFoundError:
    path = r'../REMIT/entsoe/ActualTotalLoad_6.1.A' # use your path
    load_files = glob.glob(path + "/*.csv")

    lload = []

    for filename in load_files:
        print(filename)
        df = pd.read_csv(filename, sep='\t')
        # lload.append(df.loc[df.AreaName == 'NL CTY'])
        lload.append(df.loc[df.AreaName.isin(['NL CTY','ES CTY','DE CTY','FR CTY','BE CTY','AT CTY','CZ CTY','DK CTY','PT CTY'])])

    entsoe_load = pd.concat(lload, axis=0, ignore_index=True)
    entsoe_load.set_index('DateTime', inplace=True)
    entsoe_load.index = pd.to_datetime(entsoe_load.index, utc=True)
    entsoe_load = entsoe_load.groupby([entsoe_load.index, 'AreaName']).mean().unstack(['AreaName'])['TotalLoadValue']
    entsoe_load = entsoe_load.resample('h').mean()
    with open('entsoe_load.pickle', 'wb') as handle:
        pickle.dump(entsoe_load, handle, protocol=pickle.HIGHEST_PROTOCOL)
#%%
#import entsoe generation
try:
    with open('entsoe_gen.pickle', 'rb') as handle:
        entsoe_gen = pickle.load(handle)
except FileNotFoundError:
    path = r'../REMIT/entsoe/AggregatedGenerationPerType_16.1.B_C' # use your path
    gen_files = glob.glob(path + "/*.csv")

    lgen = []

    for filename in gen_files:
        print(filename)
        df = pd.read_csv(filename, sep='\t')
        lgen.append(df.loc[(df.AreaName.isin(['NL CTY','ES CTY','DE CTY','FR CTY','BE CTY','AT CTY','CZ CTY','DK CTY','PT CTY']))&
            df.ProductionType.isin(['Wind Onshore','Solar', 'Wind Offshore'])])

    entsoe_gen = pd.concat(lgen, axis=0, ignore_index=True)
    entsoe_gen.set_index('DateTime', inplace=True)
    entsoe_gen.index = pd.to_datetime(entsoe_gen.index, utc=True)
    entsoe_gen = entsoe_gen.groupby([entsoe_gen.index, 'AreaName', 'ProductionType']).mean().unstack(['AreaName','ProductionType'])['ActualGenerationOutput']
    entsoe_gen= entsoe_gen.resample('h').mean()
    with open('entsoe_gen.pickle', 'wb') as handle:
        pickle.dump(entsoe_gen, handle, protocol=pickle.HIGHEST_PROTOCOL)


#%%
# import entsoe cap
# entsoe_cap = pd.read_csv('entsoe_cap.csv', index_col=0, header=[0, 1], parse_dates=[0])
# entsoe_cap = entsoe_cap.loc[:,
#     (entsoe_cap.columns.get_level_values('ProductionType').isin(['Solar', 'Wind Offshore', 'Wind Onshore']))&
#     (entsoe_cap.columns.get_level_values('MapCode').isin(entsoe_gen.columns.levels[0].str.replace(' CTY','')))]
# entsoe_cap.columns = entsoe_cap.columns.set_levels(entsoe_cap.columns.levels[1].str.replace('Wind ',''), level=1)
# entsoe_cap.columns = entsoe_cap.columns.map('_'.join).str.strip('_')

#%%
# entsoe_gen2 = entsoe_gen.copy()
# entsoe_gen2.columns = entsoe_gen2.columns.set_levels(entsoe_gen2.columns.levels[0].str.replace(' CTY',''), level=0)
# entsoe_gen2.columns = entsoe_gen2.columns.set_levels(entsoe_gen2.columns.levels[1].str.replace('Wind ',''), level=1)
# entsoe_gen2.columns = entsoe_gen2.columns.map('_'.join).str.strip('_')

#%%
ree_pv = pd.read_csv('REE_PV.csv', index_col=0, parse_dates=[0])
ree_pv.index.name = 'DateTime'
ree_pv.index = ree_pv.index.tz_localize('UTC')
ree_pv.columns = pd.MultiIndex.from_tuples([('ES CTY','Solar')])
entsoe_gen.drop(columns=('ES CTY','Solar'), inplace=True)
entsoe_gen = pd.concat([entsoe_gen,ree_pv], axis=1)

#%%
# Elexon data
start: str = '2015-01-01' #start of results
end: str = '2100-01-01' # end of results
api_client = MarketDataAPI(api_password='mddpwd_Diego1', api_user_name='diego.marquina@shell.com', timeout=300)
# pull the hourly wind and solar generation and price from the model outputs for all weather years
df_elexon = api_client.get_time_series(
    group_name='GB_balance_elexon',
    start=start,
    end=end, 
    granularity='hours',
    )

# %%
df_elexon.columns = df_elexon.columns.str.replace('bmreports.Generation_by_Fuel_Type;','', regex=False)
df_elexon.columns = df_elexon.columns.str.replace('555.','', regex=False)
df_elexon.columns = df_elexon.columns.str.replace('TIBCO_','', regex=False)
df_elexon.columns = df_elexon.columns.str.replace('BM-Interconnector-Flows.PRS;BM;','', regex=False)
df_elexon.columns = df_elexon.columns.str.replace(';MW;UK','', regex=False)
df_elexon.index = df_elexon.index.tz_localize('UTC')

#%%
#import ERA5 data
era5 = pd.read_csv('era5_check.csv', index_col=0, parse_dates=True, header=[0,1])
# era5 = pd.read_csv('era5_lf.csv', index_col=0, parse_dates=True, header=[0,1])

#%%
era5_dict = {
    'Austria':'AT',
    'Belgium':'BE',
    'CzechRepublic':'CZ',
    # 'DenmarkEast':'DKe',
    # 'DenmarkWest':'DKw',
    'DenmarkEast':'DK',
    'DenmarkWest':'DK',
    'France':'FR',
    'Germany':'DE',
    # 'ItalyCNOR':'ITCNOR',
    # 'ItalyNORD':'ITNORD',
    'ItalyCNOR':'IT',
    'ItalyNORD':'IT',
    'Netherlands':'NL',
    'Portugal':'PT',
    'Spain':'ES',
    'Switzerland':'CH',
    'UK':'GB',
}

#%%
era5.columns = era5.columns.map('_'.join).str.strip('_')
for k, v in era5_dict.items():
    # era5.columns = era5.columns.set_levels(era5.columns.levels[0].str.replace(k, v), level=0)
    era5.columns = era5.columns.str.replace(k, v)
era5 = era5.groupby(era5.columns, axis=1).sum()
era5.index = era5.index.tz_localize('UTC')

#%%
#fix some missing data
era5.loc['2016':'2020-07-15','AT_PV_Total_RealData'].replace(0,np.nan, inplace=True)
era5.loc['2016':'2018-12-30','ES_Wind_Total_RealData'].replace(0,np.nan, inplace=True)
#%%
for country in ['DE','GB']:
    df_cap[country+'_Total_Wind'] = df_cap[country+'_Onshore'] + df_cap[country+'_Offshore']
    df_eq[country+'_Total_Wind'] = df_eq[country+'_Onshore'] + df_eq[country+'_Offshore']
    df_eq[country+'_Total_Wind_Backcast'] = df_eq[country+'_Onshore_Backcast'] + df_eq[country+'_Offshore_Backcast']
    # temp = df_eq_wx[country+'_Onshore_wx'] + df_eq_wx[country+'_Offshore_wx']
    # temp.columns = pd.MultiIndex.from_product([[country+'_Total_Wind_wx'],temp.columns])
    # df_eq_wx = pd.concat([df_eq_wx,temp], axis=1)
    # del temp
    # if country =='DE':
    #     entsoe_gen2[country+'_Total_Wind'] = entsoe_gen2[country+'_Onshore'] + entsoe_gen2[country+'_Offshore']
    #     entsoe_cap[country+'_Total_Wind'] = entsoe_cap[country+'_Onshore'] + entsoe_cap[country+'_Offshore']        
    

#%%
# df_load_factor = df_eq.loc[:,df_eq.columns.isin(df_cap.columns)]/df_cap
# df_load_factor_backcast = df_eq.loc[:,df_eq.columns.isin(df_cap.add_suffix('_Backcast').columns)]/df_cap.add_suffix('_Backcast')
# # entsoe_lf = entsoe_gen2/df_cap
# entsoe_lf = entsoe_gen2/entsoe_cap.resample('h').interpolate().tz_localize('UTC')
#%%
# L = [df_cap.add_suffix('_wx')]*len(df_eq_wx.columns.levels[1])
# temp_aux_df = pd.concat(L, axis=1, keys=df_eq_wx.columns.levels[1]).swaplevel(0,1,axis=1)
# df_load_factor_wx = df_eq_wx.loc[:,df_eq_wx.columns.isin(temp_aux_df.columns)].div(temp_aux_df.loc[:,temp_aux_df.columns.isin(df_eq_wx.columns)], axis=0)
# df_load_factor_wx.columns = df_load_factor_wx.columns.set_levels(df_load_factor_wx.columns.levels[0].str.replace('_wx','_wx_lf'), level=0)
# del L, temp_aux_df
#%%
wx_19 = df_eq_wx.loc['2019',df_eq_wx.columns.get_level_values(1)=='y2019'].droplevel(1,axis=1)
wx_20 = df_eq_wx.loc['2020',df_eq_wx.columns.get_level_values(1)=='y2020'].droplevel(1,axis=1)
wx_ts = pd.concat([wx_19,wx_20], axis=0)

#%%
# lf_19 = df_load_factor_wx.loc['2019',df_load_factor_wx.columns.get_level_values(1)=='y2019'].droplevel(1,axis=1)
# lf_20 = df_load_factor_wx.loc['2020',df_load_factor_wx.columns.get_level_values(1)=='y2020'].droplevel(1,axis=1)
# lf_ts = pd.concat([lf_19,lf_20], axis=0)

#%%
# country frames
dict_cty = {}
for country in [
    'NL',
    'ES',
    'DE',
    'FR',
    'BE',
    'AT',
    'CZ',
    # 'CH',
    'DK',
    'PT',
    'GB',
    ]:
    print(country)
    if country == 'GB':
        dict_cty[country] = pd.concat([
            df_eq.loc[:,df_eq.columns.str.contains(country)],
            wx_ts.loc[:,wx_ts.columns.str.contains(country)],
            df_elexon[['INDO','WIND']],
            era5.loc['2015':,era5.columns.str.contains(country)]],axis=1)
            # df_load_factor.loc[:,df_load_factor.columns.str.contains(country)],
            # df_load_factor_backcast.loc[:,df_load_factor_backcast.columns.str.contains(country)],
            # lf_ts.loc[:,lf_ts.columns.str.contains(country)],
            # df_elexon[['INDO','WIND']],
            # era5.loc['2015':,(era5.columns.str.contains(country))&(era5.columns.str.contains('lf'))]],axis=1)
    else:
        dict_cty[country] = pd.concat([
            df_eq.loc[:,df_eq.columns.str.contains(country)],
            wx_ts.loc[:,wx_ts.columns.str.contains(country)],
            entsoe_load[country+' CTY'],
            entsoe_gen[country+' CTY'],
            era5.loc['2015':,era5.columns.str.contains(country)]],axis=1)
            # df_load_factor.loc[:,df_load_factor.columns.str.contains(country)],
            # df_load_factor_backcast.loc[:,df_load_factor_backcast.columns.str.contains(country)],
            # lf_ts.loc[:,lf_ts.columns.str.contains(country)],
            # entsoe_load[country+' CTY'],
            # # entsoe_gen[country+' CTY'],
            # entsoe_lf.loc[:,entsoe_lf.columns.str.contains(country)].add_suffix('_entsoe'),
            # era5.loc['2015':,(era5.columns.str.contains(country))&(era5.columns.str.contains('lf'))]],axis=1)

    if country in ['DE','GB']:
    # if country in ['DE']:
        if country == 'DE':
            dict_cty[country]['Total Wind'] = dict_cty[country]['Wind Onshore'] + dict_cty[country]['Wind Offshore']
        dict_cty[country][country+'_Total_Wind'] = dict_cty[country][country+'_Onshore'] + dict_cty[country][country+'_Offshore']
        dict_cty[country][country+'_Total_Wind_Backcast'] = dict_cty[country][country+'_Onshore_Backcast'] + dict_cty[country][country+'_Offshore_Backcast']
        dict_cty[country][country+'_Total_Wind_wx'] = dict_cty[country][country+'_Onshore_wx'] + dict_cty[country][country+'_Offshore_wx']

# #%%
# # save dict_cty as pickle
# with open('dict_cty.pickle', 'wb') as handle:
#     pickle.dump(dict_cty, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%
# # open dict_cty from pickle
# with open('dict_cty.pickle', 'rb') as handle:
#     dict_cty = pickle.load(handle)

# %%
for key, value in dict_cty.items():
    fig = value.resample('MS').mean().plot()
    fig.show()

# %%
# for key, value in dict_cty.items():
#     fig = value.resample('YS').mean().plot(kind='bar', barmode='group')
#     fig.show()    


#%%
# residuals
dict_residuals_demand = {}
dict_residuals_solar = {}
dict_residuals_onshore = {}
dict_residuals_offshore = {}
dict_residuals_wind = {}
for key, value in dict_cty.items():
    if key == 'GB':
        print(key+' --- Consumption')
        dict_residuals_demand[key] = value.loc[:,[
            key+'_Consumption',
            key+'_Consumption_Backcast',
            key+'_Consumption_wx',
        ]].sub(value.loc[:,'INDO'], axis=0)

        # print(key+' --- Total')
        # dict_residuals_wind[key] = value.loc[:,[
        #     key+'_Total_Wind',
        #     key+'_Total_Wind_Backcast',
        #     key+'_Total_Wind_wx_lf',
        #     key+'_Wind_Total_RealData',
        #     key+'_Wind_Total_perc50',
        # ]].sub(value.loc[:,'WIND'], axis=0)
    
    else:
        print(key+' --- Consumption')
        dict_residuals_demand[key] = value.loc[:,[
            key+'_Consumption',
            key+'_Consumption_Backcast',
            key+'_Consumption_wx',
        ]].sub(value.loc[:,key+' CTY'], axis=0)

        print(key+' --- Solar')
        dict_residuals_solar[key] = value.loc[:,[
            key+'_Solar',
            key+'_Solar_Backcast',
            key+'_Solar_wx',
            key+'_PV_Total_RealData',
            key+'_PV_Total_perc50',
        ]].sub(value.loc[:,'Solar'], axis=0)

        if key in ['BE','NL','DK']:
            print(key+' --- Onshore')
            dict_residuals_onshore[key] = value.loc[:,[
                key+'_Onshore',
                key+'_Onshore_Backcast',
                key+'_Onshore_wx',
                key+'_Wind_Onshore_RealData',
                key+'_Wind_Onshore_perc50',
            ]].sub(value.loc[:,'Wind Onshore'], axis=0)

            print(key+' --- Offshore')
            dict_residuals_offshore[key] = value.loc[:,[
                key+'_Offshore',
                key+'_Offshore_Backcast',
                key+'_Offshore_wx',
                key+'_Wind_Offshore_RealData',
                key+'_Wind_Offshore_perc50',
            ]].sub(value.loc[:,'Wind Offshore'], axis=0)
            
        elif key in ['DE']:
            print(key+' --- Onshore')
            dict_residuals_onshore[key] = value.loc[:,[
                key+'_Onshore',
                key+'_Onshore_Backcast',
                key+'_Onshore_wx',
            ]].sub(value.loc[:,'Wind Onshore'], axis=0)

            print(key+' --- Offshore')
            dict_residuals_offshore[key] = value.loc[:,[
                key+'_Offshore',
                key+'_Offshore_Backcast',
                key+'_Offshore_wx',
            ]].sub(value.loc[:,'Wind Offshore'], axis=0)

            print(key+' --- Total')
            dict_residuals_wind[key] = value.loc[:,[
                key+'_Total_Wind',
                key+'_Total_Wind_Backcast',
                key+'_Total_Wind_wx',
                key+'_Wind_Total_RealData',
                key+'_Wind_Total_perc50',
            ]].sub(value.loc[:,'Total Wind'], axis=0)

        else:
            print(key+' --- Onshore')
            dict_residuals_onshore[key] = value.loc[:,[
                key+'_Onshore',
                key+'_Onshore_Backcast',
                key+'_Onshore_wx',
                key+'_Wind_Total_RealData',
                key+'_Wind_Total_perc50',
            ]].sub(value.loc[:,'Wind Onshore'], axis=0)

    
# %%
# # plot histogram of residuals for demand
# for key, residuals in dict_residuals_demand.items():
#     fig = go.Figure()
#     print(key+' ---- Demand')
#     for col in residuals.columns:
#         bin_width= 10
#         nbins = int((residuals.loc['2016':'2021',col].max() - residuals.loc['2016':'2021',col].min()) / bin_width)
#         fig.add_trace(go.Histogram(
#             name=col, 
#             x=residuals.loc['2016':'2021',col],
#             # histnorm='probability density',
#             nbinsx=nbins,
#             # marker={
#         #         'line':{'width':2,
#         #         'shape':'hvh',
#                 # }
#             #  },
#         ))

#     # Overlay both histograms
#     fig.update_layout(barmode='overlay')
#     # Reduce opacity to see both histograms
#     fig.update_traces(opacity=0.5)
#     fig.show()

# # %%
# # plot histogram of residuals for solar
# for key, residuals in dict_residuals_solar.items():
#     fig = go.Figure()
#     print(key+' ---- Solar')
#     for col in residuals.columns:
#         bin_width= 10
#         nbins = int((residuals.loc['2016':'2021',col].max() - residuals.loc['2016':'2021',col].min()) / bin_width)
#         fig.add_trace(go.Histogram(
#             name=col, 
#             x=residuals.loc['2016':'2021',col],
#             # histnorm='probability density',
#             nbinsx=nbins,
#             # marker={
#         #         'line':{'width':2,
#         #         'shape':'hvh',
#                 # }
#             #  },
#         ))

#     # Overlay both histograms
#     fig.update_layout(barmode='overlay')
#     # Reduce opacity to see both histograms
#     fig.update_traces(opacity=0.5)
#     fig.show()

# # %%
# # plot histogram of residuals for onshore
# for key, residuals in dict_residuals_onshore.items():
#     fig = go.Figure()
#     print(key+' ---- Onshore')
#     for col in residuals.columns:
#         bin_width= 10
#         nbins = int((residuals.loc['2016':'2021',col].max() - residuals.loc['2016':'2021',col].min()) / bin_width)
#         fig.add_trace(go.Histogram(
#             name=col, 
#             x=residuals.loc['2016':'2021',col],
#             # histnorm='probability density',
#             nbinsx=nbins,
#             # marker={
#         #         'line':{'width':2,
#         #         'shape':'hvh',
#                 # }
#             #  },
#         ))

#     # Overlay both histograms
#     fig.update_layout(barmode='overlay')
#     # Reduce opacity to see both histograms
#     fig.update_traces(opacity=0.5)
#     fig.show()

# # %%
# # plot histogram of residuals for offshore
# for key, residuals in dict_residuals_offshore.items():
#     fig = go.Figure()
#     print(key+' ---- Offshore')
#     for col in residuals.columns:
#         bin_width= 10
#         nbins = int((residuals.loc['2016':'2021',col].max() - residuals.loc['2016':'2021',col].min()) / bin_width)
#         fig.add_trace(go.Histogram(
#             name=col, 
#             x=residuals.loc['2016':'2021',col],
#             # histnorm='probability density',
#             nbinsx=nbins,
#             # marker={
#         #         'line':{'width':2,
#         #         'shape':'hvh',
#                 # }
#             #  },
#         ))

#     # Overlay both histograms
#     fig.update_layout(barmode='overlay')
#     # Reduce opacity to see both histograms
#     fig.update_traces(opacity=0.5)
#     fig.show()
# #%%
# # plot histogram of residuals for total wind
# for key, residuals in dict_residuals_wind.items():
#     fig = go.Figure()
#     print(key+' ---- Total Wind')
#     for col in residuals.columns:
#         bin_width= 10
#         nbins = int((residuals.loc['2016':'2021',col].max() - residuals.loc['2016':'2021',col].min()) / bin_width)
#         fig.add_trace(go.Histogram(
#             name=col, 
#             x=residuals.loc['2016':'2021',col],
#             # histnorm='probability density',
#             nbinsx=nbins,
#             # marker={
#         #         'line':{'width':2,
#         #         'shape':'hvh',
#                 # }
#             #  },
#         ))

#     # Overlay both histograms
#     fig.update_layout(barmode='overlay')
#     # Reduce opacity to see both histograms
#     fig.update_traces(opacity=0.5)
#     fig.show()


# %%
# metrics demand
print('R_squared:')
for key, data in dict_cty.items():
    data = data.loc[data[dict_residuals_demand[key].columns].dropna(how='any').index,:]
    if key =='GB':
        # R2
        for col in dict_residuals_demand[key].columns:
            print(col+(max([len(col) for col in data.columns])-len(col)+10)*'.'+'\t'+'{:.4f}'\
                .format(metrics.r2_score(
                    data.loc[(~data[col].isna())&(~data['INDO'].isna()),col],
                    data.loc[(~data[col].isna())&(~data['INDO'].isna()),'INDO']
                )))   
    else:
        # R2
        for col in dict_residuals_demand[key].columns:
            print(col+(max([len(col) for col in data.columns])-len(col)+10)*'.'+'\t'+'{:.4f}'\
                .format(metrics.r2_score(
                    data.loc[(~data[col].isna())&(~data[key+' CTY'].isna()),col],
                    data.loc[(~data[col].isna())&(~data[key+' CTY'].isna()),key+' CTY']
                )))   

print('RMSE:')
for key, data in dict_cty.items():
    data = data.loc[data[dict_residuals_demand[key].columns].dropna(how='any').index,:]
    if key =='GB':
        # Root mean squared error
        for col in dict_residuals_demand[key].columns:
            print(col+(max([len(col) for col in data.columns])-len(col)+10)*'.'+'\t'+'{:.0f}'\
                .format(metrics.mean_squared_error(
                    data.loc[(~data[col].isna())&(~data['INDO'].isna()),col],
                    data.loc[(~data[col].isna())&(~data['INDO'].isna()),'INDO'],
                    squared=False
                )))
    else:
        # Root mean squared error
        for col in dict_residuals_demand[key].columns:
            print(col+(max([len(col) for col in data.columns])-len(col)+10)*'.'+'\t'+'{:.0f}'\
                .format(metrics.mean_squared_error(
                    data.loc[(~data[col].isna())&(~data[key+' CTY'].isna()),col],
                    data.loc[(~data[col].isna())&(~data[key+' CTY'].isna()),key+' CTY'],
                    squared=False
                )))

print('MAPE:')
for key, data in dict_cty.items():
    data = data.loc[data[dict_residuals_demand[key].columns].dropna(how='any').index,:]
    if key =='GB':
        # Root mean squared error
        for col in dict_residuals_demand[key].columns:
            print(col+(max([len(col) for col in data.columns])-len(col)+10)*'.'+'\t'+'{:.4f}'\
                .format(metrics.mean_absolute_percentage_error(
                    data.loc[(~data[col].isna())&(~data['INDO'].isna()),col],
                    data.loc[(~data[col].isna())&(~data['INDO'].isna()),'INDO'],
                )))
    else:
        # Root mean squared error
        for col in dict_residuals_demand[key].columns:
            print(col+(max([len(col) for col in data.columns])-len(col)+10)*'.'+'\t'+'{:.4f}'\
                .format(metrics.mean_absolute_percentage_error(
                    data.loc[(~data[col].isna())&(~data[key+' CTY'].isna()),col],
                    data.loc[(~data[col].isna())&(~data[key+' CTY'].isna()),key+' CTY'],
                )))                

# metrics solar
print('R_squared:')
for key, data in dict_cty.items():
    if key == 'GB':
        continue
    else:
    # R2
        data = data.loc[data[dict_residuals_solar[key].columns].dropna(how='any').index,:]
        for col in dict_residuals_solar[key].columns:
            print(col+(max([len(col) for col in data.columns])-len(col)+10)*'.'+'\t'+'{:.4f}'\
                .format(metrics.r2_score(
                    data.loc[(~data[col].isna())&(~data['Solar'].isna()),col],
                    data.loc[(~data[col].isna())&(~data['Solar'].isna()),'Solar']
                )))   

print('RMSE:')
for key, data in dict_cty.items():
    if key == 'GB':
        continue
    else:
    # Root mean squared error
        data = data.loc[data[dict_residuals_solar[key].columns].dropna(how='any').index,:]
        for col in dict_residuals_solar[key].columns:
            print(col+(max([len(col) for col in data.columns])-len(col)+10)*'.'+'\t'+'{:.0f}'\
                .format(metrics.mean_squared_error(
                    data.loc[(~data[col].isna())&(~data['Solar'].isna()),col],
                    data.loc[(~data[col].isna())&(~data['Solar'].isna()),'Solar'],
                    squared=False
                )))

print('MAPE:')
for key, data in dict_cty.items():
    if key == 'GB':
        continue
    else:
    # Root mean squared error
        data = data.loc[data[dict_residuals_solar[key].columns].dropna(how='any').index,:]
        for col in dict_residuals_solar[key].columns:
            print(col+(max([len(col) for col in data.columns])-len(col)+10)*'.'+'\t'+'{:.4f}'\
                .format(metrics.mean_absolute_percentage_error(
                    data.loc[(~data[col].isna())&(~data['Solar'].isna()),col],
                    data.loc[(~data[col].isna())&(~data['Solar'].isna()),'Solar'],
                )))

# metrics onshore
print('R_squared:')
for key, data in dict_cty.items():
    if key =='GB':
        continue
    else:
        # R2
        data = data.loc[data[dict_residuals_onshore[key].columns].dropna(how='any').index,:]
        for col in dict_residuals_onshore[key].columns:
            print(col+(max([len(col) for col in data.columns])-len(col)+10)*'.'+'\t'+'{:.4f}'\
                .format(metrics.r2_score(
                    data.loc[(~data[col].isna())&(~data['Wind Onshore'].isna()),col],
                    data.loc[(~data[col].isna())&(~data['Wind Onshore'].isna()),'Wind Onshore']
                )))   

print('RMSE:')
for key, data in dict_cty.items():
    if key =='GB':
        continue
    else:
        # Root mean squared error
        data = data.loc[data[dict_residuals_onshore[key].columns].dropna(how='any').index,:]
        for col in dict_residuals_onshore[key].columns:
            print(col+(max([len(col) for col in data.columns])-len(col)+10)*'.'+'\t'+'{:.0f}'\
                .format(metrics.mean_squared_error(
                    data.loc[(~data[col].isna())&(~data['Wind Onshore'].isna()),col],
                    data.loc[(~data[col].isna())&(~data['Wind Onshore'].isna()),'Wind Onshore'],
                    squared=False
                )))            

print('MAPE:')
for key, data in dict_cty.items():
    if key =='GB':
        continue
    else:
        # Root mean squared error
        data = data.loc[data[dict_residuals_onshore[key].columns].dropna(how='any').index,:]
        for col in dict_residuals_onshore[key].columns:
            print(col+(max([len(col) for col in data.columns])-len(col)+10)*'.'+'\t'+'{:.4f}'\
                .format(metrics.mean_absolute_percentage_error(
                    data.loc[(~data[col].isna())&(~data['Wind Onshore'].isna()),col],
                    data.loc[(~data[col].isna())&(~data['Wind Onshore'].isna()),'Wind Onshore'],
                )))   

# metrics offshore
print('R_squared:')
for key, data in dict_cty.items():
    if key in ['NL', 'DE', 'BE', 'DK']:
        # R2
        data = data.loc[data[dict_residuals_offshore[key].columns].dropna(how='any').index,:]
        for col in dict_residuals_offshore[key].columns:
            print(col+(max([len(col) for col in data.columns])-len(col)+10)*'.'+'\t'+'{:.4f}'\
                .format(metrics.r2_score(
                    data.loc[(~data[col].isna())&(~data['Wind Offshore'].isna()),col],
                    data.loc[(~data[col].isna())&(~data['Wind Offshore'].isna()),'Wind Offshore']
                )))
    # R2
    else:
        continue

print('RMSE:')
for key, data in dict_cty.items():
    if key in ['NL', 'DE', 'BE', 'DK']:
        # Root mean squared error
        data = data.loc[data[dict_residuals_offshore[key].columns].dropna(how='any').index,:]
        for col in dict_residuals_offshore[key].columns:
            print(col+(max([len(col) for col in data.columns])-len(col)+10)*'.'+'\t'+'{:.0f}'\
                .format(metrics.mean_squared_error(
                    data.loc[(~data[col].isna())&(~data['Wind Offshore'].isna()),col],
                    data.loc[(~data[col].isna())&(~data['Wind Offshore'].isna()),'Wind Offshore'],
                    squared=False
                )))    
    else:
        continue

print('MAPE:')
for key, data in dict_cty.items():
    if key in ['NL', 'DE', 'BE', 'DK']:
        # Root mean squared error
        data = data.loc[data[dict_residuals_offshore[key].columns].dropna(how='any').index,:]
        for col in dict_residuals_offshore[key].columns:
            print(col+(max([len(col) for col in data.columns])-len(col)+10)*'.'+'\t'+'{:.4f}'\
                .format(metrics.mean_absolute_percentage_error(
                    data.loc[(~data[col].isna())&(~data['Wind Offshore'].isna()),col],
                    data.loc[(~data[col].isna())&(~data['Wind Offshore'].isna()),'Wind Offshore'],
                )))    
    else:
        continue

# metrics total wind
print('R_squared:')
for key, data in dict_cty.items():
    if key =='DE':
    #     # R2
    #     data = data.loc[data[dict_residuals_wind[key].columns].dropna(how='any').index,:]
    #     for col in dict_residuals_wind[key].columns:
    #         print(col+(max([len(col) for col in data.columns])-len(col)+10)*'.'+'\t'+'{:.4f}'\
    #             .format(metrics.r2_score(
    #                 data.loc[(~data[col].isna())&(~data['WIND'].isna()),col],
    #                 data.loc[(~data[col].isna())&(~data['WIND'].isna()),'WIND']
    #             )))   
    # elif key == 'DE':
        # R2
        data = data.loc[data[dict_residuals_wind[key].columns].dropna(how='any').index,:]
        for col in dict_residuals_wind[key].columns:
            print(col+(max([len(col) for col in data.columns])-len(col)+10)*'.'+'\t'+'{:.4f}'\
                .format(metrics.r2_score(
                    data.loc[(~data[col].isna())&(~data['Total Wind'].isna()),col],
                    data.loc[(~data[col].isna())&(~data['Total Wind'].isna()),'Total Wind']
                )))   

print('RMSE:')
for key, data in dict_cty.items():
    if key =='DE':
    #     # Root mean squared error
    #     data = data.loc[data[dict_residuals_wind[key].columns].dropna(how='any').index,:]
    #     for col in dict_residuals_wind[key].columns:
    #         print(col+(max([len(col) for col in data.columns])-len(col)+10)*'.'+'\t'+'{:.0f}'\
    #             .format(metrics.mean_squared_error(
    #                 data.loc[(~data[col].isna())&(~data['WIND'].isna()),col],
    #                 data.loc[(~data[col].isna())&(~data['WIND'].isna()),'WIND']
                    # squared=False
    #             )))
    # elif key == 'DE':
        # Root mean squared error
        data = data.loc[data[dict_residuals_wind[key].columns].dropna(how='any').index,:]
        for col in dict_residuals_wind[key].columns:
            print(col+(max([len(col) for col in data.columns])-len(col)+10)*'.'+'\t'+'{:.0f}'\
                .format(metrics.mean_squared_error(
                    data.loc[(~data[col].isna())&(~data['Total Wind'].isna()),col],
                    data.loc[(~data[col].isna())&(~data['Total Wind'].isna()),'Total Wind'],
                    squared=False
                )))                                                    

print('MAPE:')
for key, data in dict_cty.items():
    if key =='DE':
    #     # Root mean squared error
    #     data = data.loc[data[dict_residuals_wind[key].columns].dropna(how='any').index,:]
    #     for col in dict_residuals_wind[key].columns:
    #         print(col+(max([len(col) for col in data.columns])-len(col)+10)*'.'+'\t'+'{:.4f}'\
    #             .format(metrics.mean_squared_error(
    #                 data.loc[(~data[col].isna())&(~data['WIND'].isna()),col],
    #                 data.loc[(~data[col].isna())&(~data['WIND'].isna()),'WIND']
                    # squared=False
    #             )))
    # elif key == 'DE':
        # Root mean squared error
        data = data.loc[data[dict_residuals_wind[key].columns].dropna(how='any').index,:]
        for col in dict_residuals_wind[key].columns:
            print(col+(max([len(col) for col in data.columns])-len(col)+10)*'.'+'\t'+'{:.4f}'\
                .format(metrics.mean_absolute_percentage_error(
                    data.loc[(~data[col].isna())&(~data['Total Wind'].isna()),col],
                    data.loc[(~data[col].isna())&(~data['Total Wind'].isna()),'Total Wind'],
                )))    

# %%
# import wx data from epsi
# df_epsi= pd.read_csv('tesla_wx.csv')

# start_date = dt.date.today() - timedelta(days=10)
start_date = dt.date(2016,1,1)
end_date = dt.date.today() + dt.timedelta(days=10)
epsi_dict={
    # Demand
        'NL':87346, # tesla wx
        'ES':87347, # tesla wx
        'DE':87344, # tesla wx
        'FR':87343, # tesla wx
        'BE':87342, # tesla wx
        # 'AT',
        # 'CZ',
        # 'CH',
        # 'DK',
        # 'PT',
        'GB':87345, # tesla wx
    # # Wind
    #     'NLwind':95128, # ERA5 wx years
    #     'ESwind':95129, # ERA5 wx years
    #     'DEwind':95012, # ERA5 wx years
    #     'FRwind':95125, # ERA5 wx years
    #     'BEwind':95121, # ERA5 wx years
    #     'ATwind':95120, # ERA5 wx years
    #     'CZwind':95122, # ERA5 wx years
    #     # 'CHwind':95130, # ERA5 wx years
    #     'DKwwind':95196, # ERA5 wx years
    #     'DKewind':95123, # ERA5 wx years
    #     'PTwind':107865, # ERA5 wx years
    #     'GBwind':95131, # ERA5 wx years
    # # Solar
    #     'NLsolar':95141, # ERA5 wx years
    #     'ESsolar':95142, # ERA5 wx years
    #     'DEsolar':95132, # ERA5 wx years
    #     'FRsolar':95138, # ERA5 wx years
    #     'BEsolar':95134, # ERA5 wx years
    #     'ATsolar':95133, # ERA5 wx years
    #     'CZsolar':95135, # ERA5 wx years
    #     # 'CHsolar':95143, # ERA5 wx years
    #     'DKwsolar':95137, # ERA5 wx years
    #     'DKesolar':95136, # ERA5 wx years
    #     'PTsolar':108605, # ERA5 wx years
    #     'GBsolar':95144, # ERA5 wx years
}
d = {}
i=0
print('pulling data from epsi...')
for source, id in epsi_dict.items():
    for weather_year in range(1990,2020):
        print(str(source)+' - '+str(weather_year))
        # address = f'https://epsi.genscape.com/export/download_sas_xml?account=Shell&id={id}&update=Latest &password=pv112014&start_date={start_date}&end_date={end_date}'
        address = f'https://epsi.genscape.com/export/download_sas_xml?account=Shell&id={id}&update={weather_year}&password=pv112014'
        r = requests.get(url=address, verify=False)
        # soup_dict[weather_year] = BeautifulSoup(r.text, 'html.parser')
        soup = BeautifulSoup(r.text, 'html.parser')
        for serie in soup.find_all('series'):
            for vals in serie.find_all('value'):
                d[i] = {'region': soup.country.text,
                        'source': source,
                        'date_time': vals.get('date'),
                        'metric': soup.full_type.text,
                        'value': vals.text,
                        'units': serie.units.text,
                        'original_timezone':soup.timezone.text, # I think this is wrong, looks like UTC,
                        'weather_year':weather_year
                    }
                i+=1   
        # df_dict[weather_year] = pd.DataFrame.from_dict(d, 'index')
df_epsi = pd.DataFrame.from_dict(d, 'index')
print('done')
#%%
# localize time and set UTC axis
if len(df_epsi.original_timezone.unique()) == 1:
    df_epsi['date_time'] = pd.to_datetime(df_epsi.date_time).dt.tz_localize(df_epsi.original_timezone.unique()[0],nonexistent = 'NaT', ambiguous='NaT')
    df_epsi['UTC'] = df_epsi['date_time'].dt.tz_convert(None)
else:
    print('Error: need to apply different timezones to different rows')

    # print('timezone = '+soup.timezone.text)
# df_epsi.set_index('UTC', inplace=True)
# df_epsi['date_time'] = pd.to_datetime(df_epsi['date_time']).dt.tz_localize('UTC',nonexistent = 'NaT', ambiguous='NaT')
df_epsi.set_index('date_time', inplace=True)
# # df_epsi.index = df_epsi.index.tz_convert(None)
# %%
# pivot raw data
df_epsi['value']=pd.to_numeric(df_epsi.value)
# df_epsi_pivot = df_epsi[['source','weather_year','value']].groupby(['date_time','source','weather_year']).mean().unstack(['source','weather_year'])['value']
df_epsi_pivot = df_epsi.loc[:,['region','metric','weather_year','value']].groupby(['date_time','region','metric','weather_year']).mean().unstack(['region','metric','weather_year'])['value']
# df_epsi_pivot.columns.names=['region','weather_year']
df_epsi_pivot.columns.names=['region','metric','weather_year']
# %%
tesla_wx = {}
for year in df_epsi_pivot.index.year.unique():
    tesla_wx[year]=df_epsi_pivot.loc[str(year),df_epsi_pivot.columns.get_level_values('weather_year')==str(year)].droplevel(level=['metric','weather_year'],axis=1)
df_tesla_ts = pd.concat(tesla_wx, axis=0)
# %%
