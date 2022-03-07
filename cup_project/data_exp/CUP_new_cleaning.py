#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:04:23 2019

@author: modal
"""

import pandas as pd
import numpy as np
import pickle

import CUP_loading_2v1 as cupload
#%% IMPORT CONFIG FILE
import configparser
configparser = configparser.RawConfigParser()   
configFilePath ="config.ini"
configparser.read(configFilePath)

# LOAD DEI DATASET
data_path = configparser.get('CUP-config', 'data_path')
csv_path = configparser.get('CUP-config', 'csv_path')
pickle_path = configparser.get('CUP-config', 'pickle_path')


#%% Pickle of PRESTAZIONI

df_cup = cupload.load_pickle(pickle_path+'cup_one_new.pickle')

df_cassa = cupload.load_pickle(pickle_path+'cassa_one_new.pickle')
df_annul = cupload.load_pickle(pickle_path+'cup_annul_one_new.pickle')
df_branche = cupload.load_pickle(data_path+'branche.pickle')

#%% CHECK BUCHI
import matplotlib.pyplot as plt
cols = ['sa_data_ins','sa_data_app']
asls = df_cup.sa_asl.unique()
asls.sort()

banana =   pd.DataFrame({
    'date': df_cup['sa_data_ins'].value_counts().index,
    }).sort_values(by=['date']).set_index('date').resample('D').sum()

for asl in asls:
    for col in cols: 
        banana['A'] = df_cup[df_cup.sa_asl=='A'][col].value_counts()
        banana['B'] = df_cup[df_cup.sa_asl=='B'][col].value_counts()
        banana['C'] = df_cup[df_cup.sa_asl=='C'][col].value_counts()
        banana = banana.fillna(0)

        title = "Nuovo db: {} - {}".format(asl,col)
        banana[asl].plot(figsize=(15,5), title=title)
        plt.savefig(title + '.png', dpi=300)
        plt.show()

#%%
df_cup_old = pd.read_csv(csv_path+'df_cup_c.csv')

for col in ['sa_data_ins', 'sa_data_app', 'sa_data_pren', 'sa_data_prescr']:
    df_cup_old[col] = pd.to_datetime(df_cup_old[col], errors='coerce')

df_cup_old = df_cup_old[df_cup_old['flag_stato'] == True]

#%%
import matplotlib.pyplot as plt
cols = ['sa_data_ins','sa_data_app']
asls = df_cup_old.sa_asl.unique()
asls.sort()

banana_old =   pd.DataFrame({
    'date': df_cup_old['sa_data_ins'].value_counts().index,
    }).sort_values(by=['date']).set_index('date').resample('D').sum()

for asl in asls:
    for col in cols: 
        banana_old['A'] = df_cup_old[df_cup_old.sa_asl=='A'][col].value_counts()
        banana_old['B'] = df_cup_old[df_cup_old.sa_asl=='B'][col].value_counts()
        banana_old['C'] = df_cup_old[df_cup_old.sa_asl=='C'][col].value_counts()
        banana_old = banana_old.fillna(0)

        title = "Vecchio db: {} - {}".format(asl,col)
        banana_old[asl].plot(figsize=(15,5), title=title)
        plt.savefig(title + '.png', dpi=300)
        plt.show()


#%%

print(df_cup.columns)

cols = ['sa_data_ins', 'sa_data_pren', 'sa_data_app', 'sa_data_prescr', 'sa_ass_cf', 'sa_branca_id', 'sa_pre_id', 'sa_impegnativa_id', 'sa_num_prestazioni', 'sa_sesso_id']

df_cup_old[cols].info()

df_cup[cols].info()

df_cup['sa_sesso_id'] = df_cup['sa_sesso_id'].astype('float64')
df_cup['sa_impegnativa_id'] = df_cup['sa_impegnativa_id'].astype('int64')
df_cup['sa_num_prestazioni'] = df_cup['sa_num_prestazioni'].astype('int64')
df_cup['sa_ass_cf'] = df_cup['sa_ass_cf'].astype('int64')

s1 = pd.merge(df_cup_old[cols], df_cup[cols], how='inner', on=cols)

df_cup_1 = df_cup_old[cols].drop_duplicates()
df_cup_2 = df_cup[cols].drop_duplicates()

s2 = pd.merge(df_cup_1, df_cup_2, how='inner', on=cols)

s2_uniq = df_cup_1.merge(df_cup_2, indicator=True, how='left').loc[lambda x : x['_merge']!='both']
s2_uniq_2 = df_cup_2.merge(df_cup_1, indicator=True, how='left').loc[lambda x : x['_merge']!='both']