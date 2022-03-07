#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:26:43 2019

Extraction of the bad 'impegnativa' with multiple 'medico' or multiple 'assistito'.

input: df_cup, df_annul, df_cassa

output: df_sub_bad

@author: 
"""

#%%
import numpy as np
import pandas as pd

import configparser

configparser = configparser.RawConfigParser()   
configFilePath ="config.ini"
configparser.read(configFilePath)

data_path = configparser.get('CUP-config', 'data_path')

import CUP_loading_2v1 as cupload

#%% Load of the dataset

print('Loading the datasets:')
print('- CUP')
df_cup = cupload.load_pickle(data_path+'cup.pickle')
print('- ANNUL')
df_annul = cupload.load_pickle(data_path+'annul.pickle')
print('- CASSA')
df_cassa = cupload.load_pickle(data_path+'cassa.pickle')
print('done!')
print('')

#%% Estraction of the duplicated 'impegnative' field by 'medici' and/or 'assistiti'

# Dataframe 'impegnativa-medico-assistito'
print('Extracting the \'impegnativa-medico-assistito\' fields...', end='')
df_cup_sub = df_cup[['sa_impegnativa_id','sa_med_id','sa_ass_cf']]
df_annul_sub = df_annul[['sa_impegnativa_id','sa_med_id','sa_ass_cf']]
df_cassa_sub = df_cassa[['sa_impegnativa_id','sa_med_id','sa_ass_cf']]

print('dropping duplicates...', end='')
df_cup_sub_nodup = df_cup_sub.drop_duplicates()
df_annul_sub_nodup = df_annul_sub.drop_duplicates()
df_cassa_sub_nodup = df_cassa_sub.drop_duplicates()
print('done!')

# Dataframe Impegnativa-medico
print('Extracting the \'impegnativa-medico\' pair...', end='')
df_cup_sub_impegn_med = df_cup_sub_nodup[['sa_impegnativa_id','sa_med_id']]
df_annul_sub_impegn_med = df_annul_sub_nodup[['sa_impegnativa_id','sa_med_id']]
df_cassa_sub_impegn_med = df_cassa_sub_nodup[['sa_impegnativa_id','sa_med_id']]

print('dropping duplicates...', end='')
df_cup_sub_impegn_med_nodup = df_cup_sub_impegn_med.drop_duplicates()
df_annul_sub_impegn_med_nodup = df_annul_sub_impegn_med.drop_duplicates()
df_cassa_sub_impegn_med_nodup = df_cassa_sub_impegn_med.drop_duplicates()
print('done!')

# Dataframe Impegnativa-assistito
print('Extracting the \'impegnativa-assistito\' pair...', end='')
df_cup_sub_impegn_assis = df_cup_sub_nodup[['sa_impegnativa_id','sa_ass_cf']]
df_annul_sub_impegn_assis = df_annul_sub_nodup[['sa_impegnativa_id','sa_ass_cf']]
df_cassa_sub_impegn_assis = df_cassa_sub_nodup[['sa_impegnativa_id','sa_ass_cf']]

print('dropping duplicates...', end='')
df_cup_sub_impegn_assis_nodup = df_cup_sub_impegn_assis.drop_duplicates()
df_annul_sub_impegn_assis_nodup = df_annul_sub_impegn_assis.drop_duplicates()
df_cassa_sub_impegn_assis_nodup = df_cassa_sub_impegn_assis.drop_duplicates()
print('done!')

#%% Duplicates finding
print('Counting the duplicated entries for:', end = '')
print('\'medico\' ...', end = '')
s_cup_sub_med_counts = df_cup_sub_impegn_med_nodup['sa_impegnativa_id'].value_counts()
s_annul_sub_med_counts = df_annul_sub_impegn_med_nodup['sa_impegnativa_id'].value_counts()
s_cassa_sub_med_counts = df_cassa_sub_impegn_med_nodup['sa_impegnativa_id'].value_counts()

s_cup_sub_med_dup_counts = s_cup_sub_med_counts[s_cup_sub_med_counts >= 2]
s_annul_sub_med_dup_counts = s_annul_sub_med_counts[s_annul_sub_med_counts >= 2]
s_cassa_sub_med_dup_counts = s_cassa_sub_med_counts[s_cassa_sub_med_counts >= 2]

print('\'assistito\' ...', end = '')
s_cup_sub_assis_counts = df_cup_sub_impegn_assis_nodup['sa_impegnativa_id'].value_counts()
s_annul_sub_assis_counts = df_annul_sub_impegn_assis_nodup['sa_impegnativa_id'].value_counts()
s_cassa_sub_assis_counts = df_cassa_sub_impegn_assis_nodup['sa_impegnativa_id'].value_counts()

s_cup_sub_assis_dup_counts = s_cup_sub_assis_counts[s_cup_sub_assis_counts >= 2]
s_annul_sub_assis_dup_counts = s_annul_sub_assis_counts[s_annul_sub_assis_counts >= 2]
s_cassa_sub_assis_dup_counts = s_cassa_sub_assis_counts[s_cassa_sub_assis_counts >= 2]
print('done!')

# Bad 'impegnativa' entries as indexes
print('Gathering the \'impegnativa\' containing the duplicates...', end = '')
l_cup_sub_med_dup = list(s_cup_sub_med_dup_counts.index)
l_annul_sub_med_dup = list(s_annul_sub_med_dup_counts.index)
l_cassa_sub_med_dup = list(s_cassa_sub_med_dup_counts.index)

l_cup_sub_assis_dup = list(s_cup_sub_assis_dup_counts.index)
l_annul_sub_assis_dup = list(s_annul_sub_assis_dup_counts.index)
l_cassa_sub_assis_dup = list(s_cassa_sub_assis_dup_counts.index)

l_cup_sub_union_dup = list(set(l_cup_sub_med_dup).union(set(l_cup_sub_assis_dup)))
l_annul_sub_union_dup = list(set(l_annul_sub_med_dup).union(set(l_annul_sub_assis_dup)))
l_cassa_sub_union_dup = list(set(l_cassa_sub_med_dup).union(set(l_cassa_sub_assis_dup)))
print('done!')

#%% Filtered databases
print('Creating the output databases:')
print('- CUP')
df_cup_sub_bad = df_cup_sub[df_cup_sub['sa_impegnativa_id'].isin(l_cup_sub_union_dup)]

df_cup_bad_impegn = df_cup_sub_bad.copy()
df_cup_bad_impegn['multipli_medici'] = np.where(df_cup_bad_impegn['sa_impegnativa_id'].isin(l_cup_sub_med_dup), True, False)
df_cup_bad_impegn['multipli_assistiti'] = np.where(df_cup_bad_impegn['sa_impegnativa_id'].isin(l_cup_sub_assis_dup), True, False)
df_cup_bad_impegn['in_cup'] = True
df_cup_bad_impegn['in_annul'] = False
df_cup_bad_impegn['in_cassa'] = False

print('- ANNUL')
df_annul_sub_bad = df_annul_sub[df_annul_sub['sa_impegnativa_id'].isin(l_annul_sub_union_dup)]

df_annul_bad_impegn = df_annul_sub_bad.copy()
df_annul_bad_impegn['multipli_medici'] = np.where(df_annul_bad_impegn['sa_impegnativa_id'].isin(l_annul_sub_med_dup), True, False)
df_annul_bad_impegn['multipli_assistiti'] = np.where(df_annul_bad_impegn['sa_impegnativa_id'].isin(l_annul_sub_assis_dup), True, False)
df_annul_bad_impegn['in_cup'] = False
df_annul_bad_impegn['in_annul'] = True
df_annul_bad_impegn['in_cassa'] = False

print('- CASSA')
df_cassa_sub_bad = df_cassa_sub[df_cassa_sub['sa_impegnativa_id'].isin(l_cassa_sub_union_dup)]

df_cassa_bad_impegn = df_cassa_sub_bad.copy()
df_cassa_bad_impegn['multipli_medici'] = np.where(df_cassa_bad_impegn['sa_impegnativa_id'].isin(l_cassa_sub_med_dup), True, False)
df_cassa_bad_impegn['multipli_assistiti'] = np.where(df_cassa_bad_impegn['sa_impegnativa_id'].isin(l_cassa_sub_assis_dup), True, False)
df_cassa_bad_impegn['in_cup'] = False
df_cassa_bad_impegn['in_annul'] = False
df_cassa_bad_impegn['in_cassa'] = True
print('Done!')

# Merge of the database
print('Concatenating the output databases...', end = '')
df_sub_bad = pd.concat([df_cup_bad_impegn,df_annul_bad_impegn,df_cassa_bad_impegn], ignore_index=True)

print('done!')
