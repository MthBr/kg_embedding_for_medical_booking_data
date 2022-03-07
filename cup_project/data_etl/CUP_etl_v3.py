#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 12:00:57 2019

@author: modal
"""

import pandas as pd
import numpy as np
import pickle

# LOAD DEI DATASET
file_path = '../../DATASET/'
#df_branche = pd.read_csv(file_path+'branche.csv', sep=',')
#df_prestazioni = pd.read_csv(file_path+'prestazioni.csv', sep=',')
df_quartieri = pd.read_csv(file_path+'quartieri.csv', sep=',')
df_comuni = pd.read_csv(file_path+'comuni.csv', sep=',')

#df_cup = pd.read_csv(file_path+'cup.csv', sep=',')
df_cup = pd.DataFrame()
for chunk in pd.read_csv(file_path+'cup.csv', chunksize=1000000):
    df_cup = pd.concat([df_cup, chunk], ignore_index=True)

del chunk    
#df_cup = pd.read_csv(file_path+'cup.csv', nrows = 20000)

df_cup = df_cup.rename(columns={'descrizione':'branca', 'descrizione.1':'prestazione'})
#%% DATASET DI TEST
print('Creazione dataset di test', end="")
df_cup_test = df_cup.copy()
del df_cup
# Eliminazione colonne non significative
df_cup_test['branca'] = df_cup_test['branca'].str.upper() 
df_cup_test['prestazione'] = df_cup_test['prestazione'].str.upper() 
df_cup_test = df_cup_test.drop(['sa_deleted', 'sa_data_del','sa_branca_id','sa_pre_id','id_branca','codice'], axis=1)
df_cup_test = df_cup_test.rename(columns={'branca':'sa_branca_id', 'prestazione':'sa_pre_id'})
# Conversione delle date in datetime
cols = ['sa_data_ins','sa_data_pren','sa_data_app','sa_data_prescr']

df_cup_test[cols] = df_cup_test[cols].apply(pd.to_datetime, errors = 'coerce')
for col in cols:
    df_cup_test[col+'_isNULL'] = pd.isnull(df_cup_test[col])
    
df_cup_test = df_cup_test[(df_cup_test['sa_data_ins_isNULL'] == False) &
                          (df_cup_test['sa_data_pren_isNULL'] == False) &
                          (df_cup_test['sa_data_app_isNULL'] == False) &
                          (df_cup_test['sa_data_prescr_isNULL'] == False)] 

df_cup_test = df_cup_test.drop(['sa_data_ins_isNULL','sa_data_pren_isNULL','sa_data_app_isNULL','sa_data_prescr_isNULL'], axis=1)

# Eliminazione dei record con dati fuori dal mondo
df_cup_test = df_cup_test[df_cup_test['sa_eta_id']>=0]
df_cup_test = df_cup_test[df_cup_test['sa_gg_attesa_pdisp']>=0]
df_cup_test = df_cup_test[df_cup_test['sa_gg_attesa']>=0]
df_cup_test = df_cup_test[df_cup_test['sa_gg_attesa'] >= df_cup_test['sa_gg_attesa_pdisp']]
df_cup_test = df_cup_test[df_cup_test['sa_data_ins'].dt.year>=2013]

# Parsing dei comuni
comuni = df_comuni.set_index('Codice Comune formato numerico')['Provincia'].T.to_dict()
df_cup_test.sa_comune_id = df_cup_test.sa_comune_id.map(comuni)

# Giorni di attesa come interi
df_cup_test['sa_gg_attesa'] = df_cup_test['sa_gg_attesa'].astype(int)
df_cup_test['sa_gg_attesa_pdisp'] = df_cup_test['sa_gg_attesa_pdisp'].astype(int)

print('DONE')

print('length: ',len(df_cup_test))
#%% SALVATAGGIO SU FILES PICKLE
print('Salvataggio su files PICKLE...', end="")
pickling_on = open(file_path+"df_cup.pickle","wb")
pickle.dump(df_cup_test, pickling_on)
pickling_on.close()

#pickling_on = open("DATASET/CLEAN/df_branche.pickle","wb")
#pickle.dump(df_branche, pickling_on)
#pickling_on.close()
#
#pickling_on = open("DATASET/CLEAN/df_prestazioni.pickle","wb")
#pickle.dump(df_prestazioni, pickling_on)
#pickling_on.close()
print('DONE')