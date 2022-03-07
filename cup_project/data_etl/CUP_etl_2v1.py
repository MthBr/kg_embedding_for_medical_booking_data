#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:59:03 2019

@author: modal
"""

import pandas as pd
import numpy as np
import pickle

import CUP_loading_2v1 as cupload

# LOAD DEI DATASET
file_path = '../../DATASET/'

types_cup = {
'sa_data_ins':          'str',
'sa_deleted' :          'str',
'sa_data_del':          'str',
'sa_ass_cf':            'str',
'sa_data_pren':         'str',
'sa_utente_id':         'str',
'sa_contratto_id':      'str',
'sa_data_app':          'str',
'sa_mese_app_id':       'str',
'sa_uop_codice_id':     'str',
'sa_comune_id':         'str',
'sa_branca_id':         'str',
'sa_pre_id':            'str',
'sa_med_id':            'str',
'sa_ese_id_lk':         'str',
'sa_sesso_id':          'str',
'sa_is_ad':             'str',
'sa_spr_id':            'str',
'sa_ut_id':             'str',
'sa_operazione':        'str',
'sa_stato_pren':        'str',
'sa_eta_id':            'int64',
'sa_impegnativa_id':    'str',
'sa_dti_id':            'str',
'sa_contatto_id':       'str',
'sa_gg_attesa':         'int64',
'sa_gg_attesa_pdisp':   'int64',
'sa_num_prestazioni':   'int64',
'sa_classe_priorita':   'str',
'sa_is_pre_eseguita':   'str',
'sa_data_prescr':       'str',
'sa_primo_accesso':     'str',
'sa_asl':               'str'
}

types_annul = {
'sa_data_ins':          'str',
'sa_deleted' :          'str',
'sa_data_del':          'str',
'sa_ass_cf':            'str',
'sa_data_pren':         'str',
'sa_utente_id':         'str',
'sa_utente_del':        'str',
'sa_contratto_id':      'str',
'sa_data_app':          'str',
'sa_mese_app_id':       'str',
'sa_uop_codice_id':     'str',
'sa_comune_id':         'str',
'sa_branca_id':         'str',
'sa_pre_id':            'str',
'sa_med_id':            'str',
'sa_ese_id_lk':         'str',
'sa_sesso_id':          'str',
'sa_is_ad':             'str',
'sa_spr_id':            'str',
'sa_ut_id':             'str',
'sa_operazione':        'str',
'sa_stato_pren':        'str',
'sa_eta_id':            'int64',
'sa_impegnativa_id':    'str',
'sa_dti_id':            'str',
'sa_contatto_id':       'str',
'sa_gg_attesa':         'int64',
'sa_gg_attesa_pdisp':   'int64',
'sa_num_prestazioni':   'int64',
'sa_classe_priorita':   'str',
'sa_data_prescr':       'str',
'sa_primo_accesso':     'str',
'sa_asl':               'str'
}

types_cassa = {
'sa_data_ins':          'str',
'sa_deleted' :          'str',
'sa_data_del':          'str',
'sa_utente_id':         'str',
'sa_cassa_id':          'str',
'sa_ass_cf':            'str',
'sa_data_prest':        'str',
'sa_mese_id':           'str',
'sa_data_mov':          'str',
'sa_mese_mov_id':       'str',
'sa_uop_codice_id':     'str',
'sa_comune_id':         'str',
'sa_branca_id':         'str',
'sa_pre_id':            'str',
'sa_med_id':            'str',
'sa_ese_id':            'str',
'sa_cntr_id':           'str',
'sa_sesso_id':          'str',
'sa_eta_id':            'int64',
'sa_is_ad':             'str',
'sa_impegnativa_id':    'str',
'sa_mov_id':            'str',
'sa_dti_pk':            'str',
'sa_dti_id':            'str',
'sa_dti_prg':           'str',
'sa_dti_is_pren':       'str',
'lordo':                'float64',
'ticket':               'float64',
'quota':                'float64',
'prestazioni':          'int64',
'prestazioni_ad':       'int64',
'importo_movimento':    'float64',
'importo_impegnativa':  'float64',
'importo_prest_impegnativa':   'float64',
'importo_quota_impegnativa':   'float64',
'sa_codice_causale':           'str',
'sa_asl':                      'str',
}

#%% Pickle of CUP
dates = ['sa_data_ins','sa_data_pren','sa_data_app','sa_data_prescr']

print('Generating Pickle CUP...',end='')
df_cup = cupload.load_dataset(file=file_path+'mis_cup.csv', chunksize=1000000, column_types = types_cup, parse_dates = dates)
#df_cup = cupload.load_dataset(file=file_path+'mis_cup.csv', chunksize=1000000, parse_dates = dates)

del dates

df_cup = df_cup.drop(['sa_deleted', 'sa_data_del'], axis=1)
df_cup = df_cup[df_cup['sa_eta_id']>=0]
df_cup = df_cup[np.abs(df_cup['sa_eta_id']-df_cup['sa_eta_id'].mean())<=3*df_cup['sa_eta_id'].std()]

df_cup.describe(include='all').to_csv('cup_describe.csv')

cupload.save_on_pickle(df_cup, file_path+'cup.pickle')
print('TERMINATOR!')

#%% Pickle of ANNUL

dates = ['sa_data_ins','sa_data_del','sa_data_pren','sa_data_app','sa_data_prescr']

print('Generating Pickle ANNUL...',end='')
#df_cup = cupload.load_dataset(file=file_path+'mis_cup.csv', chunksize=1000000, column_types = types, dates_columns = dates)
df_annul = cupload.load_dataset(file=file_path+'mis_annul.csv', chunksize=1000000, column_types = types_annul, parse_dates = dates)
del dates

df_annul.describe(include='all').to_csv('annul_describe.csv')
#df_cup = df_cup.drop(['sa_deleted', 'sa_data_del'], axis=1)
#df_cup = df_cup[df_cup['sa_eta_id']>=0]
#df_cup = df_cup[np.abs(df_cup['sa_eta_id']-df_cup['sa_eta_id'].mean())<=3*df_cup['sa_eta_id'].std()]

cupload.save_on_pickle(df_annul, file_path+'annul.pickle')
print('TERMINATOR!')
#%% Pickle of CASSA

dates = ['sa_data_ins','sa_data_prest','sa_data_mov']
print('Generating Pickle CASSA...',end='')
#df_cup = cupload.load_dataset(file=file_path+'mis_cup.csv', chunksize=1000000, column_types = types, dates_columns = dates)
df_cassa = cupload.load_dataset(file=file_path+'mis_cassa.csv', chunksize=1000000, column_types = types_cassa, parse_dates = dates)

#df_cup = df_cup.drop(['sa_deleted', 'sa_data_del'], axis=1)
#df_cup = df_cup[df_cup['sa_eta_id']>=0]
#df_cup = df_cup[np.abs(df_cup['sa_eta_id']-df_cup['sa_eta_id'].mean())<=3*df_cup['sa_eta_id'].std()]

df_cassa.describe(include='all').to_csv('cassa_describe.csv')

cupload.save_on_pickle(df_cassa, file_path+'cassa.pickle')
print('TERMINATOR!')
#%% Pickle of BRANCHE
#dates = ['sa_data_ins','sa_data_pren','sa_data_app','sa_data_prescr']

print('Generating Pickle BRANCHE...',end='')
#df_cup = cupload.load_dataset(file=file_path+'mis_cup.csv', chunksize=1000000, column_types = types, dates_columns = dates)
df_branche = cupload.load_dataset(file=file_path+'branche.csv')

#df_cup = df_cup.drop(['sa_deleted', 'sa_data_del'], axis=1)
#df_cup = df_cup[df_cup['sa_eta_id']>=0]
#df_cup = df_cup[np.abs(df_cup['sa_eta_id']-df_cup['sa_eta_id'].mean())<=3*df_cup['sa_eta_id'].std()]

cupload.save_on_pickle(df_branche, file_path+'branche.pickle')
print('TERMINATOR!')

#%% Pickle of PRESTAZIONI

#dates = ['sa_data_ins','sa_data_pren','sa_data_app','sa_data_prescr']
df_cup = cupload.load_pickle(file_path+'cup.pickle')
print('Generating Pickle PRESTAZIONI...',end='')
#df_cup = cupload.load_dataset(file=file_path+'mis_cup.csv', chunksize=1000000, column_types = types, dates_columns = dates)
df_pre1 = cupload.load_dataset(file=file_path+'prestazioni.csv')
df_pre2 = cupload.load_dataset(file=file_path+'prestazioni2.csv')
df_pre3 = cupload.load_dataset(file=file_path+'prestazioni3.csv')

#MEGALISTONE PRESTAZIONI
sex1 = df_pre2[['Codice Catalogo','Descrizione CATALOGO']]
sex1 = sex1.rename(columns={'Codice Catalogo': 'codice', 'Descrizione CATALOGO': 'descrizione'})

sex2 = df_pre2[['Codice Naz','Descrizione DM 18/10/2012']]
sex2 = sex2.rename(columns={'Codice Naz': 'codice', 'Descrizione DM 18/10/2012': 'descrizione'})

#sex3 = df_pre2[['Codice Reg','Descrizione CATALOGO']]
#sex3 = sex3.rename(columns={'Codice Reg': 'codice', 'Descrizione CATALOGO': 'descrizione'})

sex4 = df_pre3[df_pre3['TIPO OPERAZIONE']=='RECORD CANCELLATO'][['CODICE CATALOGO','DESCRIZIONE CATALOGO']]
sex4 = sex4.rename(columns={'CODICE CATALOGO': 'codice', 'DESCRIZIONE CATALOGO': 'descrizione'})

cod_reg = pd.concat([sex1,sex2,sex4], ignore_index=True).dropna().drop_duplicates()

# PRESTAZIONI DI df_pre1 CHE NON STANNO IN COD_REG, MA CHE STANNO IN df_cup
lc = df_pre1[~df_pre1['codice'].isin(cod_reg['codice']) & df_pre1['codice'].isin(df_cup['sa_pre_id'])]
lc_num = lc[lc['codice'].str.match('^[0-9\.-]*$')==True]
lc_lett = lc[~lc['codice'].isin(lc_num['codice'])]

# TUTTO COD_REG STA IN df_pre1
print(len(cod_reg[cod_reg['codice'].isin(df_pre1['codice'])]))
#cupload.save_on_pickle(df_branche, file_path+'branche.pickle')
print('TERMINATOR!')

#%%

for col in df_cup.columns:
    print('df_cup[',col,'] ->',df_cup[col].is_unique)

for col in df_annul.columns:
    print('df_annul[',col,'] ->',df_annul[col].is_unique)
    
for col in df_cassa.columns:
    print('df_cassa[',col,'] ->',df_cassa[col].is_unique)
#for cat in list(df_cup.select_dtypes(include=[object]).columns):
#    df_cup[cat] = df_cup[cat].astype('category')
#df_cup = cupload.load_pickle(file_path+'cup.pickle')

intersection = list(types_cup.keys() & types_annul.keys() & types_cassa.keys())
intersection = [x for x in intersection if x not in ['sa_deleted', 'sa_data_del']]

gino_name = '1323906'

alberta = '860654'
gino_sord = df_cassa[df_cassa['sa_ass_cf']==alberta][intersection]
gino_acas = df_annul[df_annul['sa_ass_cf']==alberta][intersection]
intersection.append('sa_ese_id_lk')
gino = df_cup[df_cup['sa_ass_cf']==alberta][intersection]
