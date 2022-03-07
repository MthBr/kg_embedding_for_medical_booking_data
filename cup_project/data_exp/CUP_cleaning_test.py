#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:59:03 2019

This imports from pickle and cleans everything

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


#%% Pickle of PRESTAZIONI

#dates = ['sa_data_ins','sa_data_pren','sa_data_app','sa_data_prescr']
df_cup = cupload.load_pickle(data_path+'cup.pickle')
sub_dict = {"'":"-",
            " ":""}

df_cassa = cupload.load_pickle(data_path+'cassa.pickle')
df_annul = cupload.load_pickle(data_path+'annul.pickle')
df_branche = cupload.load_pickle(data_path+'branche.pickle')

df_cup = cupload.sub_char(df_cup,'sa_uop_codice_id',sub_dict)
df_cassa = cupload.sub_char(df_cassa,'sa_uop_codice_id',sub_dict)
df_annul = cupload.sub_char(df_annul,'sa_uop_codice_id',sub_dict)

print('Generating Pickle PRESTAZIONI...',end='')
#df_cup = cupload.load_dataset(file=data_path+'mis_cup.csv', chunksize=1000000, column_types = types, dates_columns = dates)

#%% OPERAZIONI DI PARIFICAZIONE DELLE PRESTAZIONI

# Carico le prestazioni (database e regionali)
df_pre1 = cupload.load_dataset(file=data_path+'prestazioni.csv')
df_pre2 = cupload.load_dataset(file=data_path+'prestazioni2.csv')
df_pre3 = cupload.load_dataset(file=data_path+'prestazioni3.csv')

# Parifico i nomi di colonna
sex1 = df_pre2[['Codice Catalogo','Descrizione CATALOGO']]
sex1 = sex1.rename(columns={'Codice Catalogo': 'codice', 'Descrizione CATALOGO': 'descrizione'})

sex2 = df_pre2[['Codice Naz','Descrizione DM 18/10/2012']]
sex2 = sex2.rename(columns={'Codice Naz': 'codice', 'Descrizione DM 18/10/2012': 'descrizione'})

#sex3 = df_pre2[['Codice Reg','Descrizione CATALOGO']]
#sex3 = sex3.rename(columns={'Codice Reg': 'codice', 'Descrizione CATALOGO': 'descrizione'})

sex4 = df_pre3[df_pre3['TIPO OPERAZIONE']=='RECORD CANCELLATO'][['CODICE CATALOGO','DESCRIZIONE CATALOGO']]
sex4 = sex4.rename(columns={'CODICE CATALOGO': 'codice', 'DESCRIZIONE CATALOGO': 'descrizione'})

# Creo un DataFrame con tutte le prestazioni
cod_pre_db = df_pre1.dropna()
cod_pre_reg = pd.concat([sex1, sex2], ignore_index=True) # ,sex4
cod_pre_reg = cod_pre_reg.dropna().drop_duplicates() # ,sex4
cod_pre_reg_original = cod_pre_reg.copy()

# Definisco il vocabolario per le sostituzioni nelle descrizioni
sub_prest = {'SUCCESSIVA': 'CONTROLLO',
            'SUCCESSIVO': '',
            'INIBITA': '',
            'A DOMICILIO': '',
            'DOMICILIARE': '',
            'DOMICILIAR' : '',
            'NON UTILIZZARE': '',
            'T4 TOTALE' : 'FT4',
            'DENSITOMETRIA OSSEA LOMBARE - ASSORB. A RAGGI X' : 'DENSITOMETRIA OSSEA LOMBARE - D.E.X.A.',
            'DENSITOMETRIA OSSEA DISTALE - ASSORB. A RAGGI X' : 'DENSITOMETRIA OSSEA ULTRADISTALE - D.E.X.A.',
            'ECOGRAFIA OSTETRICA I TRIMESTR' : 'ECOGRAFIA GRAVIDANZA (1 TRIMESTRE)',
            'ECOGRAFIA OSTETRICA II TRIMEST' : 'ECOGRAFIA GRAVIDANZA (2 TRIMESTRE)',
            'ECOGRAFIA OSTETRICA III TRIMES' : 'ECOGRAFIA GRAVIDANZA (3 TRIMESTRE)',
            'MAGNESURIA' : 'MAGNESIO TOTALE [Urine 24h]',
            'VISITA INTERNISTICA - IPERTENSIONE' : 'VISITA CARDIOLOGICA',
            'VISITA INTERNISTICA CONTROLLO - IPERTENSIONE' : 'VISITA CARDIOLOGICA CONTROLLO',
            'VISITA INTERNISTICA - OBESITA' : 'VISITA DIABETOLOGICA',
            'VISITA INTERNISTICA CONTROLLO - OBESITA' : 'VISITA DIABETOLOGICA CONTROLLO',
            'VISITA INTERNISTICA' : 'VISITA GENERALE',
            'VISITA PSICOLOGICA' : 'COLLOQUIO PSICOLOGICO CLINICO',
            'VISITA RINOALLERGOLOGICA' : 'VISITA ALLERGOLOGICA NAS',
            'VISITA DI MEDICINA INTEGRATA + AGOPUNTURA' : 'ALTRA AGOPUNTURA',
            'SPORTIVA' : 'GENERALE',
            'DIETOLOGICA' : 'NUTRIZIONE CLINICA',
            'CRIOTERAPIA CON AZOTO LIQUIDO' : 'ASPORT O DEMOLIZIONE LOCALE LESIONE CON CRIOTE (PER SEDUTA)'}

cod_pre_reg['descrizione'] = cod_pre_reg['descrizione'].str.replace('VISITA OCULISTICA/ESAME COMPLESSIVO DELL\'OCCHIO', 'VISITA OCULISTICA', regex=False)

#%%
cod_pre_orfani = pd.DataFrame({
                'codice' : df_cup[~(df_cup['sa_pre_id'].isin(cod_pre_db.codice))]['sa_pre_id'].dropna().drop_duplicates()})
cod_pre_orfani['occs'] = df_cup[df_cup['sa_pre_id'].isin(cod_pre_orfani['codice'])]['sa_pre_id'].value_counts()[cod_pre_orfani['codice']].values
cod_pre_orfani['desc_in_reg'] = False
n = 2
m = 3
lett_len = 50
dist_to_check = 0.5

#voc, descr_cod_reg, descr_cod_toclean = cupload.generate_vocabulary(cod_pre_reg, cod_pre_toclean, n, m, lett_len)

# Funzione che restituisce vocabolario e lista delle descrizioni per la misura
voc, descr_cod_db, descr_cod_orfani = cupload.generate_vocabulary(cod_pre_db['codice'],
                                                                    cod_pre_orfani['codice'],
                                                                    n, m, lett_len)

cod_pre_orfani = cupload.fill_df_cleaned(cupload.measure_distances(descr_cod_db, descr_cod_orfani, voc), 
                                          cod_pre_orfani, cod_pre_db, 'codice_db', dist_to_check)

voc, descr_cod_reg, descr_cod_orfani = cupload.generate_vocabulary(cod_pre_reg_original['codice'],
                                                                    cod_pre_orfani['codice'],
                                                                    n, m, lett_len)

cod_pre_orfani = cupload.fill_df_cleaned(cupload.measure_distances(descr_cod_reg, descr_cod_orfani, voc), 
                                          cod_pre_orfani, cod_pre_reg_original, 'codice_reg', dist_to_check)
df_dict = pd.DataFrame(columns = ['codice_old', 'codice'])
df_dict[['codice_old', 'codice']] = cod_pre_orfani[['codice','codice_reg_codice_reg']][cod_pre_orfani.codice_reg_cleaned]
df_dict2 = pd.DataFrame(columns = ['codice_old', 'codice'])
df_dict2[['codice_old', 'codice']] = cod_pre_orfani[['codice','codice_db_codice_reg']][(~cod_pre_orfani.codice_reg_cleaned) & (cod_pre_orfani.codice_db_cleaned)]
df_dict = df_dict.append(df_dict2,ignore_index=True).astype(str)
df_dict = df_dict.set_index('codice_old').to_dict()['codice']

#Non-Exhaustive Mapping
#If you have a non-exhaustive mapping and wish to retain the existing variables for non-matches, you can add fillna:
df_cup['sa_pre_id'] = df_cup['sa_pre_id'].map(df_dict).fillna(df_cup['sa_pre_id'])
#%%

cod_pre_db['cod_in_reg'] = cod_pre_db['codice'].isin(cod_pre_reg['codice'])
cod_pre_db['cod_in_cup'] = cod_pre_db['codice'].isin(df_cup['sa_pre_id'])

cod_pre_reg = cod_pre_reg.drop_duplicates(subset = 'descrizione', keep = 'first')


# DATAFRAME che contiene le prestazioni i cui codici non sono nella tabella regionale, ma sono presenti nel database CUP+
cod_pre_toclean = cod_pre_db[['codice', 'descrizione']][(cod_pre_db['cod_in_reg']==False) & (cod_pre_db['cod_in_cup']==True)].dropna()
#sd = df_cup[df_cup['sa_pre_id'].isin(cod_pre_toclean['codice'])]['sa_pre_id'].value_counts()
#cod_pre_toclean['occs'] = sd[cod_pre_toclean['codice']].values
cod_pre_toclean['occs'] = df_cup[df_cup['sa_pre_id'].isin(cod_pre_toclean['codice'])]['sa_pre_id'].value_counts()[cod_pre_toclean['codice']].values
cod_pre_toclean.descrizione = cod_pre_toclean.descrizione.str.strip().str.upper()#.str.replace('  ',' ')
cod_pre_reg.descrizione = cod_pre_reg.descrizione.str.strip().str.upper()
cod_pre_toclean['desc_in_reg'] = cod_pre_toclean['descrizione'].isin(cod_pre_reg['descrizione'])
#return df_toclean, df_list_good, df_list_bad

cod_pre_toclean = cupload.col_preprocessing(cod_pre_toclean, 'descrizione', type_col='descr', sub_dict = sub_prest)
cod_pre_toclean = cupload.col_preprocessing(cod_pre_toclean, 'codice', type_col='cod')

print('TERMINATOR!')

#%%
n = 4
m = 5
lett_len = 5
dist_to_check = 0.38

#voc, descr_cod_reg, descr_cod_toclean = cupload.generate_vocabulary(cod_pre_reg, cod_pre_toclean, n, m, lett_len)

# Funzione che restituisce vocabolario e lista delle descrizioni per la misura
voc, descr_cod_reg, descr_cod_toclean = cupload.generate_vocabulary(cod_pre_reg['descrizione'],
                                                                    cod_pre_toclean[~cod_pre_toclean['desc_in_reg']]['descrizione'],
                                                                    n, m, lett_len)

cod_pre_toclean = cupload.fill_df_cleaned(cupload.measure_distances(descr_cod_reg, descr_cod_toclean, voc), 
                                          cod_pre_toclean, cod_pre_reg, 'descrizione', dist_to_check)
#%%
#cod_pre_toclean['codice_mod'] = np.nan
#cod_pre_toclean['codice_mod'] = cod_pre_toclean['codice'][~cod_pre_toclean['desc_cleaned']].str.replace('[^a-zA-Z0-9]','')
#cod_pre_toclean['codice_mod'][~cod_pre_toclean['codice_mod'].astype(str).str.isnumeric()] = np.nan
        
cod_pre_reg = cod_pre_reg_original.copy()
cod_pre_reg['descrizione'] = cod_pre_reg['descrizione'].str.replace('VISITA OCULISTICA/ESAME COMPLESSIVO DELL\'OCCHIO', 'VISITA OCULISTICA', regex=False)
#cod_pre_reg = cod_pre_reg.drop_duplicates(subset = 'descrizione', keep = 'last')

col_const = 'descrizione_cleaned' #'descrizione_cleaned' 'desc_in_reg'
dist = cupload.measure_distances(cod_pre_reg['codice'].str.replace('[^a-zA-Z0-9]', '').str.strip().to_list(), cod_pre_toclean[~cod_pre_toclean[col_const]]['codice'].to_list())

cod_pre_toclean = cupload.fill_df_cleaned(dist, cod_pre_toclean, cod_pre_reg, 'codice', dist_to_check=0.13, col_constrain = col_const)



#%%
#df_toclean, df_list_good, column = cod_pre_toclean, cod_pre_reg, 'codice'

last_col = 'codice_cleaned'

tot = len(df_cup['sa_pre_id'][~df_cup['sa_pre_id'].isin(cod_pre_reg_original['codice'])].dropna())
tot_eq = cod_pre_toclean['occs'][cod_pre_toclean['desc_in_reg'] == True].sum()
tot_merg = cod_pre_toclean['occs'][(cod_pre_toclean[last_col] == True) & (cod_pre_toclean['desc_in_reg'] == False)].sum()
print('\n')
print('Prima Fase: Descrizione UGUALE\n')
print('Abbiamo recuperato il ' + str(round(100*tot_eq/tot,1)) + '%\n')
print('\n')
print('Seconda Fase: Descrizione o Codice SIMILE con una distanza inferiore a ' + str(dist_to_check) + '\n')
print('Abbiamo recuperato un altro ' + str(round(100*tot_merg/tot,1)) + '% per un totale di ' + str(round(100*(tot_merg+tot_eq)/tot,1)) + '%\n')
print('\n')
print('Perdiamo in TOTALE: ' + str(tot-tot_eq-tot_merg) +' record, cioè il ' + str(round(100*(tot-tot_eq-tot_merg)/len(df_cup),1)) + '% del totale dei record con descrizioni sballate\n')
#%%
cod_pre_toclean.to_csv('to_clean.csv')
a = list(df_cup['sa_ass_cf'][df_cup['sa_pre_id'].isin(list(cod_pre_toclean['codice'][~cod_pre_toclean[last_col]]))].unique())

#%%
# ['sa_pre_id'].dropna()
cod_pre_cleaned = pd.DataFrame(columns = ['codice_old','descrizione_old'])
cod_pre_cleaned[['codice_old', 'descrizione_old']] = cod_pre_db[['codice', 'descrizione']][cod_pre_db['cod_in_reg']==True].dropna()
cod_pre_cleaned = pd.merge(cod_pre_cleaned, cod_pre_reg_original, how='inner', left_on=['codice_old'], right_on=['codice'])

cod_pre_reg = cod_pre_reg_original.copy()
cod_pre_reg.descrizione = cod_pre_reg.descrizione.str.strip().str.upper()
cod_pre_reg['descrizione'] = cod_pre_reg['descrizione'].str.replace('VISITA OCULISTICA/ESAME COMPLESSIVO DELL\'OCCHIO', 'VISITA OCULISTICA', regex=False)
cod_pre_reg = cod_pre_reg.drop_duplicates(subset = 'descrizione', keep = 'first')

#Descrizioni uguali in Tabella Regionale
cod_pre_temp1 = pd.DataFrame(columns = ['codice_old','descrizione_old','descrizione'])
cod_pre_temp1[['codice_old', 'descrizione_old','descrizione']] = cod_pre_toclean[['codice_old', 'descrizione_old', 'descrizione']][cod_pre_toclean['desc_in_reg']].dropna()
#cod_pre_temp1 = pd.merge(cod_pre_temp1, cod_pre_reg, how='inner', left_on=['descrizione'], right_on=['descrizione'])
cod_pre_temp1 = pd.merge(cod_pre_temp1, cod_pre_reg, on='descrizione', how = 'left')
cod_pre_temp1 = cod_pre_temp1[cod_pre_cleaned.columns.tolist()]

#Descrizioni vicine a descizioni in Tabella Regionale
cod_pre_temp2 = pd.DataFrame(columns = ['codice_old','descrizione_old','codice', 'descrizione'])
cod_pre_temp2[['codice_old', 'descrizione_old', 'codice', 'descrizione']] = cod_pre_toclean[['codice_old', 'descrizione_old', 
             'descrizione_codice_reg', 'descrizione_descrizione_reg']][(cod_pre_toclean['desc_in_reg']==False) &
                (cod_pre_toclean['descrizione_cleaned']==True)].dropna()
cod_pre_temp2 = cod_pre_temp2[cod_pre_cleaned.columns.tolist()]

#Codici vicini a codici in Tabella Regionale
cod_pre_temp3 = pd.DataFrame(columns = ['codice_old','descrizione_old','codice', 'descrizione'])
cod_pre_temp3[['codice_old', 'descrizione_old', 'codice', 'descrizione']] = cod_pre_toclean[['codice_old', 'descrizione_old', 
             'codice_codice_reg', 'codice_descrizione_reg']][(cod_pre_toclean['desc_in_reg']==False) &
                (cod_pre_toclean['descrizione_cleaned']==False) &
                (cod_pre_toclean['codice_cleaned']==True)].dropna()
cod_pre_temp3 = cod_pre_temp3[cod_pre_cleaned.columns.tolist()]

cod_pre_cleaned_part = pd.concat([cod_pre_temp1,cod_pre_temp2,cod_pre_temp3],sort=False)
cod_pre_cleaned_part.reset_index(inplace=True, drop=True) 

#cod_pre_cleaned_tot = pd.concat([cod_pre_cleaned, cod_pre_temp1,cod_pre_temp2,cod_pre_temp3],sort=False)
#cod_pre_cleaned_tot.reset_index(inplace=True, drop=True) 

cod_pre_cleaned_tot = cod_pre_cleaned.append(cod_pre_cleaned_part, ignore_index = True) 
#%%cod_pre_toclean

len(df_cup['sa_pre_id'][df_cup['sa_pre_id'].isin(cod_pre_db['codice'])].dropna()) - len(df_cup[df_cup['sa_pre_id'].isin(cod_pre_cleaned_tot['codice_old'])])
len(df_cup['sa_pre_id'][~df_cup['sa_pre_id'].isin(cod_pre_db['codice'])].dropna())


len(df_cup['sa_pre_id'].dropna()) - len(df_cup[df_cup['sa_pre_id'].isin(cod_pre_cleaned['codice_old'])])
len(df_cup['sa_pre_id'].dropna()) - len(df_cup[df_cup['sa_pre_id'].isin(cod_pre_cleaned_part['codice_old'])])
len(df_cup['sa_pre_id'].dropna()) - len(df_cup[df_cup['sa_pre_id'].isin(cod_pre_cleaned_tot['codice_old'])])
#%%
# Parifico i nomi di colonna
sex1 = df_pre2[['Codice Catalogo','Descrizione CATALOGO', 'Branca Erogazione Cod', 'Branca Erogazione Descrizione']]
sex1 = sex1.rename(columns={'Codice Catalogo': 'codice', 'Descrizione CATALOGO': 'descrizione',
                            'Branca Erogazione Cod' : 'branca_cod', 'Branca Erogazione Descrizione' : 'branca_descr'})

sex2 = df_pre2[['Codice Naz','Descrizione DM 18/10/2012', 'Branca Erogazione Cod', 'Branca Erogazione Descrizione']]
sex2 = sex2.rename(columns={'Codice Naz': 'codice', 'Descrizione DM 18/10/2012': 'descrizione',
                            'Branca Erogazione Cod' : 'branca_cod', 'Branca Erogazione Descrizione' : 'branca_descr'})

# Creo un DataFrame con tutte le prestazioni
cod_branca = pd.concat([sex1, sex2], ignore_index=True).dropna().drop_duplicates() # ,sex4
cod_pre_cleaned_tot = pd.merge(cod_pre_cleaned_tot, cod_branca[['codice', 'branca_cod', 'branca_descr']],
                               how='outer', on=['codice'])

sfrido1 = df_cup['sa_pre_id'][~df_cup['sa_pre_id'].isin(cod_pre_cleaned_tot.codice_old)].dropna().value_counts()
sfrido1.to_csv('sfrido1.csv')
sfrido2 = cod_pre_toclean[['codice_old', 'descrizione_old', 'occs']][cod_pre_toclean[last_col] == False].sort_values('occs', ascending=False)
sfrido2.to_csv('sfrido2.csv')
#%%

branca_sex = sex1[['branca_cod', 'branca_descr']].apply(lambda x: x.str.strip(), axis=1).drop_duplicates()
branca_sex.branca_cod = branca_sex.branca_cod.str.replace(' - ','-').str.replace(' ','-')

b_cod, b_descr, b_idx  = [], [], []
for idx in branca_sex.index:
    if(len(branca_sex['branca_cod'][idx].split('-')) == 1):
        b_cod.append(branca_sex['branca_cod'][idx])
        b_descr.append(branca_sex['branca_descr'][idx])
        b_idx.append(idx)
branche = pd.DataFrame({
                        'codice' : b_cod,
                        'descrizione' : b_descr}, index = b_idx)
    
branca_sex = branca_sex[~branca_sex.index.isin(branche.index)]
branca_sex.branca_cod = '-' + branca_sex.branca_cod + '-'

for idx in branche.index:
    branca_sex.branca_cod = branca_sex.branca_cod.str.replace('-' + branche.codice[idx] + '-','-')
    branca_sex.branca_descr = branca_sex.branca_descr.str.replace(branche.descrizione[idx],'')
    
branca_sex = branca_sex.apply(lambda x: x.str.strip().str.strip('-').str.strip(), axis=1).drop_duplicates('branca_cod')
branca_sex.branca_cod.replace('', np.nan, inplace=True)
branca_sex.branca_descr.replace('', np.nan, inplace=True)
branca_sex = branca_sex.dropna()
branca_sex.branca_descr = branca_sex.branca_descr.str.replace('[^A-Za-z0-9]', '', regex=True)
branca_sex = branca_sex.rename(columns={'branca_cod': 'codice', 'branca_descr': 'descrizione'})
branche = branche.append(branca_sex)

branche = branche.set_index('codice')
branche['indx'] = branche.index.astype(int)
branche = branche[['indx', 'descrizione']].sort_values('indx')

#%%

cod_pre_cleaned_tot = cod_pre_cleaned_tot.drop('branca_descr', axis = 1)
cod_pre_cleaned_tot.branca_cod = cod_pre_cleaned_tot.branca_cod.str.replace(' ','').str.split('-')

for idx in cod_pre_cleaned_tot.index:
    list_branche = cod_pre_cleaned_tot.branca_cod[idx]
    for i in range(0, len(list_branche)): 
#        print(int(list_branche[i]))
        list_branche[i] = int(list_branche[i].strip()) 


#%%
rel_descr_new = pd.DataFrame({
        'indx' : range(1,len(cod_pre_cleaned_tot.descrizione.drop_duplicates())+1),
        'descrizione' : cod_pre_cleaned_tot.descrizione.drop_duplicates()})
rel_descr_new = rel_descr_new.reset_index(drop=True)
dict_descr = rel_descr_new.set_index('descrizione').to_dict()['indx']

cod_pre_cleaned_tot['descrizione_indx'] = cod_pre_cleaned_tot.descrizione

#Non-Exhaustive Mapping
#If you have a non-exhaustive mapping and wish to retain the existing variables for non-matches, you can add fillna:
cod_pre_cleaned_tot['descrizione_indx'] = cod_pre_cleaned_tot['descrizione'].map(dict_descr).fillna(cod_pre_cleaned_tot['descrizione'])

#%%

rel_cod_branche_prov = cod_pre_cleaned_tot[['descrizione_indx', 'branca_cod']].drop_duplicates('descrizione_indx')

p_cod, b_cod  = [], []
for idx in rel_cod_branche_prov.index:
    for cod in rel_cod_branche_prov.branca_cod[idx]:
        p_cod.append(rel_cod_branche_prov.descrizione_indx[idx])
        b_cod.append(cod)


rel_cod_branche = pd.DataFrame({
        'pre_descr_indx' : p_cod, 
        'branca_cod' : b_cod})

del rel_cod_branche_prov

#%%
nan_index = 99999999
nan_campo = 'NON RICONOSCIUTA'
non_def = pd.DataFrame({
        'codice_old' : df_cup['sa_pre_id'][~df_cup['sa_pre_id'].isin(cod_pre_cleaned_tot.codice_old)].dropna().drop_duplicates(),
        'descrizione_old' : nan_campo,
        'codice' : nan_campo,
        'descrizione' : nan_campo,
        'branca_cod' : nan_index,
        'descrizione_indx' : nan_index})

row = [nan_index, nan_campo]
branche.loc[len(branche)] = row
rel_descr_new.loc[len(rel_descr_new)] = row



#%%
df_cup_c = df_cup.copy()
df_cassa_c = df_cassa.copy()
df_annul_c = df_annul.copy()

df_cup_c['unrepeated_dti'] = False
df_cassa_c['unrepeated_dti'] = False
df_annul_c['unrepeated_dti'] = False

df_cup_c['unrepeated_dti'] = df_cup_c.index.isin(df_cup_c.drop(columns= ['sa_dti_id','sa_contatto_id']).drop_duplicates().index)
df_cassa_c['unrepeated_dti'] = df_cassa_c.index.isin(df_cassa_c.drop(columns= ['sa_mov_id', 'sa_dti_id','sa_dti_pk']).drop_duplicates().index)
df_annul_c['unrepeated_dti'] = df_annul_c.index.isin(df_annul_c.drop(columns= ['sa_dti_id','sa_contatto_id']).drop_duplicates().index)


#%%
#df_cup_c = df_cup_c[~df_cup_c['sa_stato_pren'].isin(['A','R','s'])]
df_cup_c['sa_data_prescr'] = pd.to_datetime(df_cup_c['sa_data_prescr'], errors = 'coerce')
anag_impegnativa = df_cup_c[['sa_impegnativa_id', 'sa_med_id', 'sa_ass_cf', 'sa_comune_id', 'sa_sesso_id', 'sa_eta_id']].drop_duplicates()
anag_impegnativa_anonimo = anag_impegnativa[(anag_impegnativa.sa_impegnativa_id == '-1') |
                                    (anag_impegnativa.sa_ass_cf == '-1')]
anag_impegnativa = anag_impegnativa[(anag_impegnativa.sa_impegnativa_id != '-1') &
                                    (anag_impegnativa.sa_ass_cf != '-1')].drop_duplicates(['sa_impegnativa_id', 'sa_comune_id', 'sa_sesso_id', 'sa_eta_id'], keep=False)

anag_impegnativa = anag_impegnativa.append(anag_impegnativa_anonimo)
anag_impegnativa = anag_impegnativa.reset_index(drop=True)
anag_impegnativa.insert(0, 'indx', range(1, len(anag_impegnativa)+1))

#%%CUP

anag_impegnativa['hash'] = cupload.create_hash(anag_impegnativa, ['sa_impegnativa_id', 'sa_med_id', 'sa_ass_cf', 'sa_comune_id', 'sa_sesso_id', 'sa_eta_id'])
df_cup_c['hash'] = cupload.create_hash(df_cup_c, ['sa_impegnativa_id', 'sa_med_id', 'sa_ass_cf', 'sa_comune_id', 'sa_sesso_id', 'sa_eta_id'])
#df_cup_c['hash2'] = cupload.create_hash(df_cup_c, ['sa_impegnativa_id', 'sa_data_prescr', 'sa_med_id', 'sa_ass_cf', 'sa_comune_id', 'sa_sesso_id', 'sa_eta_id'])

dict_impegnativa = anag_impegnativa.set_index('hash').to_dict()['indx']

df_cup_c['hash'] = df_cup_c['hash'].map(dict_impegnativa).fillna(nan_index)
row = [nan_index, nan_campo, nan_campo, nan_campo, nan_campo, np.nan, np.nan, nan_campo]
anag_impegnativa.loc[len(anag_impegnativa)] = row
#df_cup_c = df_cup_c.drop(columns=['sa_impegnativa_id', 'sa_data_prescr', 'sa_med_id', 'sa_ass_cf', 'sa_comune_id', 'sa_sesso_id', 'sa_eta_id'])
df_cup_c = df_cup_c.rename(columns={'hash':'indx_impegnativa'})
df_cup_c['indx_impegnativa'] = df_cup_c['indx_impegnativa'].astype(int)

#CASSA
#, 'sa_data_prescr'
df_cassa_c['hash'] = cupload.create_hash(df_cassa_c, ['sa_impegnativa_id', 'sa_med_id', 'sa_ass_cf', 'sa_comune_id', 'sa_sesso_id', 'sa_eta_id'])

df_cassa_c['hash'] = df_cassa_c['hash'].map(dict_impegnativa).fillna(nan_index)
df_cassa_c = df_cassa_c.rename(columns={'hash':'indx_impegnativa'})
df_cassa_c['indx_impegnativa'] = df_cassa_c['indx_impegnativa'].astype(int)

#ANNUL
#, 'sa_data_prescr'
df_annul_c['hash'] = cupload.create_hash(df_annul_c, ['sa_impegnativa_id', 'sa_med_id', 'sa_ass_cf', 'sa_comune_id', 'sa_sesso_id', 'sa_eta_id'])

df_annul_c['hash'] = df_annul_c['hash'].map(dict_impegnativa).fillna(nan_index)
df_annul_c = df_annul_c.rename(columns={'hash':'indx_impegnativa'})
df_annul_c['indx_impegnativa'] = df_annul_c['indx_impegnativa'].astype(int)


#%%CUP

dict_prestazioni = cod_pre_cleaned_tot.set_index('codice_old').to_dict()['descrizione_indx']

df_cup_c['indx_prestazione']  = df_cup_c['sa_pre_id'] 
df_cup_c['indx_prestazione'] = df_cup_c['indx_prestazione'].map(dict_prestazioni).fillna(nan_index)
df_cup_c['indx_prestazione'] = df_cup_c['indx_prestazione'].astype(int)

#unique_persons = df_cup_c[(df_cup_c.indx_prestazione == nan_index) |
#                                    (df_cup_c.indx_impegnativa == nan_index)].drop_duplicates('sa_ass_cf')
#intersection = pd.merge(df_cup_c[(df_cup_c.indx_prestazione != nan_index) &
#                                    (df_cup_c.indx_impegnativa != nan_index)].drop_duplicates('sa_ass_cf'),unique_persons, how='inner', on='sa_ass_cf')

#CASSA
df_cassa_c['indx_prestazione']  = df_cassa_c['sa_pre_id'] 
df_cassa_c['indx_prestazione'] = df_cassa_c['indx_prestazione'].map(dict_prestazioni).fillna(nan_index)
df_cassa_c['indx_prestazione'] = df_cassa_c['indx_prestazione'].astype(int)

#ANNUL
df_annul_c['indx_prestazione']  = df_annul_c['sa_pre_id'] 
df_annul_c['indx_prestazione'] = df_annul_c['indx_prestazione'].map(dict_prestazioni).fillna(nan_index)
df_annul_c['indx_prestazione'] = df_annul_c['indx_prestazione'].astype(int)

#%%

#df_cup_c['hash'] = cupload.create_hash(df_cup_c, ['sa_pre_id', 'sa_impegnativa_id', 'sa_med_id', 'sa_ass_cf'])
#
#df_cup_N = df_cup_c[~(df_cup_c['sa_stato_pren'].isin(['A','R','s'])) &
#                    (df_cup_c['indx_prestazione'] != nan_index)]
#df_cup_R = df_cup_c[(df_cup_c['sa_stato_pren'].isin(['R'])) &
#                    (df_cup_c['indx_prestazione'] != nan_index)]
#
#b = df_cup_R[~df_cup_R.hash.isin(df_cup_N.hash)]
#
#
#df_cassa_c['hash'] = cupload.create_hash(df_cassa_c, ['sa_pre_id', 'sa_impegnativa_id', 'sa_med_id', 'sa_ass_cf'])
#
#d = df_cup_c[(df_cup_c.indx_impegnativa == 7155200) & (df_cup_c.unrepeated_dti)]
#e = df_cassa_c[(df_cassa_c.sa_impegnativa_id == '11491516') & (df_cassa_c.unrepeated_dti)]
#intersection = pd.merge(d, df_cassa_c, how='inner', on='hash')


#%%

unita_eroganti = cupload.load_dataset(file=data_path+'dec_unita_eroganti.csv')

uop_to_clean = pd.DataFrame({'codice' : df_cup_c['sa_uop_codice_id'].str.strip().drop_duplicates(),
                             'desc_in_reg' : False})
uop_to_clean['desc_in_reg'] = uop_to_clean.codice.isin(unita_eroganti['sa_uop_codice_id'].str.strip())
col_const = 'desc_in_reg' #'descrizione_cleaned' 'desc_in_reg'
dist = cupload.measure_distances(unita_eroganti['sa_uop_codice_id'].str.strip().to_list(), uop_to_clean[~uop_to_clean[col_const]]['codice'].to_list())

uop_to_clean = cupload.fill_df_cleaned(dist, uop_to_clean, unita_eroganti, 'codice', dist_to_check=0.21, col_constrain = col_const, col_cod_good = 'sa_uop_codice_id', descr_boole=False)

dict_uop = uop_to_clean[uop_to_clean.codice_cleaned].set_index('codice').to_dict()['codice_codice_reg']

df_cup_c['sa_uop_codice_id'] = df_cup_c['sa_uop_codice_id'].map(dict_uop).fillna(df_cup_c['sa_uop_codice_id'])

unita_eroganti = unita_eroganti.reset_index(drop=True)
unita_eroganti.insert(0, 'indx', range(1, len(unita_eroganti)+1))
row = [nan_index, nan_campo, nan_campo, nan_campo, nan_campo, nan_campo, nan_campo, 
       nan_campo, nan_campo, nan_campo, nan_campo, nan_campo, nan_index, nan_index, 
       nan_index, nan_campo, nan_campo, nan_campo, nan_campo]
unita_eroganti.loc[len(unita_eroganti)] = row

dict_uop_2 = unita_eroganti.set_index('sa_uop_codice_id').to_dict()['indx']

#CUP
df_cup_c['indx_uop'] = df_cup_c['sa_uop_codice_id']
df_cup_c['indx_uop'] = df_cup_c['indx_uop'].map(dict_uop_2).fillna(nan_index)
df_cup_c['indx_uop'] = df_cup_c['indx_uop'].astype(int)

#CASSA
df_cassa_c['indx_uop'] = df_cassa_c['sa_uop_codice_id']
df_cassa_c['indx_uop'] = df_cassa_c['indx_uop'].map(dict_uop_2).fillna(nan_index)
df_cassa_c['indx_uop'] = df_cassa_c['indx_uop'].astype(int)

#ANNUL
df_annul_c['indx_uop'] = df_annul_c['sa_uop_codice_id']
df_annul_c['indx_uop'] = df_annul_c['indx_uop'].map(dict_uop_2).fillna(nan_index)
df_annul_c['indx_uop'] = df_annul_c['indx_uop'].astype(int)

#%%CUP
df_cup_c['flag_stato'] = ~df_cup_c['sa_stato_pren'].isin(['A','R','s'])
#df_cup_c = df_cup_c.drop(columns = 'hash')
#
#df_cassa_c = df_cassa_c.drop(columns = 'hash')
#
#df_annul_c = df_annul_c.drop(columns = 'hash')

comuni = cupload.load_dataset(file=data_path+'comuni.csv')

comuni['Codice Comune formato alfanumerico'][comuni['Ripartizione geografica']=='Estero'] = '999'+comuni['Codice Comune formato alfanumerico'][comuni['Ripartizione geografica']=='Estero'].astype(str)
comuni['Codice Comune formato alfanumerico'] = comuni['Codice Comune formato alfanumerico'].astype(int)
comuni = comuni.drop(columns = 'indx')
comuni = comuni.reset_index(drop=True)
comuni.insert(0, 'indx', range(1, len(comuni)+1))
row = [nan_index, nan_index, nan_index, nan_campo, nan_campo, nan_campo, nan_campo, 
       nan_campo, nan_index, nan_index]
comuni.loc[len(comuni)] = row

anag_impegnativa['sa_comune_id'][anag_impegnativa['sa_comune_id'].str.isnumeric()==False] = nan_index
anag_impegnativa['sa_comune_id'][anag_impegnativa['sa_comune_id'].isnull()] = nan_index
anag_impegnativa['sa_comune_id'] = anag_impegnativa['sa_comune_id'].astype(int)

anag_impegnativa['sa_comune_id'][~anag_impegnativa['sa_comune_id'].isin(comuni['Codice Comune formato alfanumerico'])] = nan_index

anag_impegnativa = anag_impegnativa.drop(columns = 'hash')

#CUP
df_cup_c['sa_comune_id'][df_cup_c['sa_comune_id'].str.isnumeric()==False] = nan_index
df_cup_c['sa_comune_id'][df_cup_c['sa_comune_id'].isnull()] = nan_index
df_cup_c['sa_comune_id'] = df_cup_c['sa_comune_id'].astype(int)

df_cup_c['sa_comune_id'][~df_cup_c['sa_comune_id'].isin(comuni['Codice Comune formato alfanumerico'])] = nan_index
#CASSA
df_cassa_c['sa_comune_id'][df_cassa_c['sa_comune_id'].str.isnumeric()==False] = nan_index
df_cassa_c['sa_comune_id'][df_cassa_c['sa_comune_id'].isnull()] = nan_index
df_cassa_c['sa_comune_id'] = df_cassa_c['sa_comune_id'].astype(int)

df_cassa_c['sa_comune_id'][~df_cassa_c['sa_comune_id'].isin(comuni['Codice Comune formato alfanumerico'])] = nan_index
#ANNUL
df_annul_c['sa_comune_id'][df_annul_c['sa_comune_id'].str.isnumeric()==False] = nan_index
df_annul_c['sa_comune_id'][df_annul_c['sa_comune_id'].isnull()] = nan_index
df_annul_c['sa_comune_id'] = df_annul_c['sa_comune_id'].astype(int)

df_annul_c['sa_comune_id'][~df_annul_c['sa_comune_id'].isin(comuni['Codice Comune formato alfanumerico'])] = nan_index

#%% DROP USELESS COLS
df_cassa_c = df_cassa_c.drop(columns=['sa_deleted', 'sa_data_del'])
df_annul_c = df_annul_c.drop(columns=['sa_deleted', 'sa_data_del'])
#%% EXPORT SESSION

#branche.to_csv(csv_path+'branche_new.csv', index=False)
#rel_descr_new.to_csv(csv_path+'rel_descr_new.csv', index=False)
#rel_cod_branche.to_csv(csv_path+'rel_cod_branche.csv', index=False)
#cod_pre_cleaned_tot[['codice_old','descrizione_old','codice','descrizione_indx']].to_csv(csv_path+'prestazioni_old_to_new.csv', index=False)
#anag_impegnativa.to_csv(csv_path+'anag_impegnativa.csv', index=False)
#unita_eroganti.to_csv(csv_path+'unita_eroganti.csv', index = False)
#comuni.to_csv(csv_path+'comuni.csv', index = False)
#
#df_cup_c = df_cup_c.reset_index(drop=True)
#df_cup_c.insert(0, 'indx', range(1, len(df_cup_c)+1))
#df_cup_c.to_csv(csv_path+'df_cup_c.csv', index = False)
#
#df_cassa_c = df_cassa_c.reset_index(drop=True)
#df_cassa_c.insert(0, 'indx', range(1, len(df_cassa_c)+1))
#df_cassa_c.to_csv(csv_path+'df_cassa_c.csv', index = False)
#
#df_annul_c = df_annul_c.reset_index(drop=True)
#df_annul_c.insert(0, 'indx', range(1, len(df_annul_c)+1))
#df_annul_c.to_csv(csv_path+'df_annul_c.csv', index = False)

#%%TEST

val = cupload.extract_entity(df_cup_c[df_cup_c['unrepeated_dti'] & df_cup_c['flag_stato']], 
                             columns=['sa_data_ins', 'sa_ass_cf', 'sa_data_pren', 'sa_utente_id',
                               'sa_contratto_id', 'sa_data_app', 'sa_mese_app_id', 'sa_uop_codice_id',
                               'sa_comune_id', 'sa_branca_id', 'sa_pre_id', 'sa_med_id',
                               'sa_ese_id_lk', 'sa_sesso_id', 'sa_is_ad', 'sa_spr_id', 'sa_ut_id',
                               'sa_operazione', 'sa_stato_pren', 'sa_eta_id', 'sa_impegnativa_id',
                               'sa_gg_attesa', 'sa_gg_attesa_pdisp',
                               'sa_num_prestazioni', 'sa_classe_priorita', 'sa_is_pre_eseguita',
                               'sa_data_prescr', 'sa_primo_accesso', 'sa_asl',
                               'indx_impegnativa', 'indx_prestazione', 'indx_uop'], 
                            perc=0.01, tot=1000, with_weight=False, all_comb=False)

#['sa_data_ins', 'sa_ass_cf', 'sa_data_pren', 'sa_utente_id', 'sa_contratto_id', 'sa_data_app', 'sa_mese_app_id', 'sa_uop_codice_id', 'sa_comune_id', 'sa_sesso_id', 'sa_is_ad', 'sa_spr_id', 'sa_ut_id', 'sa_operazione', 'sa_stato_pren', 'sa_eta_id', 'sa_contatto_id', 'sa_gg_attesa', 'sa_primo_accesso', 'sa_asl', 'indx_uop']

val.to_csv(csv_path+'entities_group.csv', index=False)
