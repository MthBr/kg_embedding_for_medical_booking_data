#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:32:50 2019

@author: edoardoprezioso
"""

import numpy as np
import pandas as pd

#%% IMPORT CONFIG FILE
import configparser
configparser = configparser.RawConfigParser()   
configFilePath = "config.ini"
configparser.read(configFilePath)

# LOAD DEI DATASET
csv_path = configparser.get('CUP-config', 'csv_path')

#%%
df_cup_c = pd.read_csv(csv_path + 'df_cup_c.csv')

#%%
df_branche = pd.read_csv(csv_path + 'df_branche.csv')
df_branche_new = pd.read_csv(csv_path + 'df_branche_new.csv')

#%%

branche_in_cup_c = df_cup_c.sa_branca_id.drop_duplicates()
branche_id_values = [('AL', 26), #Altro
                     ('026', 26),
                     ('18', 26),
                     ('ANE', 1), #Anestesia
                     ('001', 1),
                     ('01', 1),
                     ('CAR', 2), #Cardiologia
                     ('002', 2),
                     ('02', 2),
                     ('04', 3), #Chirurgia generale
                     ('003', 3),
                     ('004', 4), #Chirurgia plastica
                     ('05', 4),
                     ('ANG', 5), #Chirurgia vascolare - Angiologia,
                     ('005', 5),
                     ('06', 5),
                     ('DER', 6), #Dermosifilopatia
                     ('006', 6),
                     ('10', 6),
                     ('11', 28), #Diabetologia
                     ('028', 28),
                     ('DIA', 28),
                     ('07', 7), #Diagnostica per immagini - Medicina nucleare
                     ('007', 7),
                     ('92', 7),
                     ('RAD', 8), #Diagnostica per immagini - Radiologia diagnostica
                     ('08', 8),
                     ('008', 8),
                     ('14', 9), #Endocrinologia
                     ('009', 9),
                     ('09', 9),
                     ('010', 10), #Gastroenterologia - Chirurgia ed endoscopia digestiva
                     ('GAS', 10),
                     ('011', 11), #Lab. analisi chimico cliniche e microbiologiche - Microbiologia - Virologia - Anatomia e istologia patologica - Genetica- Immunoematologia e s. trasf.
                     ('LAB', 11),
                     ('ANP', 11),
                     ('IMM', 11),
                     ('GEN', 11),
                     ('PR7', 12), #Medicina fisica e riabilitazione - Recupero e riabilitazione funzionale dei motulesi e neurolesi
                     ('PR6', 12),
                     ('PR2', 12),
                     ('PR4', 12),
                     ('PR5', 12),
                     ('41', 12),
                     ('PR9', 12),
                     ('PR8', 12),
                     ('PR1', 12),
                     ('PR3', 12),
                     ('12', 12),
                     ('012', 12),
                     ('FKT', 12),
                     ('NEF', 13), #Nefrologia
                     ('013', 13),
                     ('28', 14), #Neurochirurgia
                     ('014', 14),
                     ('015', 15), #Neurologia
                     ('15', 15),
                     ('NEU', 15),
                     ('98', 15),
                     ('NRA', 15),
                     ('016', 16), #Oculistica
                     ('16', 16),
                     ('OCU', 16),
                     ('31', 17), #Odontostomatologia - Chirurgia maxillo facciale
                     ('017', 17),
                     ('17', 17),
                     ('32', 18), #Oncologia
                     ('018', 18),
                     ('ORT', 19), #Ortopedia e traumatologia
                     ('19', 19),
                     ('019', 19),
                     ('34', 20), #Ostetricia e ginecologia
                     ('020', 20),
                     ('20', 20),
                     ('021', 21), #Otorinolaringoiatria
                     ('21', 21),
                     ('OTO', 21),
                     ('022', 22), #Pneumologia
                     ('22', 22),
                     ('PNE', 22),
                     ('ALL', 22),
                     ('37', 23), #Psichiatria
                     ('023', 23),
                     ('23', 23),
                     ('24', 24), #Radioterapia
                     ('024', 24),
                     ('38', 24),
                     ('URO', 25), #Urologia
                     ('025', 25)]

df_branche_id = pd.DataFrame(branche_id_values, columns = ['codice','indx'])

good_branche_not_in_cup = df_branche_id[~df_branche_id.codice.isin(branche_in_cup_c)]
branche_cup_not_in_good_branche = branche_in_cup_c[~branche_in_cup_c.isin(df_branche_id.codice)]
branche_cup_in_df_branche = branche_in_cup_c[branche_in_cup_c.isin(df_branche.id_branca)]
branche_cup_not_in_df_branche = branche_in_cup_c[~branche_in_cup_c.isin(df_branche.id_branca)]
df_branche_not_in_df_cup = df_branche[~df_branche.id_branca.isin(branche_in_cup_c)]

values_branche_in_cup = df_cup_c.sa_branca_id.value_counts()
values_branche_cup_not_in_good_branche = values_branche_in_cup[~values_branche_in_cup.index.isin(df_branche_id.codice)]

df_branche_not_in_list = df_branche[~df_branche.id_branca.isin(df_branche_id.codice)]

df_branche_not_in_list_codice = df_branche_not_in_list[['id_branca']].copy()
df_branche_not_in_list_codice = df_branche_not_in_list_codice.rename(columns={'id_branca':'codice'})
df_branche_not_in_list_codice['indx'] = 99999999 # NON RICONOSCIUTA

df_branche_codice_id = pd.concat([df_branche_id,df_branche_not_in_list_codice], ignore_index=True)
df_branche_codice_id = df_branche_codice_id.sort_values(['indx','codice'])

df_branche_codice_id.to_csv(csv_path + 'df_branche_indx.csv')

#%%
dict_branche_id = df_branche_codice_id.set_index('codice').to_dict()['indx']

df_cup_c_indx_branca = df_cup_c.copy()
df_cup_c_indx_branca['indx_branca'] = df_cup_c_indx_branca['sa_branca_id'].replace(dict_branche_id)

print(len(df_cup_c_indx_branca[df_cup_c_indx_branca['indx_branca'].isna()]))

df_cup_c_indx_branca.to_csv(csv_path + 'df_cup_c_indx_branca.csv', index=False)
