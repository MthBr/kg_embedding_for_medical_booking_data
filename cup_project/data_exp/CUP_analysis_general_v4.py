#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 12:07:10 2019
Generates plots.
DOES NOT WORK! review input files!

@author: modal
"""

import pandas as pd
import numpy as np
import pickle

# Disable warnings in Anaconda
import warnings
warnings.filterwarnings('ignore')

# Matplotlib forms basis for visualization in Python
import matplotlib.pyplot as plt

# We will use the Seaborn library
import seaborn as sns
sns.set()

#%% LOAD DEI DATASET PULITI
file_path = '../../DATASET/'

pickle_off = open(file_path+"df_cup.pickle","rb")
df_cup = pickle.load(pickle_off)

#%% CREAZIONE DEL DATAFRAME DI INTERESSE
df_cup_extraction = pd.DataFrame({
        'anno_ins': df_cup.sa_data_ins.dt.year,
        'assistito': df_cup.sa_ass_cf,
        'operatore': df_cup.sa_ut_id,
        'sesso_assistito': df_cup.sa_sesso_id,
        'eta_assistito': df_cup.sa_eta_id,
        'comune': df_cup.sa_comune_id,
        'branca': df_cup.sa_branca_id,
        'prestazione': df_cup.sa_pre_id,
        'num_prestazioni': df_cup.sa_num_prestazioni,
        'priorita': df_cup.sa_classe_priorita,
        'data_app' : df_cup.sa_data_app,
        'mese_app': df_cup.sa_data_app.dt.month,
        'anno_app': df_cup.sa_data_app.dt.year,
        'attesa_app': df_cup.sa_gg_attesa,
        'attesa_disp': df_cup.sa_gg_attesa_pdisp,
        'asl': df_cup.sa_asl
        })

#%%GRAFICA GENERALE
file_path = '../../'
# NUMERO DI PRENOTAZIONI PER BRANCA E PER ASL
stats_df = (df_cup_extraction.groupby('asl')['branca'].value_counts().unstack().fillna(0).T)/(df_cup_extraction['branca'].value_counts().sum())
stats_df = stats_df.reset_index()
stats_df = stats_df.rename(columns={'index':'branca'})
stats_df['pdf'] = stats_df.sum(axis=1)
stats_df = stats_df.sort_values(by='pdf', ascending=False)
stats_df['cdf'] = stats_df['pdf'].cumsum()

asl_list = list(df_cup_extraction['asl'].value_counts().index)

fig, ax = plt.subplots()
fig = stats_df.plot.bar(x='branca', y=asl_list, stacked=True, rot=90, legend=True, figsize=(24, 6), width=0.7, ax = ax)
fig.set_xlabel("Branca")
fig.set_ylabel("Frequency on Dataset")
ax2 = ax.twinx()
fig = stats_df.plot(x='branca', y=['cdf'], color = 'r', linewidth = 3, legend=False, ax = ax2, alpha = 0.5, grid=False)
ax.legend(bbox_to_anchor=(1.08, 0.90))
ax.set_xlim(-0.5,len(stats_df)-1)
plt.legend(bbox_to_anchor=(1.08, 1))
plt.savefig(file_path+'IMG/PRENOTAZIONI(branca,asl).png', format='png', dpi=300, bbox_inches = "tight")
plt.show()

fig, ax = plt.subplots()
fig = stats_df.plot.bar(x='branca', y='pdf', stacked=True, rot=90, legend=True, figsize=(24, 6), width=0.7, ax = ax)
fig.set_xlabel("Branca")
fig.set_ylabel("Frequency on Dataset")
ax2 = ax.twinx()
fig = stats_df.plot(x='branca', y=['cdf'], color = 'r', linewidth = 3, legend=False, ax = ax2, alpha = 0.5, grid=False)
ax.legend(bbox_to_anchor=(1.08, 0.90))
ax.set_xlim(-0.5,len(stats_df)-1)
plt.legend(bbox_to_anchor=(1.08, 1))
plt.savefig(file_path+'IMG/PRENOTAZIONI(branca).png', format='png', dpi=300, bbox_inches = "tight")
plt.show()

# Salvo la lista dele branche ignobili
list_branche_sign = list(stats_df['branca'].loc[stats_df['cdf']<=0.9])
del stats_df

# NUMERO DI PRENOTAZIONI PER COMUNE
stats_df = (df_cup_extraction['comune'].value_counts()/len(df_cup_extraction)).to_frame()
stats_df = stats_df.reset_index()
stats_df = stats_df.rename(columns={'comune':'prenotazioni'})
stats_df = stats_df.rename(columns={'index':'comune'})

fig, ax = plt.subplots()
fig = stats_df.plot.bar(x='comune', y='prenotazioni', rot=90, legend=False, figsize=(24, 6), width=0.7, ax = ax)
fig.set_xlabel("Comune")
fig.set_ylabel("Frequency on Dataset")
plt.savefig(file_path+'IMG/PRENOTAZIONI(comune).png', format='png', dpi=300, bbox_inches = "tight")
plt.show()
del stats_df

# NUMERO DI PRENOTAZIONI PER COMUNE (SOLO CAMPANIA ESPLICITI)
comuni_list = ['Napoli','Salerno','Avellino','Benevento','Caserta','Altro']

stats_df = (df_cup_extraction['comune'].value_counts()/len(df_cup_extraction)).to_frame()
stats_df = stats_df.reset_index()
stats_df = stats_df.rename(columns={'comune':'prenotazioni'})
stats_df = stats_df.rename(columns={'index':'comune'})
stats_df.loc[-1] = ['Altro', stats_df[~stats_df['comune'].isin(comuni_list)]['prenotazioni'].sum()]
stats_df = stats_df[stats_df['comune'].isin(comuni_list)]

fig, ax = plt.subplots()
fig = stats_df.plot.bar(x='comune', y='prenotazioni', rot=90, legend=False, figsize=(24, 6), width=0.7, ax = ax)
fig.set_xlabel("Comune")
fig.set_ylabel("Frequency on Dataset")
plt.savefig(file_path+'IMG/PRENOTAZIONI(comune_campania).png', format='png', dpi=300, bbox_inches = "tight")
plt.show()
del stats_df

# ANDAMENTO BRANCHE NEGLI ANNI
import re
data = df_cup_extraction.groupby('anno_ins')['branca'].value_counts().unstack().fillna(0)
fig, ax = plt.subplots()
fig =data.plot(rot=90, legend=True, figsize=(12, 6), linewidth = 3, ax = ax)
#    fig.set_xlabel("Comune")
fig.set_ylabel("#Prestazioni")   
ax.legend(bbox_to_anchor=(1, 1.01),ncol=2)
#fig.set_title(branca)
branca_name = re.sub(r"[^A-Za-z]+", '', branca)

colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired   
colors = [colormap(i) for i in np.linspace(0, 1,len(ax.lines))]
for i,j in enumerate(ax.lines):
    j.set_color(colors[i])


ax.legend(loc=2,bbox_to_anchor=(1, 1.01),ncol=2)
plt.savefig(file_path+'IMG/ANDAMENTO_BRANCHE/ANDAMENTOgenerale.png', format='png', dpi=300, bbox_inches = "tight")
plt.show()
  




for branca in list(data.columns):
    fig, ax = plt.subplots()
    fig =data.plot(
        y = branca,rot=90, legend=False, figsize=(12, 6), linewidth = 3, ax = ax)
#    fig.set_xlabel("Comune")
    fig.set_ylabel("#Prestazioni")     
    fig.set_title(branca)
    branca_name = re.sub(r"[^A-Za-z]+", '', branca)
    plt.savefig(file_path+'IMG/ANDAMENTO_BRANCHE/'+branca_name+'.png', format='png', dpi=300, bbox_inches = "tight")
    plt.show()
del data

# CODICE DI PRIORITA' E ATTESA
df_cup_extraction['priorita'].loc[np.isnan(df_cup_extraction['priorita'])]= 'NAN'
data = pd.DataFrame({
        'attesa': df_cup_extraction.groupby('priorita')['attesa_app'].mean(),
        'disponibilita': df_cup_extraction.groupby('priorita')['attesa_disp'].mean()})

fig, ax = plt.subplots()
fig =data.plot.bar(y = ['attesa','disponibilita'],rot=90, legend=True, figsize=(12, 6), ax = ax)
fig.set_xlabel("Codice Priorità")
fig.set_ylabel("Average Value (days)")  
fig.legend(["Attesa", "Dsiponibilità"])
branca_name = re.sub(r"[^A-Za-z]+", '', branca)
plt.savefig(file_path+'IMG/ATTESA-DISPONIBILITA(priorita).png', format='png', dpi=300, bbox_inches = "tight")
plt.show()

data = pd.DataFrame({
        'attesa': df_cup_extraction.groupby('priorita')['attesa_app'].median(),
        'disponibilita': df_cup_extraction.groupby('priorita')['attesa_disp'].median()})

fig, ax = plt.subplots()
fig =data.plot.bar(y = ['attesa','disponibilita'],rot=90, legend=True, figsize=(12, 6), ax = ax)
fig.set_xlabel("Codice Priorità")
fig.set_ylabel("Median Value (days)")  
fig.legend(["Attesa", "Dsiponibilità"])
branca_name = re.sub(r"[^A-Za-z]+", '', branca)
plt.savefig(file_path+'IMG/ATTESA-DISPONIBILITA(priorita) - median.png', format='png', dpi=300, bbox_inches = "tight")
plt.show()

for branca in list(df_cup_extraction.branca.unique()):
    data = pd.DataFrame({
            'attesa': df_cup_extraction.loc[df_cup_extraction.branca==branca].groupby('priorita')['attesa_app'].mean(),
            'disponibilita': df_cup_extraction.loc[df_cup_extraction.branca==branca].groupby('priorita')['attesa_disp'].mean()})
    
    fig, ax = plt.subplots()
    fig =data.plot.bar(y = ['attesa','disponibilita'],rot=90, legend=True, figsize=(12, 6), ax = ax)
    fig.set_xlabel("Codice Priorità")
    fig.set_ylabel("Average Value (days)")  
    fig.legend(["Attesa", "Dsiponibilità"])
    branca_name = re.sub(r"[^A-Za-z]+", '', branca)
    plt.savefig(file_path+'IMG/PRIORITA/AD(priorita) - mean - '+branca_name+'.png', format='png', dpi=300, bbox_inches = "tight")
    plt.show()
    
for branca in list(df_cup_extraction.branca.unique()):
    data = pd.DataFrame({
            'attesa': df_cup_extraction.loc[df_cup_extraction.branca==branca].groupby('priorita')['attesa_app'].median(),
            'disponibilita': df_cup_extraction.loc[df_cup_extraction.branca==branca].groupby('priorita')['attesa_disp'].median()})
    
    fig, ax = plt.subplots()
    fig =data.plot.bar(y = ['attesa','disponibilita'],rot=90, legend=True, figsize=(12, 6), ax = ax)
    fig.set_xlabel("Codice Priorità")
    fig.set_ylabel("Median Value (days)")  
    fig.legend(["Attesa", "Dsiponibilità"])
    branca_name = re.sub(r"[^A-Za-z]+", '', branca)
    plt.savefig(file_path+'IMG/PRIORITA/AD(priorita) - median - '+branca_name+'.png', format='png', dpi=300, bbox_inches = "tight")
    plt.show()
#%% SFELLO IL DATASET
# Sostituisco i comuni ignobili
df_cup_extraction.loc[~df_cup_extraction['comune'].isin(comuni_list),'comune'] = 'Altro'
# Sostituisco le branche ignobili
df_cup_extraction.loc[~df_cup_extraction['branca'].isin(list_branche_sign),'branca'] = 'ALTRO'

#%% GRAFICA PRENOTAZIONI
# NUMERO DI PRENOTAZIONI PER COMUNE E PER ASL (dopo il taglio)
stats_df = (df_cup_extraction.groupby('asl')['comune'].value_counts().unstack().fillna(0).T)/(df_cup_extraction['comune'].value_counts().sum())
stats_df = stats_df.reset_index()
stats_df = stats_df.rename(columns={'index':'comune'})
stats_df['pdf'] = stats_df.sum(axis=1)
stats_df = stats_df.sort_values(by='pdf', ascending=False)
stats_df['cdf'] = stats_df['pdf'].cumsum()

asl_list = list(df_cup_extraction['asl'].value_counts().index)

fig, ax = plt.subplots()
fig = stats_df.plot.bar(x='comune', y=asl_list, stacked=True, rot=90, legend=True, figsize=(24, 6), width=0.7, ax = ax)
fig.set_xlabel("Comune")
fig.set_ylabel("Frequency on Dataset")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.savefig(file_path+'IMG/PRENOTAZIONI(comune,asl).png', format='png', dpi=300, bbox_inches = "tight")
plt.show()
del stats_df

# NUMERO DI PRENOTAZIONI PER BRANCA E PER ASL (dopo il taglio)
stats_df = (df_cup_extraction.groupby('asl')['branca'].value_counts().unstack().fillna(0).T)/(df_cup_extraction['branca'].value_counts().sum())
stats_df = stats_df.reset_index()
stats_df = stats_df.rename(columns={'index':'branca'})
stats_df['pdf'] = stats_df.sum(axis=1)
stats_df = stats_df.sort_values(by='pdf', ascending=False)
stats_df['cdf'] = stats_df['pdf'].cumsum()

asl_list = list(df_cup_extraction['asl'].value_counts().index)

fig, ax = plt.subplots()
fig = stats_df.plot.bar(x='branca', y=asl_list, stacked=True, rot=90, legend=True, figsize=(24, 6), width=0.7, ax = ax)
fig.set_xlabel("Branca")
fig.set_ylabel("Frequency on Dataset")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.savefig(file_path+'IMG/PRENOTAZIONI_2(branca,asl).png', format='png', dpi=300, bbox_inches = "tight")
plt.show()

fig, ax = plt.subplots()
fig = stats_df.plot.bar(x='branca', y='pdf', stacked=True, rot=90, legend=True, figsize=(24, 6), width=0.7, ax = ax)
fig.set_xlabel("Branca")
fig.set_ylabel("Frequency on Dataset")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.savefig(file_path+'IMG/PRENOTAZIONI_2(branca).png', format='png', dpi=300, bbox_inches = "tight")
plt.show()

# Salvo la lista dele branche ignobili
list_branche_sign = list(stats_df['branca'].loc[stats_df['cdf']<=0.9])
del stats_df

# NUMERO DI PRENOTAZIONI PER BRANCA E PER COMUNE
stats_df = (df_cup_extraction.groupby('comune')['branca'].value_counts().unstack().fillna(0).T)/(df_cup_extraction['branca'].value_counts().sum())
stats_df = stats_df.reset_index()
stats_df = stats_df.rename(columns={'index':'branca'})
stats_df['pdf'] = stats_df.sum(axis=1)
stats_df = stats_df.sort_values(by='pdf', ascending=False)
stats_df['cdf'] = stats_df['pdf'].cumsum()

fig, ax = plt.subplots()
fig = stats_df.plot.bar(x='branca', y=comuni_list, stacked=True, rot=90, legend=True, figsize=(24, 8), width=0.8, ax = ax)
fig.set_xlabel("Branca")
fig.set_ylabel("Frequency on Dataset")
#ax2 = ax.twinx()
#fig = stats_dfF.plot.bar(x='branca', y=fascia_eta_list, stacked=True, rot=90, width=0.4, ax = ax2, position=0, grid=False)
##ax.legend(bbox_to_anchor=(0.99, 0.95))
#ax.set_xlim(-0.5,len(stats_df2)-1)
plt.legend(bbox_to_anchor=(1.09, 1))
plt.savefig(file_path+'IMG/PRENOTAZIONI(branca,comune).png', format='png', dpi=300, bbox_inches = "tight")
plt.show()
del stats_df

# NUMERO DI PRENOTAZIONI PER BRANCA, PER COMUNE E PER SESSO
stats_df_M = (df_cup_extraction[df_cup_extraction['sesso_assistito']==1].groupby('comune')['branca'].value_counts().unstack().fillna(0).T)/(df_cup_extraction['branca'].value_counts().sum())
stats_df_M['pdf'] = stats_df_M.sum(axis=1)
stats_df_M['cdf'] = stats_df_M['pdf'].cumsum()
stats_df_M = stats_df_M.add_suffix('_M')

stats_df_F = (df_cup_extraction[df_cup_extraction['sesso_assistito']==2].groupby('comune')['branca'].value_counts().unstack().fillna(0).T)/(df_cup_extraction['branca'].value_counts().sum())
stats_df_F['pdf'] = stats_df_F.sum(axis=1)
stats_df_F['cdf'] = stats_df_F['pdf'].cumsum()
stats_df_F = stats_df_F.add_suffix('_F')

stats_df = pd.concat([stats_df_M, stats_df_F], axis=1, sort=False)
stats_df = stats_df.reset_index()
stats_df = stats_df.rename(columns={'index':'branca'})
stats_df = stats_df.sort_values(by='pdf_M', ascending=False)

comune_M = [item+'_M' for item in comuni_list]
comune_F = [item+'_F' for item in comuni_list]

fig, ax = plt.subplots()
fig = stats_df.plot.bar(x='branca', y=comune_M, stacked=True, rot=90, legend=True, figsize=(32, 8), width=0.4, ax = ax, position=1)
fig.set_xlabel("Branca")
fig.set_ylabel("Frequency on Dataset - M/F")
fig = stats_df.plot.bar(x='branca', y=comune_F, stacked=True, rot=90, width=0.4, ax = ax, position=0, grid=True, legend = False)
ax.set_xlim(-0.5,len(stats_df_M)+0.5)
plt.legend(comuni_list,bbox_to_anchor=(1.07, 1))
plt.savefig(file_path+'IMG/PRENOTAZIONI(branca,comune,sesso).png', format='png', dpi=300, bbox_inches = "tight")
plt.show()
del stats_df_M, stats_df_F, stats_df

# NUMERO DI PRENOTAZIONI PER BRANCA, PER COMUNE E PER SESSO (ASLbyASL)
for idx, asl in enumerate(asl_list):
    stats_df_M = (df_cup_extraction[(df_cup_extraction['sesso_assistito']==1) & (df_cup_extraction['asl']==asl)].groupby(
            'comune')['branca'].value_counts().unstack().fillna(0).T)/(df_cup_extraction['branca'].value_counts().sum())
    stats_df_M['pdf'] = stats_df_M.sum(axis=1)
    stats_df_M['cdf'] = stats_df_M['pdf'].cumsum()
    stats_df_M = stats_df_M.add_suffix('_M')
    
    stats_df_F = (df_cup_extraction[(df_cup_extraction['sesso_assistito']==2) & (df_cup_extraction['asl']==asl)].groupby(
            'comune')['branca'].value_counts().unstack().fillna(0).T)/(df_cup_extraction['branca'].value_counts().sum())
    stats_df_F['pdf'] = stats_df_F.sum(axis=1)
    stats_df_F['cdf'] = stats_df_F['pdf'].cumsum()
    stats_df_F = stats_df_F.add_suffix('_F')
    
    stats_df = pd.concat([stats_df_M, stats_df_F], axis=1, sort=False)
    stats_df = stats_df.reset_index()
    stats_df = stats_df.rename(columns={'index':'branca'})
    stats_df = stats_df.sort_values(by='pdf_M', ascending=False)
    
    comune_M = [item+'_M' for item in comuni_list]
    comune_F = [item+'_F' for item in comuni_list]

    fig, ax = plt.subplots()
    fig = stats_df.plot.bar(x='branca', y=comune_M, stacked=True, rot=90, legend=True, figsize=(32, 8), width=0.4, ax = ax, position=1)
    fig.set_xlabel("Branca")
    fig.set_ylabel("Frequency on Dataset - M/F")
    fig = stats_df.plot.bar(x='branca', y=comune_F, stacked=True, rot=90, width=0.4, ax = ax, position=0, grid=True, legend = False)
    ax.set_xlim(-0.5,len(stats_df_M)+0.5)
    ax.set_ylim(0,0.08)
    plt.legend(comuni_list, bbox_to_anchor=(1.07, 1))
    plt.savefig(file_path+'IMG/PRENOTAZIONI(branca,comune,sesso,asl_'+asl+').png', format='png', dpi=300, bbox_inches = "tight")
    plt.show()
    
del stats_df_M, stats_df_F, stats_df

# NUMERO DI PRENOTAZIONI PER BRANCA E PER FASCIA D'ETA
df_cup_extraction['fascia_eta'] = np.nan
df_cup_extraction['fascia_eta'].loc[df_cup_extraction['eta_assistito']<=17] = '0-17'
df_cup_extraction['fascia_eta'].loc[(df_cup_extraction['eta_assistito']>17) & (df_cup_extraction['eta_assistito']<=35)] = '18-35'
df_cup_extraction['fascia_eta'].loc[(df_cup_extraction['eta_assistito']>35) & (df_cup_extraction['eta_assistito']<=45)] = '36-45'
df_cup_extraction['fascia_eta'].loc[(df_cup_extraction['eta_assistito']>45) & (df_cup_extraction['eta_assistito']<=65)] = '46-65'
df_cup_extraction['fascia_eta'].loc[(df_cup_extraction['eta_assistito']>65)] = 'over65'

fascia_eta_list = list(df_cup_extraction['fascia_eta'].value_counts().index)
fascia_eta_list.sort(reverse=True)

stats_df = (df_cup_extraction.groupby('fascia_eta')['branca'].value_counts().unstack().fillna(0).T)/(df_cup_extraction['branca'].value_counts().sum())
stats_df = stats_df.reset_index()
stats_df = stats_df.rename(columns={'index':'branca'})
stats_df['pdf'] = stats_df.sum(axis=1)
stats_df = stats_df.sort_values(by='pdf', ascending=False)
stats_df['cdf'] = stats_df['pdf'].cumsum()

fig, ax = plt.subplots()
fig = stats_df.plot.bar(x='branca', y=fascia_eta_list, stacked=True, rot=90, legend=True, figsize=(24, 8), width=0.8, ax = ax)
fig.set_xlabel("Branca")
fig.set_ylabel("Frequency on Dataset")
#ax2 = ax.twinx()
#fig = stats_dfF.plot.bar(x='branca', y=fascia_eta_list, stacked=True, rot=90, width=0.4, ax = ax2, position=0, grid=False)
##ax.legend(bbox_to_anchor=(0.99, 0.95))
#ax.set_xlim(-0.5,len(stats_df2)-1)
plt.legend(bbox_to_anchor=(1.07, 1))
plt.savefig(file_path+'IMG/PRENOTAZIONI(branca,fascia_eta).png', format='png', dpi=300, bbox_inches = "tight")
plt.show()
del stats_df

# NUMERO DI PRENOTAZIONI PER BRANCA, PER FASCIA D'ETA E PER SESSO
stats_df_M = (df_cup_extraction[df_cup_extraction['sesso_assistito']==1].groupby('fascia_eta')['branca'].value_counts().unstack().fillna(0).T)/(df_cup_extraction['branca'].value_counts().sum())
#stats_df_M = stats_df_M.reset_index()
#stats_df_M = stats_df_M.rename(columns={'index':'branca'})
stats_df_M['pdf'] = stats_df_M.sum(axis=1)
#stats_df_M = stats_df_M.sort_values(by=index, ascending=False)
stats_df_M['cdf'] = stats_df_M['pdf'].cumsum()
stats_df_M = stats_df_M.add_suffix('_M')

stats_df_F = (df_cup_extraction[df_cup_extraction['sesso_assistito']==2].groupby('fascia_eta')['branca'].value_counts().unstack().fillna(0).T)/(df_cup_extraction['branca'].value_counts().sum())
#stats_df_F = stats_df_F.reset_index()
#stats_df_F = stats_df_F.rename(columns={'index':'branca'})
stats_df_F['pdf'] = stats_df_F.sum(axis=1)
#stats_df_F = stats_df_F.sort_values(by=index, ascending=False)
stats_df_F['cdf'] = stats_df_F['pdf'].cumsum()
stats_df_F = stats_df_F.add_suffix('_F')

stats_df = pd.concat([stats_df_M, stats_df_F], axis=1, sort=False)
stats_df = stats_df.reset_index()
stats_df = stats_df.rename(columns={'index':'branca'})
stats_df = stats_df.sort_values(by='pdf_M', ascending=False)

fascia_M = [item+'_M' for item in fascia_eta_list]
fascia_F = [item+'_F' for item in fascia_eta_list]

fig, ax = plt.subplots()
fig = stats_df.plot.bar(x='branca', y=fascia_M, stacked=True, rot=90, legend=True, figsize=(32, 8), width=0.4, ax = ax, position=1)
fig.set_xlabel("Branca")
fig.set_ylabel("Frequency on Dataset - M/F")
#ax2 = ax.twinx()
fig = stats_df.plot.bar(x='branca', y=fascia_F, stacked=True, rot=90, width=0.4, ax = ax, position=0, grid=True, legend = False)
#ax.legend(bbox_to_anchor=(0.99, 0.95))
ax.set_xlim(-0.5,len(stats_df_M)+0.5)
plt.legend(fascia_eta_list, bbox_to_anchor=(1.055, 1))
plt.savefig(file_path+'IMG/PRENOTAZIONI(branca,fascia_eta,sesso).png', format='png', dpi=300, bbox_inches = "tight")
plt.show()
del stats_df_M, stats_df_F, stats_df

# NUMERO DI PRENOTAZIONI PER BRANCA, PER FASCIA D'ETA E PER SESSO (ASLbyASL)
for idx, asl in enumerate(asl_list):   
    stats_df_M = (df_cup_extraction[(df_cup_extraction['sesso_assistito']==1) & (df_cup_extraction['asl']==asl)].groupby(
    'fascia_eta')['branca'].value_counts().unstack().fillna(0).T)/(df_cup_extraction['branca'].value_counts().sum())
    #stats_df_M = stats_df_M.reset_index()
    #stats_df_M = stats_df_M.rename(columns={'index':'branca'})
    stats_df_M['pdf'] = stats_df_M.sum(axis=1)
    #stats_df_M = stats_df_M.sort_values(by=index, ascending=False)
    stats_df_M['cdf'] = stats_df_M['pdf'].cumsum()
    stats_df_M = stats_df_M.add_suffix('_M')
    
    stats_df_F = (df_cup_extraction[(df_cup_extraction['sesso_assistito']==2) & (df_cup_extraction['asl']==asl)].groupby(
            'fascia_eta')['branca'].value_counts().unstack().fillna(0).T)/(df_cup_extraction['branca'].value_counts().sum())
    #stats_df_F = stats_df_F.reset_index()
    #stats_df_F = stats_df_F.rename(columns={'index':'branca'})
    stats_df_F['pdf'] = stats_df_F.sum(axis=1)
    #stats_df_F = stats_df_F.sort_values(by=index, ascending=False)
    stats_df_F['cdf'] = stats_df_F['pdf'].cumsum()
    stats_df_F = stats_df_F.add_suffix('_F')
    
    stats_df = pd.concat([stats_df_M, stats_df_F], axis=1, sort=False)
    stats_df = stats_df.reset_index()
    stats_df = stats_df.rename(columns={'index':'branca'})
    stats_df = stats_df.sort_values(by='pdf_M', ascending=False)
    
    fascia_M = [item+'_M' for item in fascia_eta_list]
    fascia_F = [item+'_F' for item in fascia_eta_list]
        
    fig, ax = plt.subplots()
    stats_df = stats_df.sort_values('branca')
    fig = stats_df.plot.bar(x='branca', y=fascia_M, stacked=True, rot=90, legend=True, figsize=(32, 8), width=0.4, ax = ax, position=1)
    fig.set_xlabel("Branca")
    fig.set_ylabel("Frequency on Dataset - M/F")
    #ax2 = ax.twinx()
    fig = stats_df.plot.bar(x='branca', y=fascia_F, stacked=True, rot=90, width=0.4, ax = ax, position=0, grid=True, legend = False)
    #ax.legend(bbox_to_anchor=(0.99, 0.95))
    ax.set_xlim(-0.5,len(stats_df_M)-0.5)
    ax.set_ylim(0,0.08)
    plt.legend(fascia_eta_list,bbox_to_anchor=(1.055, 1))
    plt.savefig(file_path+'IMG/PRENOTAZIONI(branca,fascia_eta,sesso,asl_'+asl+').png', format='png', dpi=300, bbox_inches = "tight")
    plt.show()
    del stats_df_M, stats_df_F, stats_df
#%% DISPONIBILITA VS ATTESA

# DISPONIBILITÀ E ATTESA IN MEDIA
df_mean_waits = df_cup_extraction.groupby('branca')[['attesa_disp','attesa_app']].mean().sort_values(by='attesa_disp', ascending=False)
df_mean_waits['branca'] = df_mean_waits.index
df_mean_waits = df_mean_waits.reset_index(drop=True)
df_mean_waits = df_mean_waits.sort_values('branca')
fig = df_mean_waits.plot.bar(x='branca', y=['attesa_disp','attesa_app'], rot=90, legend=True, width=0.7, figsize=(32, 8))
fig.set_ylim(0,140)
fig.set_xlabel("Branca")
fig.set_ylabel("Mean waiting time")
fig.legend(["Disponibilità", "Appuntamento"],bbox_to_anchor=(1.08, 1))
plt.savefig(file_path+'IMG/DISPONIBILIAvsATTESAmean.png', format='png', dpi=300, bbox_inches = "tight")
plt.show()

# DISPONIBILITÀ E ATTESA IN MEDIA (ASLbyASL)
for idx, asl in enumerate(asl_list):
    df_mean_waits = df_cup_extraction[df_cup_extraction['asl']==asl].groupby('branca')[['attesa_disp','attesa_app']].mean().sort_values(by='attesa_disp', ascending=False)
    df_mean_waits['branca'] = df_mean_waits.index
    df_mean_waits = df_mean_waits.reset_index(drop=True)
    
    df_mean_waits = df_mean_waits.sort_values('branca')
    fig = df_mean_waits.plot.bar(x='branca', y=['attesa_disp','attesa_app'], rot=90, legend=True, width=0.7, figsize=(32, 8))
    fig.set_ylim(0,140)
    fig.set_xlabel("Branca")
    fig.set_ylabel("Mean waiting time")
    fig.legend(["Disponibilità", "Appuntamento"],bbox_to_anchor=(1.08, 1))
    plt.savefig(file_path+'IMG/DISPONIBILIAvsATTESAmean_asl_'+asl+'.png', format='png', dpi=300, bbox_inches = "tight")
    plt.show()
    
# DISPONIBILITAvsATTESA (Scatter)
plt.figure(figsize=(8, 8))
#sns.lmplot('attesa_disp', 'attesa_app', data=df_cup_extraction, hue='asl', fit_reg=False)
sns.scatterplot(x = 'attesa_disp', y = 'attesa_app', hue='asl', 
                data=df_cup_extraction.sample(100000), alpha=.7, s=80, edgecolor="none")
plt.xlabel("Disponibilità")
plt.ylabel("Attesa")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.savefig(file_path+'IMG/DISPONIBILITAvsATTESA(asl).png', format='png', dpi=300, bbox_inches = "tight")
plt.show()

#%% DISPONIBILITÀ E ATTESA per ANNO

# DISPONIBILITA' IN GIORNI PER BRANCA E PER ANNO
plt.figure(figsize=(32, 8))
fig = sns.boxplot(x='branca', y='attesa_disp', hue='anno_ins', data=df_cup_extraction.sort_values(by='branca'))
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
fig.set(ylabel='Disponibilità', xlabel='Branca')
#handles, _ = fig.get_legend_handles_labels()
#fig.legend(handles, ["Male", "Female"])
fig.set_yscale("symlog")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.savefig(file_path+'IMG/DISPONIBILITA(branca,anno).png', format='png', dpi=300, bbox_inches = "tight")
plt.show()

# DISPONIBILITA' IN GIORNI PER BRANCA E PER ANNI - ASLbyASL
for idx, asl in enumerate(asl_list):
    plt.figure(figsize=(32, 8))
    fig = sns.boxplot(x='branca', y='attesa_disp', hue='anno_ins', data=df_cup_extraction[df_cup_extraction['asl']==asl].sort_values(by='branca'))
    fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
    fig.set(ylabel='Disponibilità', xlabel='Branca')
    #handles, _ = fig.get_legend_handles_labels()
    #fig.legend(handles, ["Male", "Female"])
    fig.set_yscale("symlog")
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.savefig(file_path+'IMG/DISPONIBILITA(branca,anno,asl_'+asl+').png', format='png', dpi=300, bbox_inches = "tight")
    plt.show()


# ATTESA IN GIORNI PER BRANCA E PER ANNO
plt.figure(figsize=(32, 8))
fig = sns.boxplot(x='branca', y='attesa_app', hue='anno_ins', data=df_cup_extraction.sort_values(by='branca'))
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
fig.set(ylabel='Attesa', xlabel='Branca')
#handles, _ = fig.get_legend_handles_labels()
#fig.legend(handles, ["Male", "Female"])
fig.set_yscale("symlog")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.savefig(file_path+'IMG/ATTESA(branca,anno).png', format='png', dpi=300, bbox_inches = "tight")
plt.show()

# ATTESA IN GIORNI PER BRANCA E PER ANNI - ASLbyASL
for idx, asl in enumerate(asl_list):
    plt.figure(figsize=(32, 8))
    fig = sns.boxplot(x='branca', y='attesa_app', hue='anno_ins', data=df_cup_extraction[df_cup_extraction['asl']==asl].sort_values(by='branca'))
    fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
    fig.set(ylabel='Attesa', xlabel='Branca')
    #handles, _ = fig.get_legend_handles_labels()
    #fig.legend(handles, ["Male", "Female"])
    fig.set_yscale("symlog")
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.savefig(file_path+'IMG/ATTESA(branca,anno,asl_'+asl+').png', format='png', dpi=300, bbox_inches = "tight")
    plt.show()
    

#%% DISPONIBILITÀ E ATTESA per ASL

# DISPONIBILITA' IN GIORNI PER BRANCA E PER ASL
plt.figure(figsize=(32, 8))
fig = sns.boxplot(x='branca', y='attesa_disp', hue='asl', data=df_cup_extraction.sort_values(by='branca'))
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
fig.set(ylabel='Disponibilità', xlabel='Branca')
#handles, _ = fig.get_legend_handles_labels()
#fig.legend(handles, ["Male", "Female"])
fig.set_yscale("symlog")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.savefig(file_path+'IMG/DISPONIBILITA(branca,asl).png', format='png', dpi=300, bbox_inches = "tight")
plt.show()

## DISPONIBILITA' IN GIORNI PER BRANCA E PER ASL (COMUNEbyCOMUNE)
#for idx, comune in enumerate(comuni_list):
#    plt.figure(figsize=(32, 8))
#    fig = sns.boxplot(x='branca', y='attesa_disp', hue='asl', data=df_cup_extraction[df_cup_extraction['comune']==comune].sort_values(by='branca'))
#    fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
#    fig.set(ylabel='Disponibilità', xlabel='Branca')
#    #handles, _ = fig.get_legend_handles_labels()
#    #fig.legend(handles, ["Male", "Female"])
#    fig.set_yscale("symlog")
#    plt.legend(bbox_to_anchor=(1.05, 1))
#    plt.savefig('IMG/DISPONIBILITA(branca,asl,comune_'+comune+').png', format='png', dpi=300, bbox_inches = "tight")
#    plt.show()


stats_df = df_cup_extraction.groupby(['branca','anno_ins'])['attesa_disp'].mean().unstack().sort_values(by='branca')
fig = stats_df.plot.bar(rot=90, legend=True, width=0.7, figsize=(32, 8))
fig.set_ylim(0,80)
fig.set_xlabel("Branca")
fig.set_ylabel("Mean waiting time")
fig.legend(bbox_to_anchor=(1.05, 1))
plt.savefig(file_path+'IMG/DISPONIBILITAmean(anno).png', format='png', dpi=300, bbox_inches = "tight")
plt.show()

# DISPONIBILITA' MEDIA PER BRANCA e ANNI (ASLbyASL)
for idx, asl in enumerate(asl_list):
    stats_df = df_cup_extraction[df_cup_extraction['asl']==asl].groupby(['branca','anno_ins'])['attesa_disp'].mean().unstack().sort_values(by='branca')
    fig = stats_df.plot.bar(rot=90, legend=True, width=0.7, figsize=(32, 8))
    fig.set_ylim(0,80)
    fig.set_xlabel("Branca")
    fig.set_ylabel("Mean waiting time")
    fig.legend(bbox_to_anchor=(1.05, 1))
    plt.savefig(file_path+'IMG/DISPONIBILITAmean(anno,asl_'+asl+').png', format='png', dpi=300, bbox_inches = "tight")
    plt.show()

del stats_df

# ATTESA IN GIORNI PER BRANCA E PER ASL
plt.figure(figsize=(32, 8))
fig = sns.boxplot(x='branca', y='attesa_app', hue='asl', data=df_cup_extraction)
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
fig.set(ylabel='Attesa', xlabel='Branca')
#handles, _ = fig.get_legend_handles_labels()
#fig.legend(handles, ["Male", "Female"])
fig.set_yscale("symlog")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.savefig(file_path+'IMG/ATTESA(branca,asl).png', format='png', dpi=300, bbox_inches = "tight")
plt.show()

# ATTESA' IN GIORNI PER BRANCA E PER ASL (COMUNEbyCOMUNE)
for idx, comune in enumerate(comuni_list):
    plt.figure(figsize=(32, 8))
    fig = sns.boxplot(x='branca', y='attesa_app', hue='asl', data=df_cup_extraction[df_cup_extraction['comune']==comune].sort_values(by='branca'))
    fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
    fig.set(ylabel='Attesa', xlabel='Branca')
    #handles, _ = fig.get_legend_handles_labels()
    #fig.legend(handles, ["Male", "Female"])
    fig.set_yscale("symlog")
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.savefig(file_path+'IMG/ATTESA(branca,asl,comune_'+comune+').png', format='png', dpi=300, bbox_inches = "tight")
    plt.show()

# ATTESA/DISPONIBILITA' RISPETTO ALLE PRIORITA'
stats_df = df_cup_extraction.groupby(['branca','priorita'])['attesa_disp'].mean().unstack().sort_values(by='branca')
fig = stats_df.plot.bar(rot=90, legend=True, width=0.7, figsize=(32, 8))
fig.set_ylim(0,40)
fig.set_xlabel("Branca")
fig.set_ylabel("Mean waiting time")
fig.legend(bbox_to_anchor=(1.05, 1))
plt.savefig(file_path+'IMG/DISPONIBILITAmean(branca,priorita).png', format='png', dpi=300, bbox_inches = "tight")
plt.show()

stats_df = df_cup_extraction.groupby(['branca','priorita'])['attesa_app'].mean().unstack().sort_values(by='branca')
fig = stats_df.plot.bar(rot=90, legend=True, width=0.7, figsize=(32, 8))
fig.set_ylim(0,120)
fig.set_xlabel("Branca")
fig.set_ylabel("Mean waiting time")
fig.legend(bbox_to_anchor=(1.05, 1))
plt.savefig(file_path+'IMG/ATTESAmean(branca,priorita).png', format='png', dpi=300, bbox_inches = "tight")
plt.show()

## ATTESA/DISPONIBILITA' RISPETTO ALLE PRIORITA' BOX
#stats_df = df_cup_extraction.groupby(['branca','priorita'])['attesa_disp']
#fig = stats_df.plot.bar(rot=90, legend=True, width=0.7, figsize=(32, 8))
#fig = sns.boxplot(x='branca', y='attesa_disp', hue='priorita', data=stats_df)
#fig.set_ylim(0,40)
#fig.set_xlabel("Branca")
#fig.set_ylabel("Mean waiting time")
#fig.legend(bbox_to_anchor=(1.05, 1))
#plt.savefig(file_path+'IMG/DISPONIBILITAmean(branca,priorita)_box.png', format='png', dpi=300, bbox_inches = "tight")
#plt.show()
#
#stats_df = df_cup_extraction.groupby(['branca','priorita'])['attesa_app'].unstack().sort_values(by='branca')
#fig = stats_df.plot.bar(rot=90, legend=True, width=0.7, figsize=(32, 8))
#fig.set_ylim(0,120)
#fig.set_xlabel("Branca")
#fig.set_ylabel("Mean waiting time")
#fig.legend(bbox_to_anchor=(1.05, 1))
#plt.savefig(file_path+'IMG/ATTESAmean(branca,priorita)_box.png', format='png', dpi=300, bbox_inches = "tight")
#plt.show()
#%% SALVATAGGIO SU FILES PICKLE
print('Salvataggio su files PICKLE...', end="")
pickling_on = open(file_path+"DATASET/df_cup_RF.pickle","wb")
pickle.dump(df_cup_extraction, pickling_on)
pickling_on.close()


#df_cup_diabe = df_cup_extraction[df_cup_extraction['branca']=='DIABETOLOGIA']
#print('Salvataggio su files PICKLE...', end="")
#pickling_on = open("DATASET/CLEAN/df_cup_diabe.pickle","wb")
#pickle.dump(df_cup_diabe, pickling_on)
#pickling_on.close()
#
#df_cup_cardio = df_cup_extraction[df_cup_extraction['branca']=='CARDIOLOGIA']
#print('Salvataggio su files PICKLE...', end="")
#pickling_on = open("DATASET/CLEAN/df_cup_cardio.pickle","wb")
#pickle.dump(df_cup_cardio, pickling_on)
#pickling_on.close()
















