#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version 3, wip, of CUP_etl_2v1
ETL module, Jupyter like, that trandofrms data from csv/database to pikle
This could be improved by accessing direcly the Postgres database


@author:
"""
#%% Importing
from cup_project.config import data_dir, reportings_dir
import etl_utils as cup_load

import numpy as np


# Dataset paths
raw_data_dir = data_dir / 'raw'
interm_data_dir = raw_data = data_dir / 'intermediate'
describe_data_dir = reportings_dir / 'description'

#%% Pickle of CUP
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
# file_name = 'dwh_mis_cup'
# sep=','
# dates_cup = ['sa_data_ins','sa_data_pren','sa_data_app','sa_data_prescr']
# cup_load.load_describe_save(file_name, sep, raw_data_dir, describe_data_dir, interm_data_dir, dates_cassa, types_cassa)



#%% Pickle of CASSA

file_name = 'dwh_mis_cassa'
sep=';'
dates_cassa = ['sa_data_ins','sa_data_prest','sa_data_mov']
cup_load.load_describe_save(file_name, sep, raw_data_dir, describe_data_dir, interm_data_dir, dates_cassa, types_cassa)


#%% Pickle of BRANCHE
file_name = 'branche'
sep=';'
cup_load.load_describe_save(file_name, sep, raw_data_dir, describe_data_dir, interm_data_dir)



#%% Pickle of PRESTAZIONI

#TODO