#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:01:06 2019

@author: enx
"""
import pandas as pd
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import pickle

#%% GLOBAL VARIABLES
_external_path = os.path.abspath(os.path.join(file_dir, "../../../../Pycodes"))


# LOAD DEI DATASET

csv_path = os.path.abspath(
    os.path.join(_external_path, "csv/random")   # mini random
)
os.makedirs(csv_path, exist_ok=True) # create folder

pickle_path = os.path.abspath(
    os.path.join(_external_path, "pickle")
)

_cup_datasets_path = os.path.abspath(
    os.path.join(_external_path, "cup_datasets")
)

#%%

import csv


# Carico le prestazioni (database e regionali)
# df = cupload.load_dataset(file=csv_path+'df_cup_c.csv')

#%%

# Asl C
# df_aslC = df[df.sa_asl=='C'].copy()
with open(os.path.join(pickle_path, "df_cup_c_aslC.pickle"), "rb") as f:
    df_aslC = pickle.load(f)

# Bad dataframe
df_aslC["is_good_data"] = ~(df_aslC[df_aslC.columns] == 99999999).any(axis=1)

df_bad = df_aslC[df_aslC["is_good_data"] == False]

df_aslC_orig = df_aslC.copy()

df_aslC = df_aslC[df_aslC["is_good_data"] == True]

print(df_aslC.shape[0])
df_aslC = df_aslC.sample(n=3*10**2) #max 10**7

# Selection of good columns
# Inspired by: https://stackoverflow.com/a/47107164


def good_columns_dataframe(df, df_bad, cols):
    df_cols = df[cols]
#    df_bad_cols = df_bad[cols]
#    df_merge_cols = df_cols.merge(
#        df_bad_cols.drop_duplicates(), how="left", indicator=True
#    )
#    df_good_cols = (
#        df_merge_cols[df_merge_cols["_merge"] == "left_only"][cols]
#        .drop_duplicates()
#        .sort_values(cols)
#        .reset_index(drop=True)
#    )
    df_good_cols = df_cols.drop_duplicates()
    return df_good_cols


def print_entity_quality(entity_cols, df_good_entity, entity_name, col_duplicate):
    print(f"Statistics for {entity_name}:")
    total_entity_count = df_aslC[entity_cols].drop_duplicates(col_duplicate).shape[0]
    good_entity_count = df_good_entity.shape[0]
    print(f"Total {entity_name} count: %d" % total_entity_count)
    print(f"Good {entity_name} count: %d" % good_entity_count)
    print(
        f"Percentage good {entity_name}: %f%%"
        % (good_entity_count * 100.0 / total_entity_count)
    )


# Good practitioners
practitioners_cols = ["sa_med_id"]
df_good_practitioners = good_columns_dataframe(df_aslC, df_bad, practitioners_cols)
print_entity_quality(
    practitioners_cols, df_good_practitioners, "practitioners", "sa_med_id"
)
df_good_practitioners.to_csv(
    os.path.join(csv_path, "practitioner.csv"), quoting=csv.QUOTE_ALL
)

# Good patients
patients_cols = ["sa_ass_cf", "sa_sesso_id", "sa_comune_id"]
df_good_patients = good_columns_dataframe(df_aslC, df_bad, patients_cols)
print_entity_quality(patients_cols, df_good_patients, "patients", "sa_ass_cf")
df_good_patients.to_csv(os.path.join(csv_path, "patient.csv"), quoting=csv.QUOTE_ALL)


# Good booking staff
bookingStaff1_cols = ["sa_ut_id", "sa_asl"]
df_good_bookingStaff1 = good_columns_dataframe(df_aslC, df_bad, bookingStaff1_cols)
print_entity_quality(
    bookingStaff1_cols, df_good_bookingStaff1, "bookingStaff1", "sa_ut_id"
)


bookingStaff2_cols = ["sa_utente_id", "sa_asl"]
df_good_bookingStaff2 = good_columns_dataframe(df_aslC, df_bad, bookingStaff2_cols)
print_entity_quality(
    bookingStaff2_cols, df_good_bookingStaff2, "bookingStaff2", "sa_utente_id"
)


df_good_bookingStaff = pd.concat(
    [
        df_good_bookingStaff1,
        df_good_bookingStaff2.rename(columns={"sa_utente_id": "sa_ut_id"}),
    ],
    ignore_index=True,
).drop_duplicates()
print(df_good_bookingStaff)
df_good_bookingStaff.to_csv(
    os.path.join(csv_path, "booking-staff.csv"), quoting=csv.QUOTE_ALL
)


# Good Unita Eroganti

csv_file = "dec_unita_eroganti_201910031115.csv"
data = pd.read_csv(os.path.join(_cup_datasets_path, csv_file), sep=";")
unita_eroganti_cols = ["sa_uop_codice_id", "struttura_comune", "struttura_codasl"]
df_all_unita_eroganti = data[unita_eroganti_cols].copy().drop_duplicates()

unita_eroganti_cols = ["sa_uop_codice_id"]
df_good_unita_eroganti = good_columns_dataframe(df_aslC, df_bad, unita_eroganti_cols)
print_entity_quality(
    unita_eroganti_cols, df_good_unita_eroganti, "unita_eroganti", "sa_uop_codice_id"
)


df_all_unita_eroganti["sa_uop_codice_id"]
# df_good_unita_eroganti
merged = pd.merge(df_all_unita_eroganti, df_good_unita_eroganti)  # 544

if len(df_good_unita_eroganti) != len(merged):
    print("WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: è un cavolo di problema: UOP")
    merged = merged[merged.struttura_codasl == 'C']
    merged.to_csv(
        os.path.join(csv_path, "appointment-provider.csv"), quoting=csv.QUOTE_ALL
    )
else:
    print("tutto ok Uop")
    merged.to_csv(
        os.path.join(csv_path, "appointment-provider.csv"), quoting=csv.QUOTE_ALL
    )

# Good Prestazioni

csv_file = "prestazioni_to_branche_cup3.csv"
data = pd.read_csv(os.path.join(_cup_datasets_path, csv_file), sep=",")
prestazioni_cols = ["id_prestazione", "id_branca"]
df_all_prestazioni = data[prestazioni_cols].copy().drop_duplicates()

df_all_prest_to_branca = (
    df_all_prestazioni.groupby("id_prestazione")["id_branca"]
    .apply(list)
    .reset_index(name="list_branca")
)
df_all_prest_to_branca = df_all_prest_to_branca.append(
    {"id_prestazione": 99999999, "list_branca": []}, ignore_index=True
)


prest_cols = ["indx_prestazione", "sa_pre_id"]  # ,'sa_branca_id'
df_good_prest = good_columns_dataframe(df_aslC, df_bad, prest_cols)
print_entity_quality(prest_cols, df_good_prest, "prest", "sa_pre_id")


csv_file = "prestazioni.csv"
prest_data = pd.read_csv(os.path.join(_cup_datasets_path, csv_file), sep=";")
prestazioni_cols = ["id", "descrizione"]
df_orig_prestazioni = prest_data[prestazioni_cols].copy().drop_duplicates()


merged_prest = pd.merge(
    left=df_good_prest, right=prest_data, left_on="indx_prestazione", right_on="id"
).drop(columns="id")

merged_prest = (
    merged_prest.groupby(["indx_prestazione", "descrizione"])["sa_pre_id"]
    .apply(list)
    .reset_index(name="list_sa_pre_id")
)


merged_prest_branche = pd.merge(
    left=merged_prest,
    right=df_all_prest_to_branca,
    left_on="indx_prestazione",
    right_on="id_prestazione",
).drop(columns="id_prestazione")

merged_prest_branche.to_csv(
    os.path.join(csv_path, "health-service-provision.csv"), quoting=csv.QUOTE_ALL
)

if len(merged_prest_branche) != len(
    merged_prest_branche["indx_prestazione"].drop_duplicates()
):
    print("WARNING: è un cavolo di problema: PREST")
else:
    print("tutto ok: PREST")


# Good Branca

csv_file = "branche_cup3.csv"
branche_cup = pd.read_csv(os.path.join(_cup_datasets_path, csv_file), sep=",")
branche_cols = ["id", "descrizione"]
df_all_branche = branche_cup[branche_cols].copy().drop_duplicates()


csv_file = "df_branche_indx.csv"
branche_index = pd.read_csv(os.path.join(_cup_datasets_path, csv_file), sep=",")
branche_cols = ["codice", "indx"]
branche_index = branche_index[branche_cols].drop_duplicates()

csv_file = "branche_201910031114.csv"
branche_old = pd.read_csv(os.path.join(_cup_datasets_path, csv_file), sep=";")
branche_cols = ["id_branca", "descrizione"]
branche_old = branche_old[branche_cols].drop_duplicates()


merged_branche = pd.merge(
    left=branche_index, right=branche_cup, left_on="indx", right_on="id"
)
merged_branche.pop("id")
merged_full_branche = pd.merge(
    left=merged_branche, right=branche_old, left_on="codice", right_on="id_branca"
).drop(columns="codice")


branca_cols = ["sa_branca_id"]
df_good_branca = good_columns_dataframe(df_aslC, df_bad, branca_cols)
print_entity_quality(branca_cols, df_good_branca, "branca", "sa_branca_id")

# NEO è 99999!

merged_full_branche.to_csv(
    os.path.join(csv_path, "medical-branch.csv"), quoting=csv.QUOTE_ALL
)

if len(merged_full_branche) != len(merged_full_branche["id_branca"].drop_duplicates()):
    print("WARNING: è un cavolo di problema: branche")
else:
    print("tutto ok: branche")


#%% Relations


# relationship practitioner patient with column impegnativa  'sa_num_prestazioni',  indx_prestazione OPPURE sa_pre_id
PREST_KIND = "indx_prestazione"  # sa_pre_id    indx_prestazione
referral_cols = [
    "sa_branca_id",
    PREST_KIND,
    "sa_impegnativa_id",
    "indx_impegnativa",
    "sa_data_prescr",
    "sa_eta_id",
    "sa_ese_id_lk",
    "sa_classe_priorita",
]


res_relation_cols = list(df_good_practitioners.columns) + list(df_good_patients.columns) + referral_cols

df_practitioner_patient_relation= good_columns_dataframe(df_aslC, df_bad, res_relation_cols
)


cols = list(df_practitioner_patient_relation.columns)
cols.remove(PREST_KIND)
df_impegnativa_to_multi = (
    df_practitioner_patient_relation.groupby(cols)[PREST_KIND]
    .apply(list)
    .reset_index(name=f"list_{PREST_KIND}")
)

df_impegnativa_to_multi.to_csv(
    os.path.join(csv_path, "referral.csv"), quoting=csv.QUOTE_ALL
)

#
## alternate Referall
#referall_basic_cols = [
#    "sa_impegnativa_id",
#    "indx_impegnativa",
#    "sa_data_prescr",
#    "sa_eta_id",
#    "sa_ese_id_lk",
#    "sa_classe_priorita",
#]
#df_referall_basic = good_columns_dataframe(df_aslC, df_bad, referall_basic_cols)
#df_impegnativa_to_multi.to_csv(os.path.join(csv_path, "referral_BASIC.csv"))
#
#referall_full_cols = (
#    list(df_good_practitioners.columns)
#    + list(df_good_patients.columns)
#    + referall_basic_cols
#    + ["sa_branca_id", "sa_pre_id"]
#)
#df_referall_full = good_columns_dataframe(df_aslC, df_bad, referall_full_cols)
#df_impegnativa_to_multi.to_csv(os.path.join(csv_path, "referral_FULL.csv"))


# relationship uop operator with column prenotazione!  'sa_num_prestazioni',
reservation_cols = [
    "sa_dti_id",
    "sa_pre_id",
    "sa_data_pren",
    "sa_data_app",
    "sa_data_ins",
    "sa_uop_codice_id",
    "sa_utente_id",
    "sa_ut_id",
    "sa_num_prestazioni",
    "sa_is_ad",
    "sa_asl",
]


prenot_cols = list(df_practitioner_patient_relation.columns) + reservation_cols
df_uop_operator_prenot_relation = good_columns_dataframe(df_aslC, df_bad, prenot_cols)



df_uop_operator_prenot_relation.to_csv(
    os.path.join(csv_path, "reservation.csv"), quoting=csv.QUOTE_ALL
)
#
#
## alternate UOP
#reservation_basic_cols = [
#    "sa_dti_id",
#    "sa_data_pren",
#    "sa_data_app",
#    "sa_data_ins",
#    "sa_num_prestazioni",
#    "sa_is_ad",
#    "sa_asl",
#]
#df_reservation_basic = good_columns_dataframe(df_aslC, df_bad, reservation_basic_cols)
#df_reservation_basic.to_csv(os.path.join(csv_path, "reservation_BASIC.csv"))
#
#reservation_full_cols = (
#    referall_basic_cols
#    + ["sa_uop_codice_id", "sa_utente_id", "sa_ut_id"]
#    + reservation_basic_cols  # list(df_good_bookingStaff.columns) +
#    + ["sa_pre_id"]  # indx_impegnativa
#)
#df_reservation_full = good_columns_dataframe(df_aslC, df_bad, reservation_full_cols)
#df_reservation_full.to_csv(os.path.join(csv_path, "reservation_FULL.csv"))
#
#
##%%
#
#final_practitioners = good_columns_dataframe(
#    df_practitioner_patient_relation, df_bad, practitioners_cols
#)
#final_practitioners.to_csv(os.path.join(csv_path, "final_practitioners.csv"))
#
#final_patients = good_columns_dataframe(
#    df_practitioner_patient_relation, df_bad, patients_cols
#)
#final_patients.to_csv(os.path.join(csv_path, "final_patients.csv"))

