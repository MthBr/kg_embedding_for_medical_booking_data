# import datetime

import logging
import os
import sys
import numpy as np
import pickle
import csv
import datetime as dt
import pandas as pd
import matplotlib.pylab as plt
import pickle

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_local_dir = False
CUP_WORK = True

#%% Dirs variables

if _local_dir:
    working_dir = os.path.join(file_dir, os.pardir)
else:
    working_dir = os.path.abspath(os.path.join(file_dir, f"../../../../Pycodes/"))
    if CUP_WORK:
        working_dir = os.path.join(working_dir, "cup_kg")

_model_path = os.path.abspath(os.path.join(working_dir, "models"))
_3ples_directory = os.path.abspath(os.path.join(working_dir, "3ples"))
_dictionay_directory = os.path.abspath(os.path.join(working_dir, "dicts"))
_pickle_path = os.path.abspath(os.path.join(working_dir, os.pardir, "pickle"))
_csv_path = os.path.abspath(
    os.path.join(working_dir, os.pardir, "csv", "random_1747_1_True")
)
_cup_datasets_path = os.path.abspath(
    os.path.join(working_dir, os.pardir, "cup_datasets")
)
_txt_path = os.path.abspath(os.path.join(working_dir, os.pardir, "txt"))
os.makedirs(_txt_path, exist_ok=True)

#%% GLOBAL VARIABLES

entity_name_patient = "patient"
entity_id_patient = "encrypted-nin-id"
entity_name_provision = "health-service-provision"
entity_id_provision = "refined-health-service-id"
cf_id = "sa_ass_cf"
prest_id = "indx_prestazione"

KEYSPACE_NAME = "cup_1747_1"  # cup_1747_1 cup_100_1
dict_name_id = {
    "medical-branch": "medical-branch-id",
    "appointment-provider": "referral-centre-id",
    "practitioner": "practitioner-id",
    "health-service-provision": "refined-health-service-id",
    "patient": "encrypted-nin-id",
    "booking-staff": "booking-agent-id",
}
dict_id_col = {
    "medical-branch-id": "sa_branca_id",
    "referral-centre-id": "sa_uop_codice_id",
    "practitioner-id": "sa_med_id",
    "refined-health-service-id": "indx_prestazione",
    "encrypted-nin-id": "sa_ass_cf",
    "booking-agent-id": "sa_utente_id",
}
dict_col_name = {
    "sa_branca_id": "medical-branch",
    "sa_uop_codice_id": "appointment-provider",
    "sa_med_id": "practitioner",
    "indx_prestazione": "health-service-provision",
    "sa_ass_cf": "patient",
    "sa_utente_id": "booking-staff",
    "sa_ut_id": "booking-staff",
}
dict_col_id = {
    "sa_branca_id": "medical-branch-id",
    "sa_uop_codice_id": "referral-centre-id",
    "sa_med_id": "practitioner-id",
    "indx_prestazione": "refined-health-service-id",
    "sa_ass_cf": "encrypted-nin-id",
    "sa_utente_id": "booking-agent-id",
    "sa_data_ins": "last-reservation-change-date",
    "sa_data_prescr": "referral-date",
    "sa_data_app": "booked-date",
    "sa_data_pren": "reservation-date",
    "sa_comune_id": "nuts-istat-code",
    "sa_sesso_id": "gender",
    "sa_ut_id": "updating-booking-agent-id",
    "sa_num_prestazioni": "number-of-health-services",
    "sa_classe_priorita": "priority-code",
    "sa_asl": "local-health-department-id",
    "sa_eta_id": "patient-age",
    "sa_gg_attesa": "res-waiting-days",
    "sa_gg_attesa_pdisp": "first-res-waiting-days",
}
dict_table = {
    "date_col": ["sa_data_ins", "sa_data_prescr", "sa_data_app", "sa_data_pren"],
    "category_col": [
        "sa_ass_cf",
        "sa_utente_id",
        "sa_uop_codice_id",
        "sa_comune_id",
        "sa_branca_id",
        "sa_med_id",
        "sa_sesso_id",
        "sa_ut_id",
        "sa_num_prestazioni",
        "sa_classe_priorita",
        "sa_asl",
        "indx_prestazione",
    ],
    "number_col": ["sa_eta_id", "sa_gg_attesa", "sa_gg_attesa_pdisp"],
}

cup_date_id = "sa_data_pren"
time_limit_training = "{year}-{month}-{day}".format(year=2018, month=1, day=1)
time_limit_training = dt.datetime.strptime(time_limit_training, "%Y-%m-%d")
time_period_days = 365
time_limit_test = time_limit_training + dt.timedelta(days=time_period_days)
RSEED = 0
N_CORES = -2
HOW_MANY_SAMPLES = 2
entities_names = [
    "medical-branch",
    "appointment-provider",
    "practitioner",
    "health-service-provision",
    "patient",
    "booking-staff",
]
relations_names = ["referral", "reservation", "health-care", "provision"]


#%% IMPORT

dictionay_name = f"{KEYSPACE_NAME}_dict_embeddings.pkl"
dictionay_file = os.path.join(_dictionay_directory, dictionay_name)
with open(dictionay_file, "rb") as open_file:
    dict_emb_concepts = pickle.load(open_file)

dictionay_name = f"{KEYSPACE_NAME}_dict_concepts.pkl"
dictionay_file = os.path.join(_dictionay_directory, dictionay_name)
with open(dictionay_file, "rb") as open_file:
    dict_entities = pickle.load(open_file)

csv_name = f"triples_{KEYSPACE_NAME}.csv"
csv_file = os.path.join(_3ples_directory, csv_name)
with open(csv_file) as open_file:
    triples = list(csv.reader(open_file, delimiter=","))

csv_name = f"df_aslC.csv"
csv_file = os.path.join(_csv_path, csv_name)
df_cup = pd.read_csv(csv_file, index_col=0)
# transform to datetime
for date_col in ["sa_data_ins", "sa_data_prescr", "sa_data_app", "sa_data_pren"]:
    df_cup[date_col] = pd.to_datetime(df_cup[date_col], errors="coerce")

csv_name = "prestazioni.csv"
csv_file = os.path.join(_cup_datasets_path, csv_name)
df_prest = pd.read_csv(csv_file, sep=";", index_col=0)

dict_prest = df_prest["descrizione"].to_dict()
del df_prest

csv_name = "prestazioni_to_branche_cup3.csv"
csv_file = os.path.join(_cup_datasets_path, csv_name)
df_prest_to_branche = pd.read_csv(csv_file, sep=",")

df_prest_to_branche = (
    df_prest_to_branche.fillna("")
    .groupby("id_prestazione")["id_branca"]
    .apply(list)
    .reset_index(name="list_branca")
)
df_prest_to_branche = df_prest_to_branche.append(
    {"id_prestazione": 99999999, "list_branca": []}, ignore_index=True
)
dict_prest_to_branche = df_prest_to_branche.set_index("id_prestazione")[
    "list_branca"
].to_dict()
del df_prest_to_branche

#%% FUNCTIONS

# from id to identification
def from_id_to_identification(triples, dict_entities, entity_id):
    dict_map = {}
    for row in triples:
        if row[1] == f"@has-{entity_id}":
            dict_map[row[0]] = dict_entities[row[2]]["value"]
    return dict_map


def create_embedding_dataframe(
    dict_emb_concepts, entity_name, with_mapping=False, dict_map=None, index_col=None
):
    if with_mapping:
        # create embedding dict with identification
        dict_new = {}
        for key, value in dict_emb_concepts[entity_name].items():
            dict_new[dict_map[key]] = value

        # create dataFrame of embedding
        df_emb = pd.DataFrame.from_dict(dict_new, orient="index")
        df_emb.index.name = index_col
    else:
        # create dataFrame of embedding
        df_emb = pd.DataFrame.from_dict(dict_emb_concepts[entity_name], orient="index")
        df_emb.index.name = entity_name

    return df_emb


#%% MAIN

names = entities_names
# names = relations_names
# names = entities_names + relations_names

with open(os.path.join(_pickle_path, KEYSPACE_NAME + "_emb_concepts.pkl"), "rb") as f:
    df_emb = pickle.load(f)

with open(os.path.join(_csv_path, KEYSPACE_NAME + "_clusters.csv")) as f:
    clustering = pd.read_csv(f, index_col=0)
df_emb["clusters"] = clustering
max_K = int(clustering.max())

del clustering

# dict of mapping
dict_map = {}
for entity_name in names:
    dict_map.update(
        from_id_to_identification(triples, dict_entities, dict_name_id[entity_name])
    )

entity_name = "health-service-provision"
dict_prest_to_clusters = {}
dict_descriptions = {new_list: [] for new_list in list(np.arange(1, max_K + 1))}
for index, cluster in df_emb["clusters"][df_emb["entity"] == entity_name].iteritems():
    dict_descriptions[cluster].append(
        [
            dict_prest[dict_map[index]],
            df_cup[dict_id_col[dict_name_id[entity_name]]].value_counts()[
                dict_map[index]
            ],
            dict_prest_to_branche[dict_map[index]],
        ]
    )
    dict_prest_to_clusters[dict_map[index]] = cluster

from operator import itemgetter

for key, value in dict_descriptions.items():
    dict_descriptions[key] = sorted(value, key=itemgetter(1), reverse=True)

# print dictionary
with open(os.path.join(_txt_path, "clusters_healthservices" + ".txt"), "w") as f:
    for key, value in dict_descriptions.items():
        for text in value:
            f.write("%s:%s\n" % (key, text))
        f.write("\n")

# create column cluster
df_cup["clusters"] = None
for index, indx_prest in df_cup[dict_id_col[dict_name_id[entity_name]]].iteritems():
    df_cup.set_value(index, "clusters", dict_prest_to_clusters[indx_prest])


DIM_SPACE = len(df_emb.columns) - len(names) - 2

# create an empty dataFrame of labels/prescriptions
df_label = pd.DataFrame(
    0,
    index=[dict_map[x] for x in df_emb.loc[df_emb["patient"] == 1].index.tolist()],
    columns=list(df_emb["clusters"][df_emb["health-service-provision"] == 1].unique()),
)
df_label.index = df_label.index.map(int)

# fill dataFrame with 1 if there is at least one record
for index, row in df_cup.iterrows():
    if time_limit_training <= row[cup_date_id] < time_limit_test:
        df_label.set_value(row[cf_id], row["clusters"], 1)

print(df_label.describe())
print(df_label.mean(axis=0))

# a = [sum(x) for x in np.transpose(df_label.values)]
pd.Series([sum(x) for x in np.transpose(df_label.values)]).describe()
# a = [sum(x) for x in df_label.values]
pd.Series([sum(x) for x in df_label.values]).describe()

#%%PREDOICTION

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)

treshold = 0.01
selected_Ks = [
    x for x in list(df_label.columns) if df_label.mean(axis=0).loc[x] > treshold
]
print(selected_Ks)

df_results = pd.DataFrame(
    columns=["rf_kg", "xgb_kg", "rf_baseline", "xgb_baseline"], index=selected_Ks
)
df_results.index.name = "Cluster"

for k in selected_Ks:
    X = df_emb.loc[df_emb["patient"] == 1].iloc[:, :DIM_SPACE].values
    y = df_label[k].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RSEED
    )

    # Feature Scaling
    # from sklearn.preprocessing import StandardScaler

    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    regressor = RandomForestClassifier(n_estimators=100, random_state=RSEED)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    # Probabilities for each class
    rf_probs = regressor.predict_proba(X_test)[:, 1]

    # print(confusion_matrix(y_test, y_pred))
    print("K = ", k)
    print("AUC: ", round(roc_auc_score(y_test, rf_probs), 2), "\n")
    # print(classification_report(y_test, y_pred), "\n")
    df_results.loc[k, "rf_kg"] = round(roc_auc_score(y_test, rf_probs), 2)

for k in selected_Ks:
    X = df_emb.loc[df_emb["patient"] == 1].iloc[:, :DIM_SPACE].values
    y = df_label[k].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RSEED
    )

    # Feature Scaling
    # from sklearn.preprocessing import StandardScaler

    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    regressor = XGBClassifier(random_state=RSEED)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    # Probabilities for each class
    rf_probs = regressor.predict_proba(X_test)[:, 1]

    # print(confusion_matrix(y_test, y_pred))
    print("K = ", k)
    print("AUC: ", round(roc_auc_score(y_test, rf_probs), 2), "\n")
    # print(classification_report(y_test, y_pred), "\n")
    df_results.loc[k, "xgb_kg"] = round(roc_auc_score(y_test, rf_probs), 2)

#%%Baseline

with open(
    os.path.join(
        _csv_path,
        os.pardir,
        "DEEP_LEARNING",
        "df_selected_patients_aslABC_filt30_IN_OUT.pickle",
    ),
    "rb",
) as f:
    df_baseline = pickle.load(f)

X = df_baseline.loc[df_label.index.tolist(), df_baseline.columns.tolist()[:-22]].values

print(selected_Ks)

for k in selected_Ks:
    y = df_label[k].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RSEED
    )

    # Feature Scaling
    # from sklearn.preprocessing import StandardScaler

    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    regressor = RandomForestClassifier(n_estimators=100, random_state=RSEED)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    # Probabilities for each class
    rf_probs = regressor.predict_proba(X_test)[:, 1]

    # print(confusion_matrix(y_test, y_pred))
    print("K = ", k)
    print("AUC: ", round(roc_auc_score(y_test, rf_probs), 2), "\n")
    # print(classification_report(y_test, y_pred), "\n")
    df_results.loc[k, "rf_baseline"] = round(roc_auc_score(y_test, rf_probs), 2)

for k in selected_Ks:
    y = df_label[k].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RSEED
    )

    # Feature Scaling
    # from sklearn.preprocessing import StandardScaler

    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    regressor = XGBClassifier(random_state=RSEED)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    # Probabilities for each class
    rf_probs = regressor.predict_proba(X_test)[:, 1]

    # print(confusion_matrix(y_test, y_pred))
    print("K = ", k)
    print("AUC: ", round(roc_auc_score(y_test, rf_probs), 2), "\n")
    # print(classification_report(y_test, y_pred), "\n")
    df_results.loc[k, "xgb_baseline"] = round(roc_auc_score(y_test, rf_probs), 2)

df_results.loc["MEAN"] = round(df_results.mean(), 3)

#%% Next Phase without baseline

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, ShuffleSplit

names_clf = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    # "Decision Tree",
    "Random Forest",
    "Neural Net",
    # "AdaBoost",
    "Naive Bayes",
    "QDA",
    "XGBoost",
]

classifiers = [
    KNeighborsClassifier(3, n_jobs=N_CORES),
    SVC(kernel="linear", C=0.025),
    SVC(),
    GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs=N_CORES),  # too slow
    # DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(n_estimators=100, n_jobs=N_CORES),
    MLPClassifier(alpha=1, max_iter=1000),
    # AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    XGBClassifier(n_jobs=N_CORES),
]

treshold = 0.01
selected_Ks = [
    x for x in list(df_label.columns) if df_label.mean(axis=0).loc[x] > treshold
]
print(selected_Ks)

df_results = pd.DataFrame(columns=names_clf, index=selected_Ks)
df_results.index.name = "Cluster"

X = df_emb.loc[df_emb["patient"] == 1].iloc[:, :DIM_SPACE].values

for k in selected_Ks:
    print("Cluster ", k)
    y = df_label[k].values
    # iterate over classifiers
    for name, clf in zip(names_clf, classifiers):
        print("\tclassifier: ", name)
        cv = ShuffleSplit(n_splits=HOW_MANY_SAMPLES, test_size=0.2, random_state=RSEED)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        df_results.loc[k, name] = (
            str(round(scores.mean(), 2)) + " \u00B1 " + str(round(scores.std(), 2))
        )
    print("\n")

for col in df_results.columns.tolist():
    print(col)
    scores = (
        df_results.loc[selected_Ks, col]
        .apply(lambda x: x.split(" \u00B1 ")[0])
        .map(float)
    )
    df_results.loc["MEAN", col] = (
        str(round(scores.mean(), 2)) + " \u00B1 " + str(round(scores.std(), 2))
    )

df_results.to_latex(os.path.join(_txt_path, "latex_prediction.txt"))

#%%OTHER per vincenzo
# df_prova = pd.DataFrame(triples)
# c = df_prova[~df_prova[1].str.contains("@has-")]
# c_1 = pd.DataFrame(c[1].value_counts())
# c_1 = c_1.reset_index().rename(columns={'index': 'Relations', 1: 'Number of triples'})
# c_1.to_latex(os.path.join(_txt_path, "table_relations_no_has.txt"), index=False)

