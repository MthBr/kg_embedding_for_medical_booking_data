from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt

import cup_utils

from cup_config import CSV_PATH,PICKLE_PATH,DATA_PATH,IMAGE_PATH

#%%

RSEED = 1000000

#%%

df_prestazioni = pd.read_csv(DATA_PATH + 'prestazioni_cup3.csv')
df_from_prestazioni_to_aggregated = pd.read_csv(DATA_PATH + 'df_from_prestazioni_to_aggregated.csv')
df_aggregated_prestazioni = pd.read_csv(DATA_PATH + 'df_aggregated_prestazioni.csv')

df_prestazioni_clustering_orig = cup_utils.load_pickle(PICKLE_PATH + 'prestazioni_clustering.pickle', verbose=1)

clustering_cols = ['NUM_TOT_VISITE','NUM_TOT_PAZIENTI','MEDIA_ANNO_NASCITA','VAR_ANNO_NASCITA','PERC_DONNE']

df_prestazioni_clustering_with10001 = df_prestazioni_clustering_orig[clustering_cols].copy()

df_prestazioni_clustering = df_prestazioni_clustering_with10001.iloc[1:].copy()

df_aggr_prestazioni_and_clustering = df_prestazioni_clustering_orig[['ID_UNIVOCO']].copy()
#%%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

choice = 1

if choice == 0:
    scaled_cols = ['NUM_TOT_VISITE','NUM_TOT_PAZIENTI','MEDIA_ANNO_NASCITA','VAR_ANNO_NASCITA']
    
    df_prestazioni_clustering[scaled_cols] = scaler.fit_transform(df_prestazioni_clustering[scaled_cols])
else:
    scaled_cols = ['NUM_TOT_VISITE','NUM_TOT_PAZIENTI', 'MEDIA_ANNO_NASCITA']
    
    # If X is a random variable, then for any c in R, mean(c*X) = c*mean(X), var(c*X) = c^2*var(X)
    # we apply this to scale X (= MEDIA_ANNO_NASCITA) with c = 1/(max - min)
    c = 1/(df_prestazioni_clustering['MEDIA_ANNO_NASCITA'].max() - df_prestazioni_clustering['MEDIA_ANNO_NASCITA'].min())
    
    df_prestazioni_clustering[scaled_cols] = scaler.fit_transform(df_prestazioni_clustering[scaled_cols])
    
    df_prestazioni_clustering['VAR_ANNO_NASCITA'] = df_prestazioni_clustering['VAR_ANNO_NASCITA']*c*c


#%% GLOBAL VARIABLES

cf_id = "sa_ass_cf"
prest_id = "indx_prestazione"
cup_date_id = "sa_data_pren"
time_limit_training = "{year}-{month}-{day}".format(year=2018, month=1, day=1)
time_limit_training = dt.datetime.strptime(time_limit_training, "%Y-%m-%d")
time_period_days = 365
time_limit_test = time_limit_training + dt.timedelta(days=time_period_days)
# RSEED = 0


# #%% IMPORT

# dictionay_name = f"{KEYSPACE_NAME}_dict_embeddings.pkl"
# dictionay_file = os.path.join(_pickle_path, dictionay_name)
# with open(dictionay_file, "rb") as open_file:
#     dict_emb_concepts = pickle.load(open_file)

# dictionay_name = f"{KEYSPACE_NAME}_dict_entities.pkl"
# dictionay_file = os.path.join(_dictionay_directory, dictionay_name)
# with open(dictionay_file, "rb") as open_file:
#     dict_entities = pickle.load(open_file)

# csv_name = f"triples_{KEYSPACE_NAME}.csv"
# csv_file = os.path.join(_3ples_directory, csv_name)
# with open(csv_file) as open_file:
#     triples = list(csv.reader(open_file, delimiter=","))

# csv_name = f"df_aslC.csv"
# csv_file = os.path.join(_csv_path, csv_name)
# df_cup = pd.read_csv(csv_file, index_col=0)
# # transform to datetime
# for date_col in ["sa_data_ins", "sa_data_prescr", "sa_data_app", "sa_data_pren"]:
#     df_cup[date_col] = pd.to_datetime(df_cup[date_col], errors="coerce")

# csv_name = "prestazioni.csv"
# csv_file = os.path.join(_cup_datasets_path, csv_name)
# df_prest = pd.read_csv(csv_file, sep=";", index_col=0)

# csv_name = "prestazioni_to_branche_cup3.csv"
# csv_file = os.path.join(_cup_datasets_path, csv_name)
# df_prest_to_branche = pd.read_csv(csv_file, sep=",")

# df_prest_to_branche = (
#     df_prest_to_branche.fillna("")
#     .groupby("id_prestazione")["id_branca"]
#     .apply(list)
#     .reset_index(name="list_branca")
# )
# df_prest_to_branche = df_prest_to_branche.append(
#     {"id_prestazione": 99999999, "list_branca": []}, ignore_index=True
# )
# df_prest_to_branche = df_prest_to_branche.set_index("id_prestazione")[
#     "list_branca"
# ].to_dict()

# #%% FUNCTIONS

# # from id to identification
# def from_id_to_identification(triples, dict_entities, entity_id):
#     dict_map = {}
#     for row in triples:
#         if row[1] == f"@has-{entity_id}":
#             dict_map[row[0]] = dict_entities[row[2]]["value"]
#     return dict_map


# def create_embedding_dataframe(
#     dict_emb_concepts, entity_name, with_mapping=False, dict_map=None, index_col=None
# ):
#     if with_mapping:
#         # create embedding dict with identification
#         dict_new = {}
#         for key, value in dict_emb_concepts[entity_name].items():
#             dict_new[dict_map[key]] = value

#         # create dataFrame of embedding
#         df_emb = pd.DataFrame.from_dict(dict_new, orient="index")
#         df_emb.index.name = index_col
#     else:
#         # create dataFrame of embedding
#         df_emb = pd.DataFrame.from_dict(dict_emb_concepts[entity_name], orient="index")
#         df_emb.index.name = entity_name

#     return df_emb


# #%% MAIN

# # Patients
# dict_map_patients = from_id_to_identification(triples, dict_entities, entity_id_patient)
# df_emb_patients = create_embedding_dataframe(
#     dict_emb_concepts,
#     entity_name_patient,
#     with_mapping=True,
#     dict_map=dict_map_patients,
#     index_col=cf_id,
# )

# # Provisions
# dict_map_provisions = from_id_to_identification(
#     triples, dict_entities, entity_id_provision
# )
# df_emb_provisions = create_embedding_dataframe(
#     dict_emb_concepts,
#     entity_name_provision,
#     with_mapping=True,
#     dict_map=dict_map_provisions,
#     index_col=prest_id,
# )


# # create an empty dataFrame of labels/prescriptions
# df_label = pd.DataFrame(0, index=df_emb_patients.index, columns=df_emb_provisions.index)

# # fill dataFrame with 1 if there is at least one record
# for index, row in df_cup.iterrows():
#     if time_limit_training <= row[cup_date_id] < time_limit_test:
#         df_label[row[prest_id]][str(row[cf_id])] = 1

# # a = [sum(x) for x in df_label.values]

#%% Kmeans

from sklearn.cluster import KMeans

K_max = 10


Num_executions = 10

distortions = []
for indx in range(Num_executions):
    # calculate distortion for a range of number of cluster
    distortions = distortions + [[]]
    for i in range(1, K_max + 1):
        km = KMeans(
            n_clusters=i, init="random", n_init=10, max_iter=300, tol=1e-04#, random_state=RSEED
        )
        km.fit(df_prestazioni_clustering)
        distortions[indx].append(km.inertia_)
distortions = list(np.mean(distortions, axis=0))

# plot
plt.plot(range(1, K_max + 1), distortions, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Distortion")
plt.show()

Ktests = [7]#3,5,7]
for K in Ktests:
    km = KMeans(
        n_clusters=K, init="random", n_init=10, max_iter=300, tol=1e-04, random_state=RSEED
    ).fit(df_prestazioni_clustering)
    
    newcol = 'clusters_{}'.format(K+1)
    
    s_cluster_0 = pd.Series(0, index=['10001'])
    
    s_clusters_kmeans = pd.Series(km.labels_ + 1, index=df_prestazioni_clustering.index)
    
    s_clusters_final = pd.concat([s_cluster_0, s_clusters_kmeans])
    
    df_aggr_prestazioni_and_clustering[newcol] = s_clusters_final

df_aggr_prestazioni_and_clustering['DESCRIZIONE'] = df_aggr_prestazioni_and_clustering.ID_UNIVOCO.apply(lambda indx:
                                                                                          df_aggregated_prestazioni[df_aggregated_prestazioni.id == (indx-10000)].descrizione.values[0] if indx > 10000
                                                                                          else df_prestazioni[df_prestazioni.id == indx].descrizione.values[0]
                                                                                              )

df_aggr_prestazioni_and_clustering = df_aggr_prestazioni_and_clustering[['ID_UNIVOCO', 'DESCRIZIONE', 'clusters_8']].sort_values('clusters_8')

df_prestazioni_and_clustering = df_prestazioni[['id', 'descrizione']].copy()

df_prestazioni_and_clustering = df_prestazioni_and_clustering.set_index(df_prestazioni_and_clustering['id'].values, drop=False)

df_prestazioni_and_clustering['id_aggregato'] = 0
df_prestazioni_and_clustering['descrizione_aggregato'] = 0

for indx in df_prestazioni['id']:
    if len(df_from_prestazioni_to_aggregated[df_from_prestazioni_to_aggregated.id == indx]) > 0:
        id_aggr = df_from_prestazioni_to_aggregated[df_from_prestazioni_to_aggregated.id == indx].id_aggregated.values[0]
        df_prestazioni_and_clustering.loc[indx,'id_aggregato'] = id_aggr+10000
        df_prestazioni_and_clustering.loc[indx,'descrizione_aggregato'] = df_aggregated_prestazioni[df_aggregated_prestazioni.id == id_aggr].descrizione.values[0]
    else:
        if len(df_aggr_prestazioni_and_clustering[df_aggr_prestazioni_and_clustering.ID_UNIVOCO == indx]) > 0:
            df_prestazioni_and_clustering.loc[indx, 'id_aggregato'] = df_aggr_prestazioni_and_clustering[df_aggr_prestazioni_and_clustering.ID_UNIVOCO == indx].ID_UNIVOCO.values[0]
            df_prestazioni_and_clustering.loc[indx, 'descrizione_aggregato'] = df_prestazioni_and_clustering.loc[indx,'descrizione']

df_prestazioni_and_clustering = df_prestazioni_and_clustering[df_prestazioni_and_clustering.id_aggregato > 0]

df_prestazioni_and_clustering['cluster'] = df_prestazioni_and_clustering['id_aggregato'].apply(lambda indx: df_aggr_prestazioni_and_clustering.loc[str(indx), 'clusters_8'])


#%%
cup_utils.save_on_pickle(df_prestazioni_and_clustering, CSV_PATH + 'DEEP_LEARNING/df_prestazioni_and_clustering_8.pickle')
df_prestazioni_and_clustering.to_csv(CSV_PATH + 'DEEP_LEARNING/df_prestazioni_and_clustering_8.csv')


#%%TSne and PCA

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import time

sizex = 10
sizey = 10

time_start = time.time()
tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, random_state=RSEED, verbose=1)
#tsne_results = tsne.fit_transform(df_prestazioni_clustering_with10001)
tsne_results = tsne.fit_transform(df_prestazioni_clustering)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_subset = df_prestazioni_clustering.copy()
#df_subset = df_prestazioni_clustering_with10001.copy()
df_subset["y"] = km.labels_
#df_subset["y"] = np.array([0,*(km.labels_ + 1)])
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(sizex,sizey))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", K),#K+1),
    data=df_subset,
    legend="full",
    alpha=0.9,
)

#%%
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_prestazioni_clustering)
df_subset['pca-one'] = pca_result[:,0]
df_subset['pca-two'] = pca_result[:,1] 
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

plt.figure(figsize=(sizex,sizey))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", K),
    data=df_subset,
    legend="full",
    alpha=0.9,
)

#%%PREDOICTION
df_prestazioni_and_clustering = cup_utils.load_pickle(CSV_PATH + 'DEEP_LEARNING/df_prestazioni_and_clustering_8.pickle')

df_patients = cup_utils.load_pickle(CSV_PATH + 'DEEP_LEARNING/df_selected_patients_aslABC_filt30_noid0.pickle', verbose=1)

K = df_prestazioni_and_clustering.cluster.nunique()

prest_cols = [x for x in df_patients.columns if x.startswith('prest_')]

prest_ids_no2018_cols = [x for x in prest_cols if x.split('_')[2] < '2018']

df_emb_patients = df_patients[prest_ids_no2018_cols]


# create an empty dataFrame of labels/prescriptions
df_label = pd.DataFrame(0, index=df_emb_patients.index, columns=list(np.arange(K)))

df_cup = cup_utils.load_pickle(CSV_PATH + 'DEEP_LEARNING/df_cup_filt.pickle', verbose=1)

df_cup = df_cup[df_cup.sa_ass_cf.isin(df_emb_patients.index)]

BAD_INDEX = 99999999
# Remove bad prestazioni
df_cup = df_cup[df_cup.indx_prestazione != BAD_INDEX]

# Remove weird no prestazioni
df_cup = df_cup[df_cup.sa_num_prestazioni > 0]

# Remove negative gg_attesa
df_cup = df_cup[df_cup.sa_gg_attesa >= 0]

# Exclude prestazioni outside the clusters
df_cup = df_cup[df_cup.indx_prestazione.isin(df_prestazioni_and_clustering.id.values)]

df_cup = df_cup[(df_cup[cup_date_id] >= time_limit_training) & (df_cup[cup_date_id] < time_limit_test)]

df_cup = df_cup[[cf_id, 'indx_prestazione']]

df_cup['cluster'] = df_cup.indx_prestazione.apply(lambda indx: df_prestazioni_and_clustering[df_prestazioni_and_clustering.id == indx].cluster.values[0])

print("Filling labels...")
for col in df_label.columns:
    df_subcup = df_cup[df_cup.cluster == col]
    
    df_label[col] = df_label.index.to_series().apply(lambda indx: 1 if len(df_subcup[df_subcup.sa_ass_cf == indx]) > 0 else 0)

print(df_label.describe())

a = [sum(x) for x in np.transpose(df_label.values)]
pd.Series(a).describe()
a = [sum(x) for x in df_label.values]
pd.Series(a).describe()

from sklearn.model_selection import train_test_split

for k in range(0, K):
    X = df_emb_patients.values
    y = df_label[k].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RSEED)

    # Feature Scaling
    # from sklearn.preprocessing import StandardScaler

    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    from sklearn.ensemble import RandomForestClassifier

    regressor = RandomForestClassifier(n_estimators=200, random_state=RSEED)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
    # Probabilities for each class
    rf_probs = regressor.predict_proba(X_test)[:, 1]

    print("K = ", k)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("AUC: ", round(roc_auc_score(y_test, rf_probs), 2))

for k in range(0, K):
    X = df_emb_patients.values
    y = df_label[k].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RSEED)

    # Feature Scaling
    # from sklearn.preprocessing import StandardScaler

    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    from xgboost import XGBClassifier
    
    regressor = XGBClassifier(random_state=RSEED)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
    # Probabilities for each class
    rf_probs = regressor.predict_proba(X_test)[:, 1]

    print("K = ", k)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("AUC: ", round(roc_auc_score(y_test, rf_probs), 2))
