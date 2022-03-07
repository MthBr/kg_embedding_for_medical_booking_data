#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:46:07 2019


@author: 
"""

import logging
import os
import sys
import numpy as np
import pickle
import csv
import datetime as dt
import pandas as pd
import matplotlib.pylab as plt

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
    os.path.join(
        working_dir, os.pardir, "csv", "random_1747_1_True"
    )  # random_1747_1_True random_100_1_False
)
_cup_datasets_path = os.path.abspath(
    os.path.join(working_dir, os.pardir, "cup_datasets")
)
_txt_path = os.path.abspath(os.path.join(working_dir, os.pardir, "txt"))
os.makedirs(_txt_path, exist_ok=True)
_image_path = os.path.abspath(os.path.join(working_dir, os.pardir, "images"))
os.makedirs(_image_path, exist_ok=True)

#%% GLOBAL VARIABLES

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
    "sa_branca_id": "medical-branch-id",
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
entities_names = [
    "medical-branch",
    "appointment-provider",
    "practitioner",
    "health-service-provision",
    "patient",
    "booking-staff",
]
relations_names = ["referral", "reservation", "health-care", "provision"]
cols_first_table = [
    "encrypted-nin-id",
    "gender",
    "priority-code",
    "nuts-istat-code",
    "practitioner-id",
    "booking-agent-id",
    "updating-booking-agent-id",
]
cols_second_table = [
    "referral-centre-id",
    "number-of-health-services",
    "local-health-department-id",
    "refined-health-service-id",
]
dict_colors = {
    "appointment-provider": "#66C2A5",         
    "booking-staff": "#FC8D62",
    "health-service-provision": "#8DA0CB",        
    "medical-branch": "#E78AC3",  
    "patient": "#A6D854",
    "practitioner": "#FFD92F",
}
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
del date_col

csv_name = "prestazioni.csv"
csv_file = os.path.join(_cup_datasets_path, csv_name)
df_prest = pd.read_csv(csv_file, sep=";", index_col=0)
dict_prest = df_prest["descrizione"].to_dict()

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
df_prest_to_branche = df_prest_to_branche.set_index("id_prestazione")[
    "list_branca"
].to_dict()

#%% FUNCTIONS

# from id to identification
def from_id_to_identification(triples, dict_entities, entity_id):
    dict_map = {}
    for row in triples:
        if row[1] == f"@has-{entity_id}":
            dict_map[row[0]] = dict_entities[row[2]]["value"]
    return dict_map


def create_embedding_dataframe(
    dict_emb_concepts,
    entity_name,
    new_column=False,
    with_mapping=False,
    dict_map=None,
    index_col=None,
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

    if new_column:
        df_emb["entity"] = [entity_name] * len(df_emb)
        df_emb[entity_name] = np.ones((len(df_emb),), dtype=int)

    return df_emb


def plot_groups(df, which="TSNE", groups_col=None, groups_num=1, save=False, save_path=None, colors=None):
    import seaborn as sns
    import warnings

    if which == "TSNE":
        from sklearn.manifold import TSNE
        import time

        time_start = time.time()
        tsne = TSNE(
            n_components=2,
            perplexity=30.0,
            early_exaggeration=12.0,
            learning_rate=200.0,
            n_iter=1000,
            verbose=1,
        )
        results = tsne.fit_transform(df)[:, 0:2]

        print("t-SNE done! Time elapsed: {} seconds".format(time.time() - time_start))

    elif which == "PCA":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        results = pca.fit_transform(df)[:, 0:2]
        print(
            "Explained variation per principal component: {}".format(
                pca.explained_variance_ratio_
            )
        )
    else:
        warnings.warn("Select TSNE or PCA")

    df["labels"] = groups_col
    df["one"] = results[:, 0]
    df["two"] = results[:, 1]
    
    if colors is None:
        colors = sns.color_palette("hls", groups_num)
        
    plt.figure(figsize=(16, groups_num)).suptitle(which, fontsize=18)
    sns.scatterplot(
        x="one",
        y="two",
        hue="labels",
        palette=colors,
        data=df,
        legend="full",
        alpha=0.9,
    )
    if save:
        plt.savefig(os.path.join(save_path, which + ".png"), dpi=300)


def create_df_description(df, which, list_col, only_star=False):
    df_description = pd.DataFrame(columns=[dict_col_id[x] for x in list_col])

    if which == "category_col":
        for col in list_col:
            val_count = []
            if col in dict_col_name.keys():
                # if dict_col_name[col] in set(df_emb["entity"][df_emb["clusters"] == selected_K]):
                id_in_cluster = (
                    df_emb[
                        (df_emb["clusters"] == selected_K)
                        & (df_emb["entity"] == dict_col_name[col])
                    ]
                    .index.to_series()
                    .map(dict_map)
                    .tolist()
                )
                if only_star:
                    if id_in_cluster:  # if it's not empty
                        if col == "indx_prestazione":
                            val_count = [
                                (r"$\star$", dict_prest[key], value)
                                for key, value in df[col]
                                .value_counts()
                                .to_dict()
                                .items()
                                if key in id_in_cluster
                            ]
                        elif type(id_in_cluster[0]) is int:
                            val_count = [
                                (r"$\star$", key, value)
                                for key, value in df[col]
                                .value_counts()
                                .to_dict()
                                .items()
                                if key in id_in_cluster
                            ]
                        else:
                            val_count = [
                                (r"$\star$", key, value)
                                for key, value in df[col]
                                .value_counts()
                                .to_dict()
                                .items()
                                if str(key) in id_in_cluster
                            ]
                else:
                    if id_in_cluster:  # if it's not empty
                        if col == "indx_prestazione":
                            val_count = [
                                (r"$\star$", dict_prest[key], value)
                                if key in id_in_cluster
                                else (dict_prest[key], value)
                                for key, value in df[col]
                                .value_counts()
                                .to_dict()
                                .items()
                            ]
                        elif type(id_in_cluster[0]) is int:
                            val_count = [
                                (r"$\star$", key, value)
                                if key in id_in_cluster
                                else (key, value)
                                for key, value in df[col]
                                .value_counts()
                                .to_dict()
                                .items()
                            ]
                        else:
                            val_count = [
                                (r"$\star$", key, value)
                                if str(key) in id_in_cluster
                                else (key, value)
                                for key, value in df[col]
                                .value_counts()
                                .to_dict()
                                .items()
                            ]
                    else:
                        if col == "indx_prestazione":
                            val_count = [
                                (dict_prest[key], value)
                                for key, value in df[col].value_counts().to_dict().items()
                            ]
                        else:
                            val_count = [
                                (key, value)
                                for key, value in df[col].value_counts().to_dict().items()
                            ]
            else:
                val_count = list(df[col].value_counts().to_dict().items())

            df_description[dict_col_id[col]] = pd.Series(val_count)
        df_description = df_description.fillna("")

    elif which == "number_col":
        for col in list_col:
            df_description[dict_col_id[col]] = pd.Series(df[col].describe().round(1))
        df_description = df_description.drop("count")

    elif which == "date_col":

        for col in list_col:
            if pd.isnull(df[col].mean()):
                dict_val = {
                    "mean": df[col].mean(),
                    "std": "",
                    "min": df[col].min(),
                    "25%": df[col].quantile(0.25),
                    "50%": df[col].quantile(0.5),
                    "75%": df[col].quantile(0.75),
                    "max": df[col].max(),
                }
            else:
                dict_val = {
                    "mean": df[col].mean().strftime("%Y-%m-%d"),
                    "std": "",
                    "min": df[col].min().strftime("%Y-%m-%d"),
                    "25%": df[col].quantile(0.25).strftime("%Y-%m-%d"),
                    "50%": df[col].quantile(0.5).strftime("%Y-%m-%d"),
                    "75%": df[col].quantile(0.75).strftime("%Y-%m-%d"),
                    "max": df[col].max().strftime("%Y-%m-%d"),
                }
            df_description[dict_col_id[col]] = pd.DataFrame.from_dict(
                dict_val, orient="index", columns=[dict_col_id[col]]
            )[dict_col_id[col]]

    return df_description

#%% MAIN

names = entities_names
# names = relations_names
# names = entities_names + relations_names

# create whole dataFrame of embedding
df_emb = pd.DataFrame()
for entity_name in names:
    df_emb = pd.concat(
        [
            df_emb,
            create_embedding_dataframe(dict_emb_concepts, entity_name, new_column=True),
        ],
        sort=True,
    )
df_emb = df_emb.fillna(0)
df_emb[names] = df_emb[names].astype(int)

# Save to pickle
import pickle

file_emb = KEYSPACE_NAME + "_emb_concepts.pkl"
pkl_file = os.path.join(_pickle_path, file_emb)
output = open(pkl_file, "wb")
pickle.dump(df_emb, output)
output.close()

df_emb_copy = df_emb.copy()

#%%TSne and PCA

DIM_SPACE = len(df_emb.columns) - len(names) - 1

plot_groups(
    df_emb.iloc[:, :DIM_SPACE],
    which="TSNE",
    groups_col=df_emb["entity"],
    groups_num=len(names),
    save=True,
    save_path=_image_path,
    colors=dict_colors,
)

plot_groups(
    df_emb.iloc[:, :DIM_SPACE],
    which="PCA",
    groups_col=df_emb["entity"],
    groups_num=len(names),
    save=True,
    save_path=_image_path,
    colors=dict_colors,
)

#%%DBSCAN
def dbscan_search():
    import random
    import time
    import datetime
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score

    random.seed(RSEED)
    N_MAX = 10000
    cols_dbscan = [
        "eps",
        "min_samples",
        "metric",
        "n_clusters",
        "n_outliers",
        "avg_silhouette",
    ]
    df_dbscan = []
    clustering_time = []
    j = 0
    for i in range(N_MAX):
        starting_time = time.time()
        rnd_eps = round(random.uniform(1e-3, 100), 3)
        rnd_min_samples = random.randint(2, round(len(df_emb) / 5))
        rnd_metric = random.choice(
            ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"]
        )

        clustering = DBSCAN(
            eps=rnd_eps, min_samples=rnd_min_samples, metric=rnd_metric
        ).fit(df_emb.iloc[:, :DIM_SPACE])

        labels = clustering.labels_
        if len(set(labels)) > 1:
            j += 1
            print(f"\tFound the {j}th clustering over {i} iterations\n")
            df_dbscan.append(
                {
                    "eps": rnd_eps,
                    "min_samples": rnd_min_samples,
                    "metric": rnd_metric,
                    "n_clusters": len(set(labels)),
                    "n_outliers": list(labels).count(-1),
                    "avg_silhouette": silhouette_score(
                        df_emb.iloc[:, :DIM_SPACE], labels
                    ),
                }
            )
        finish_time = time.time()
        clustering_time.append(finish_time - starting_time)
        print(
            "at "
            + str(datetime.datetime.now())
            + "\n"
            + "\tClustering made in "
            + str(datetime.timedelta(seconds=(finish_time - starting_time)))
            + "\n"
            + "\tNumber of clustering found "
            + str(j)
            + "\n"
            + "\tNumber of clustering made "
            + str(i + 1)
            + "\n"
            + "\tNumber of clustering to make "
            + str(N_MAX - 1 - i)
            + "\n"
            + "\t Remaining clustering time "
            + str(
                datetime.timedelta(
                    seconds=round(np.mean(clustering_time) * (N_MAX - 1 - i))
                )
            )
            + " seconds"
            + "\n"
        )

    df_dbscan = pd.DataFrame(df_dbscan, columns=cols_dbscan).sort_values(
        by=["avg_silhouette"], ascending=False
    )
    return df_dbscan


# df_emb["DBSCAN"] = clustering.labels_

#%% Describe Phase 3

from tabulate import tabulate

# import plotly.graph_objects as go

with open(os.path.join(_csv_path, KEYSPACE_NAME + "_clusters.csv")) as f:
    clustering = pd.read_csv(f, index_col=0)
df_emb["clusters"] = clustering

max_K = int(clustering.max())
selected_Ks = list(np.arange(1, max_K + 1))  # list(np.arange(1, 17 + 1))

try:
    os.remove(os.path.join(_txt_path, "latex_clustering_" + str(max_K) + ".txt"))
except OSError:
    pass

# dict of mapping
dict_map = {}
for entity_name in names:
    dict_map.update(
        from_id_to_identification(triples, dict_entities, dict_name_id[entity_name])
    )

# df_cup_restr_time
df_cup_restr_time = df_cup[df_cup[cup_date_id] < time_limit_training].copy()
# df_cup_restr_time[dict_table["category_col"]] = df_cup_restr_time[dict_table["category_col"]].astype('category')

for selected_K in selected_Ks:
    df_clustering = df_emb["entity"][df_emb["clusters"] == selected_K]
    for entity_name in set(df_clustering):
        list_id = list(df_clustering[df_clustering == entity_name].index)
        list_id = [dict_map[x] for x in list_id]
        df_cup_entity = df_cup_restr_time[
            df_cup_restr_time[dict_id_col[dict_name_id[entity_name]]].isin(list_id)
        ]
        df_description_cat = create_df_description(
            df_cup_entity,
            "category_col",
            [
                x
                for x in dict_table["category_col"]
                if x != dict_id_col[dict_name_id[entity_name]]
            ],
            only_star=True, # restricted tables
        )
        df_description_num = pd.concat(
            [
                create_df_description(
                    df_cup_entity, "number_col", dict_table["number_col"]
                ),
                create_df_description(
                    df_cup_entity, "date_col", dict_table["date_col"]
                ),
            ],
            axis=1,
        )
        with open(
            os.path.join(_txt_path, "latex_clustering_" + str(max_K) + ".txt"),
            "a",
            encoding="utf-8",
        ) as outputfile:
            outputfile.writelines(
                [
                    "Cluster " + str(selected_K) + " - \n",
                    entity_name.upper() + "\n",
                    r"""\newline
                    """
                    + "\n",
                    r"""\begin{table}[H]
	                    \scalebox{0.7}{"""
                    + "\n",
                    df_description_num.to_latex(),
                    r"""}
                    \end{table}"""
                    + "\n",  # \caption{}
                    r"""\begin{table}[H]
	                    \scalebox{0.7}{"""
                    + "\n",
                    df_description_cat[
                        [x for x in cols_first_table if x in df_description_cat.columns]
                    ]
                    .head(7)
                    .to_latex(index=False),
                    r"""}
                    \end{table}"""
                    + "\n",  # \caption{}
                    r"""\begin{table}[H]
	                    \scalebox{0.7}{"""
                    + "\n",
                    df_description_cat[
                        [
                            x
                            for x in cols_second_table
                            if x in df_description_cat.columns
                        ]
                    ]
                    .head(7)
                    .to_latex(index=False),
                    r"""}
                    \end{table}"""
                    + "\n",  # \caption{}
                    r"""
                    \vspace*{0.5cm}
                    """
                    + "\n",
                ]
            )
        with open(
            os.path.join(_txt_path, entity_name + "_cat_" + str(selected_K) + ".txt"),
            "w",
            encoding="utf-8",
        ) as outputfile:
            outputfile.write(
                tabulate(
                    df_description_cat.head(7),
                    headers="keys",
                    tablefmt="psql",
                    showindex=False,
                )
            )
        with open(
            os.path.join(_txt_path, entity_name + "_num_" + str(selected_K) + ".txt"),
            "w",
            encoding="utf-8",
        ) as outputfile:
            outputfile.write(
                tabulate(df_description_num, headers="keys", tablefmt="psql")
            )
