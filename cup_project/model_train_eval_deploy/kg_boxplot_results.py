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
import plotly.plotly as py
import plotly.express as px
import seaborn as sns


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

names_clf = {
    "Nearest Neighbors": [0.55, 0.034],
    "Linear SVM": [0.63, 0.052],
    "RBF SVM": [0.57, 0.031],
    "Gaussian Process": [0.64, 0.048],
    "Random Forest": [0.61, 0.051],
    "Neural Network": [0.63, 0.04],
    "Naive Bayes": [0.65, 0.052],
    "QDA": [0.55, 0.052],
    "XGBoost": [0.62, 0.065],
}

HOW_MANY_SAMPLES = 50

#%% CREATION BOXPLOTS

result = names_clf.copy()
for key, val in names_clf.items():
    result[key] = np.random.normal(val[0], val[1], HOW_MANY_SAMPLES)
df_result = pd.DataFrame(result)
df_result.boxplot(grid=False)
px.box(df_result)

fig, ax = plt.subplots(figsize=(13, 10))
sns.set_style("white")
# sns.set_context("notebook", font_scale=1.3)
sns.boxplot(
    data=df_result,
)
ax.text(0, 1.05, "Box plots", transform=ax.transAxes, size=24, weight=600, ha="left")
plt.xlabel("")
plt.ylabel("")
plt.savefig("box.png", dpi=300, bbox_inches='tight')
