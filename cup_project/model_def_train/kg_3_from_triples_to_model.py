# import numpy as np
import os
# %%PARAMS
from cup_project.config import data_dir, reportings_dir, log_dir


import ampligraph
print(ampligraph.__version__)

KEYSPACE_NAME = "cup_1747_1"  # cup_100_1 cup_200_2  cup_1000_1
# KEYSPACE_NAME = "cup_1"

# %% Dirs variables

_3ples_directory = data_dir / "processed" / "3ples"
_dictionay_directory = data_dir / "processed" / "dicts"

_model_path = data_dir / 'models'

_embedding_directory = data_dir / "model_embeddings"

# %% main method definition
def embed_triples(triples):

    from numpy import array

    X = array(triples)

    model = best_model(X)  # split_and_train(X)  best_model(X)

    return model


# %%
def X_dict_trn_tst_vl(X):
    from ampligraph.evaluation import train_test_split_no_unseen
    #X_train, X_valid
    X_train_test, X_valid = train_test_split_no_unseen(
        X, test_size=round(len(X) * 5 / 100), seed=0, allow_duplication=True
    )  # len(X)*3/100 len(X)*4/100   testlen(triples)/40  len(X)/31   40,39,31,30  test_size=100, seed=0, allow_duplication=False

    X_train, X_test = train_test_split_no_unseen(
        X_train_test, test_size=round(len(X) * 10 / 100), allow_duplication=True)

    print("Train set size: ", X_train.shape)
    print("Test set size: ", X_test.shape)
    print("Validation set size: ", X_valid.shape)

    X_dict = dict()
    X_dict['train'] = X_train
    X_dict['valid'] = X_valid
    X_dict['test'] = X_test

    return X_dict


def best_model(X, model_path=_model_path):

    X_dict = X_dict_trn_tst_vl(X)

    print(f"Using  {KEYSPACE_NAME}")

    from ampligraph.evaluation import select_best_model_ranking
    from ampligraph.latent_features import ComplEx, HolE, TransE, ConvE
    #import numpy as np

    model_class = ConvE
    #model_class = TransE
    # model_class = ComplEx  #http://docs.ampligraph.org/en/latest/generated/ampligraph.latent_features.ComplEx.html

    #model_file_name = f"best_model_{KEYSPACE_NAME}{len(X)}_ComplEx_5_10_grid.pkl"
    model_file_name = f"best_model_{KEYSPACE_NAME}{len(X)}_ConvE_5_10_grid.pkl"
    #model_file_name = "best_model.pkl"

    param_grid_basic = {
        "batches_count": [8192, 16384],  # 32 64 128 256 512  50 100 200 NOO:512, 1024
        "seed": [0, 1, 2],
        "epochs": [4000, 500],  # , 4000 , 500
        "k": [100, 150, 200],
        "regularizer": ["LP", None],  # , None
        "regularizer_params":  {
            "p": [1, 3],
            # 1e-3, 1e-4, 1e-5  extra: 1e-1, 1e-2
            "lambda": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        },
        "optimizer": ["adam"],  # "adagrad", "adam", "sgd"
        "optimizer_params": {
            # [5e-4, 5e-5]  lambda: np.random.uniform(0.0001, 0.1)
            "lr": [1e-1, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
        },
        "verbose": True
    }
    param_grid_ComplEx = {
        "eta": [5, 10, 15, 20],
        "loss": ["pairwise", "nll", "multiclass_nll"],  # bce
        "loss_params": {
            "margin": [0.5, 1, 2]
        },
        "embedding_model_params": {
            "corrupt_sides": ['s+o', 's', 'o'],  # 's,o'
            # generate corruption using all entities during training
            "negative_corruption_entities": ["all", "batch"]
        }
    }
    param_grid_ConvE = {
        "loss": "bce",
        "loss_params": {
            "lr": [0.1, 0.001, 0.01],
            "label_smoothing":[1, 10, 100, 0.1, 0.001, 0.0001]
        },
        "embedding_model_params": {
            #"dropout_embed": [0.1],  # 0.0, 0.1, 0.2, 1, 0.01
            #"dropout_conv": [0.1],  #Feature map
            #"dropout_dense": [0.5],  #Hidden Layer
            "corrupt_side": ['o'],  # 's,o'
            'entities_size': 0
        },
        # "early_stopping_params": {
        #     "corrupt_side": 'o',  # 's,o'
        #     'entities_size': 0
        # },
        "low_memory": True
    }
    param_grid = {**param_grid_basic, **param_grid_ConvE}
    # Train the model on all possibile combinations of hyperparameters.
    # Models are validated on the validation set.
    # It returnes a model re-trained on training and validation sets.
    best_model, best_params, best_mrr_train, ranks_test, mrr_test, experimental_history = select_best_model_ranking(
        model_class,  # Class handle of the model to be used
        X_dict['train'], X_dict['valid'], X_dict['test'],  # Dataset
        param_grid,  # Parameter grid
        max_combinations=64,  # if not None, will be random
        # Use filtered set for eval - raw MRR with False - True, will use the entire input dataset X to compute filtered MRR (default: True).
        use_filter=False,
        corrupt_side='o',
        verbose=True,  # Log all the model hyperparams and evaluation stats
        early_stopping=False
    )

    model_file = os.path.join(model_path, model_file_name)
    from ampligraph.latent_features import save_model

    save_model(best_model, model_file)

    import pickle
    dict_embexperimental_history = KEYSPACE_NAME + "_dict_experimental_history.pkl"
    dict_file = os.path.join(_dictionay_directory,
                             dict_embexperimental_history)
    output = open(dict_file, "wb")
    pickle.dump(experimental_history, output)
    output.close()

    return best_model, best_params, best_mrr_train, ranks_test, mrr_test, experimental_history


# %%


if __name__ == "__main__":
    csv_directory = _3ples_directory
    csv_file = "triples_" + KEYSPACE_NAME + ".csv"

    from ampligraph.datasets import load_from_csv

    data = load_from_csv(csv_directory, csv_file, sep=",")

    #    data = pd.read_csv(csv_directory + '/' + csv_file, sep=',', header=None, names=None, dtype=str)
    #    data = data.drop_duplicates()
    #    data = data.values

    embed_triples(data)
