# import datetime
import logging
import os
#import sys

_local_dir=False

#%% Dirs variables
file_dir = os.path.dirname(__file__)
#sys.path.append(file_dir)
if _local_dir:
    working_dir = os.path.join(file_dir, os.pardir)
else:
    working_dir = os.path.abspath(os.path.join(file_dir, f"../../../../Pycodes/cup_kg/"))
_3ples_directory = os.path.abspath(os.path.join(working_dir, "3ples"))
_model_path = os.path.abspath(os.path.join(working_dir, 'models'))
_dictionay_directory = os.path.abspath(os.path.join(working_dir, "dicts"))
_pickle_path = os.path.abspath(os.path.join(working_dir, os.pardir, "pickle"))


os.makedirs(_model_path, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

KEYSPACE_NAME = "cup_1747_1" # cup_100_1 cup_200_2  cup_1000_1
# KEYSPACE_NAME = "cup_1_new"
# KEYSPACE_NAME = "football_results"
# KEYSPACE_NAME = "phone_calls"
# KEYSPACE_NAME = "GoT_false"
# KEYSPACE_NAME = "GoT"


def main():
    
    dict_emb_concepts = get_dict(KEYSPACE_NAME, "embeddings")
    
    dict_entities = get_dict(KEYSPACE_NAME, "concepts") # ex entities

    from ampligraph.datasets import load_from_csv
    csv_name = f"triples_{KEYSPACE_NAME}.csv"
    triples = load_from_csv(_3ples_directory, csv_name, sep=",")
    
    
    entity_id_patient = "encrypted-nin-id"
    dict_map_patients = from_id_to_identification(triples, dict_entities, entity_id_patient)
    entity_name_patient = "patient"
    df_emb_patients = create_embedding_dataframe(
        dict_emb_concepts,
        entity_name_patient,
        with_mapping=True,
        dict_map=dict_map_patients,
        index_col=cf_id,
    )
    
    # Provisions
    entity_id_provision = "refined-health-service-id"
    dict_map_provisions = from_id_to_identification(
        triples, dict_entities, entity_id_provision
    )
    entity_name_provision = "health-service-provision"
    prest_id = "indx_prestazione"
    df_emb_provisions = create_embedding_dataframe(
        dict_emb_concepts,
        entity_name_provision,
        with_mapping=True,
        dict_map=dict_map_provisions,
        index_col=prest_id,
    )
        
    
    
    
    
    


#%% FUNCTIONS
def get_dict(keyspace=KEYSPACE_NAME, kind="concepts"):
    import pickle
    # load dict labels
    dictionay_name = f"{keyspace}_dict_{kind}.pkl"
    dictionay_file = os.path.join(_dictionay_directory, dictionay_name)
    print(dictionay_file)
    if os.path.isfile(dictionay_file):
        pkl_file = open(dictionay_file, "rb")
        dict_entities = pickle.load(pkl_file)
        pkl_file.close()
    else:
        dict_entities = None

    return dict_entities

def get_emb(keyspace=KEYSPACE_NAME, kind="concepts"):
    import pickle
    # load dict labels
    file_name = f"{keyspace}_emb_{kind}.pkl"
    pkl_file = os.path.join(_pickle_path, file_name)
    print(pkl_file)
    if os.path.isfile(pkl_file):
        output = open(pkl_file, "rb")
        df = pickle.load(output)
        output.close()
    else:
        df = None

    return df

def from_id_to_identification(triples, dict_entities, entity_id):
    dict_map = {}
    for row in triples:
        if row[1] == f"@has-{entity_id}":
            dict_map[row[0]] = dict_entities[row[2]]["value"]
    return dict_map


def create_embedding_dataframe(
    dict_emb_concepts, entity_name, with_mapping=False, dict_map=None, index_col=None
):
    import pandas as pd
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



#%% Knowledge related functions!

from from_triples_to_embedding import get_concept_embeddings
def dump_full_embedding(keyspace=KEYSPACE_NAME):
    data, model = get_data_model()

    # embedding = get_concept_embeddings(data, model, embedding_type='entity')
    entities_emb, relations_emb = get_datamodel_embedding_dict(data, model, True)
    # return embedding
    return entities_emb


def get_datamodel_embedding_dict(X, model, merge=True):
    import numpy as np

    relations = np.unique(X[:, 1])
    relations_emb = dict(
        zip(relations, model.get_embeddings(relations, embedding_type="relation"))
    )

    subs = np.unique(X[:, 0])
    objs = np.unique(X[:, 2])

    if merge:
        entities = np.unique(np.concatenate((subs, objs), axis=None))
        entities_emb = get_concept_embeddings(entities, model, embedding_type="entity")
        return entities_emb, relations_emb
    else:
        subs_emb = get_concept_embeddings(subs, model, embedding_type="entity")
        objs_emb = get_concept_embeddings(objs, model, embedding_type="entity")
        return subs_emb, relations_emb, objs_emb


def get_datamodel_embedding(X, model, mode="entity"):
    if not model.is_fitted:
        msg = "Model has not been fitted."
        logger.error(msg)
        raise ValueError(msg)

    modes = ("triple", "entity", "relation")
    if mode not in modes:
        msg = "Argument `mode` must be one of the following: {}.".format(
            ", ".join(modes)
        )
        logger.error(msg)
        raise ValueError(msg)

    if mode == "triple" and (len(X.shape) != 2 or X.shape[1] != 3):
        msg = "For 'triple' mode the input X must be a matrix with three columns."
        logger.error(msg)
        raise ValueError(msg)

    if mode in ("entity", "relation") and len(X.shape) != 1:
        msg = "For 'entity' or 'relation' mode the input X must be an array."
        raise ValueError(msg)

    if mode == "triple":
        import numpy as np

        s = model.get_embeddings(X[:, 0], embedding_type="entity")
        p = model.get_embeddings(X[:, 1], embedding_type="relation")
        o = model.get_embeddings(X[:, 2], embedding_type="entity")
        emb = np.hstack((s, p, o))
    else:
        emb = model.get_embeddings(X, embedding_type=mode)

    return emb

#%% Extra functions!
def get_model(model_file_name="best_model.pkl"):
    model_file = os.path.join(_model_path, model_file_name)
    exists_model = os.path.isfile(model_file)

    from ampligraph.latent_features import restore_model

    if exists_model:
        model = restore_model(model_file)
    else:
        try:
            input("Warning, No model! Press enter to continue")
        except SyntaxError:
            pass

    return model

def get_data_model(keyspace=KEYSPACE_NAME):
    # get original triples
    csv_file = "triples_" + keyspace + ".csv"
    from ampligraph.datasets import load_from_csv

    data = load_from_csv(_3ples_directory, csv_file, sep=",")
    # get model
    model_file_name = f"best_model_{keyspace}_{len(data)}_ComplEx.pkl"
    # model_file_name = "best_model.pkl"
    model = get_model(model_file_name)  # model_file_name = "best_model.pkl"

    return data, model



#%%
if __name__ == "__main__":
    print("This program is being run by itself")
    main()

else:
    print("I am being imported from another module")
