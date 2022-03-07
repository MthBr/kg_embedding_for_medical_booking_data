# import datetime

import logging
import os
import sys
import numpy as np

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_model_path = os.path.abspath(os.path.join(file_dir, os.pardir, "models"))
_3ples_directory = os.path.abspath(os.path.join(file_dir, os.pardir, "3ples"))
_dictionay_directory = os.path.abspath(os.path.join(file_dir, os.pardir, "dicts"))

#%% GLOBAL VARIABLES

KEYSPACE_NAME = "cup_1"
K_model = 150

#%% main method definition


#%% general method definition


def get_all_concept_embeddings(model, entities, relations):
    list_concepts_ids = {}
    print("get dict ids")
    concept_dict_ids = get_dict(keyspace=KEYSPACE_NAME, kind="ids")

    for entity in entities:
        entity_list = concept_dict_ids[entity]
        print(f"get {entity} embedding")
        embedding_dict = get_concept_embeddings(
            entity_list, model, embedding_type="entity"
        )
        list_concepts_ids[entity] = embedding_dict
        # print(concept_embedding)

    for relation in relations:
        entity_list = concept_dict_ids[relation]
        print(f"get {relation} embedding")
        embedding_dict = get_concept_embeddings(
            entity_list, model, embedding_type="entity"
        )
        list_concepts_ids[relation] = embedding_dict
        # print(concept_embedding)
    return list_concepts_ids


def get_concept_embeddings(concept_list, model, embedding_type="entity"):
    concept_arr = np.array(concept_list, dtype=object)
    # print(arr.size)
    # uniq_array = np.unique(arr)
    concept_embedding = dict(
        zip(concept_arr, model.get_embeddings(concept_arr, embedding_type))
    )
    return concept_embedding


def get_dict(keyspace=KEYSPACE_NAME, kind="entities"):
    import pickle

    # load dict labels
    dictionay_name = f"{keyspace}_dict_{kind}.pkl"
    dictionay_file = os.path.join(_dictionay_directory, dictionay_name)
    if os.path.isfile(dictionay_file):
        pkl_file = open(dictionay_file, "rb")
        dict_entities = pickle.load(pkl_file)
        pkl_file.close()
    else:
        dict_entities = None

    return dict_entities


# def dump_full_embedding(keyspace=KEYSPACE_NAME):

#     data, model = get_data_model()

#     # embedding = get_concept_embeddings(data, model, embedding_type='entity')
#     entities_emb, relations_emb = get_datamodel_embedding_dict(data, model, True)
#     # return embedding
#     return entities_emb


# def get_datamodel_embedding_dict(X, model, merge=True):
#     # import numpy as np

#     relations = np.unique(X[:, 1])
#     relations_emb = dict(
#         zip(relations, model.get_embeddings(relations, embedding_type="relation"))
#     )

#     subs = np.unique(X[:, 0])
#     objs = np.unique(X[:, 2])

#     if merge:
#         entities = np.unique(np.concatenate((subs, objs), axis=None))
#         entities_emb = get_concept_embeddings(entities, model, embedding_type="entity")
#         return entities_emb, relations_emb
#     else:
#         subs_emb = get_concept_embeddings(subs, model, embedding_type="entity")
#         objs_emb = get_concept_embeddings(objs, model, embedding_type="entity")
#         return subs_emb, relations_emb, objs_emb


# def get_datamodel_embedding(X, model, mode="entity"):
#     if not model.is_fitted:
#         msg = "Model has not been fitted."
#         logger.error(msg)
#         raise ValueError(msg)

#     modes = ("triple", "entity", "relation")
#     if mode not in modes:
#         msg = "Argument `mode` must be one of the following: {}.".format(
#             ", ".join(modes)
#         )
#         logger.error(msg)
#         raise ValueError(msg)

#     if mode == "triple" and (len(X.shape) != 2 or X.shape[1] != 3):
#         msg = "For 'triple' mode the input X must be a matrix with three columns."
#         logger.error(msg)
#         raise ValueError(msg)

#     if mode in ("entity", "relation") and len(X.shape) != 1:
#         msg = "For 'entity' or 'relation' mode the input X must be an array."
#         raise ValueError(msg)

#     if mode == "triple":
#         import numpy as np

#         s = model.get_embeddings(X[:, 0], embedding_type="entity")
#         p = model.get_embeddings(X[:, 1], embedding_type="relation")
#         o = model.get_embeddings(X[:, 2], embedding_type="entity")
#         emb = np.hstack((s, p, o))
#     else:
#         emb = model.get_embeddings(X, embedding_type=mode)

#     return emb


def get_data_model(keyspace=KEYSPACE_NAME):
    # get original triples
    csv_file = "triples_" + keyspace + ".csv"
    from ampligraph.datasets import load_from_csv

    data = load_from_csv(_3ples_directory, csv_file, sep=",")
    # get model
    model_file_name = f"best_model_{keyspace}{len(data)}_ComplEx.pkl"
    # model_file_name = "best_model.pkl"
    model = get_model(model_file_name)  # model_file_name = "best_model.pkl"

    return data, model


def get_concepts(keyspace=KEYSPACE_NAME, uri="localhost:48555"):
    import from_grakn_to_triples

    entities = from_grakn_to_triples.get_all_entities(keyspace=keyspace, uri=uri)
    relations = from_grakn_to_triples.get_all_relations(keyspace=keyspace, uri=uri)
    return entities, relations


#%%


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


#%%
if __name__ == "__main__":
    print("This program is being run by itself")
    main()
else:
    print("I am being imported from another module")
