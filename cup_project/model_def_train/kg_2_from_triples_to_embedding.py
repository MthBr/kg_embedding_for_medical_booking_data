# import numpy as np
import os
import logging

#%%PARAMS
from cup_project.config import data_dir, reportings_dir, log_dir

KEYSPACE_NAME = "cup_1747_1" # cup_1747_1 cup_100_1 cup_200_2  cup_1000_1
# KEYSPACE_NAME = "cup_1"


#%% Dirs variables

_3ples_directory = data_dir / "processed" / "3ples"
_dictionay_directory = data_dir / "processed" /  "dicts"

_model_path = data_dir / 'models'

_embedding_directory = data_dir / "model_embeddings"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



#%% main method definition
def embed_triples(triples):

    from numpy import array

    X = array(triples)

    model = split_and_train(X)
    all_concepts_emb = get_save_all_concepts_emnb(model)
    
    generate_model_visualization(model) #
    
    return all_concepts_emb


#%%

def split_and_train(X, model_path = _model_path):
    from ampligraph.evaluation import train_test_split_no_unseen
    
    #X_train, X_valid
    X_train, X_test = train_test_split_no_unseen( 
        X, test_size=round(len(X) * 15 / 100), seed=0, allow_duplication=False
    )  # len(X)*3/100 len(X)*4/100   testlen(triples)/40  len(X)/31   40,39,31,30  test_size=100, seed=0, allow_duplication=False

    print("Train set size: ", X_train.shape)
    print("Test set size: ", X_test.shape)

    print(f"Using  {KEYSPACE_NAME}")

    model_file_name = f"best_model_{KEYSPACE_NAME}_{len(X)}_ConvE.pkl" #_ComplEx.pkl
    # model_file_name = "best_model.pkl"

    model_file = os.path.join(model_path, model_file_name)

    exists_model = os.path.isfile(model_file)
    from ampligraph.latent_features import save_model, restore_model

    if exists_model:
        model = restore_model(model_file)
    else:
        # AmpliGraph has implemented several Knoweldge Graph Embedding models
        # (TransE, ComplEx, DistMult, HolE),
        # but to begin with we're just going to use the ComplEx
        # model (with  default values), so lets import that:
        from ampligraph.latent_features import ConvE#, ComplEx

#        model = ComplEx(
#            batches_count=512, # 64 128 
#            seed=0,
#            epochs=250,
#            k=350,
#            eta=15, # 5 15 20
#            optimizer="adam",
#            optimizer_params={"lr": 1e-3},   # 1e-4
#            loss="nll", # multiclass_nll
#            regularizer="LP",
#            regularizer_params={"p": 3, "lambda": 1e-2},  # 1e-5
#            verbose=True,
#        )
        model = ConvE(
            batches_count=16384, # 64 128  8192 16384
            seed=2,
            epochs=400,
            k=150,
            #eta=20, # 5 15 20
            optimizer="adam",
            optimizer_params={"lr": 1e-1},   # 1e-4
            loss="bce", # multiclass_nll
            regularizer="LP",
            regularizer_params={"p": 3, "lambda": 1e-05},  # 1e-5
            verbose=True,
            low_memory=True
        )
        import tensorflow as tf

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        model.fit(X_train,
        early_stopping=False,
        early_stopping_params={'corrupt_side':'o', 'entities_size':0}
        )
        save_model(model, model_file)
        # del model

    if model.is_fitted:
        print("The model is fit!")
    else:
        print("The model is not fit! Did you skip a step?")

    from ampligraph.evaluation import evaluate_performance

    positives_filter = X  # np.concatenate((X_train, X_test))   #X_valid
    ranks = evaluate_performance(
        X_test,
        model=model,
        filter_triples=positives_filter,  # Corruption strategy filter defined above
        use_default_protocol=True,  # corrupt subj and obj separately while evaluating
        verbose=True,
    )

    from ampligraph.evaluation import mr_score, mrr_score, hits_at_n_score

    mr = mr_score(ranks)
    print("\nMR: %.2f" % (mr))

    mrr = mrr_score(ranks)
    print("MRR: %.2f" % (mrr))

    hits_10 = hits_at_n_score(ranks, n=10)
    print("Hits@10: %.2f" % (hits_10))
    hits_3 = hits_at_n_score(ranks, n=3)
    print("Hits@3: %.2f" % (hits_3))
    hits_1 = hits_at_n_score(ranks, n=1)
    print("Hits@1: %.2f" % (hits_1))

    return model




def generate_model_visualization(model, keyspace = KEYSPACE_NAME):
    folder_name = keyspace + "_embeddings"
    vis_emb_folder= os.path.join(_embedding_directory, folder_name)
    
    from ampligraph.utils import create_tensorboard_visualizations

    create_tensorboard_visualizations(model, vis_emb_folder)

    dict_entities = get_dict(keyspace, "concepts")

    labels_file = os.path.join(folder_name, "metadata.tsv")
    if os.path.isfile(labels_file):
        print(f"Substitute lables on file: {labels_file}")
        with open(labels_file, "r+") as f:
            s = f.read()
            for key in dict_entities:
                if "value" in dict_entities[key].keys():
                    s = s.replace(key, f"{key} : {str(dict_entities[key]['value'])}")
                else:
                    s = s.replace(
                        key, f"{dict_entities[key]['type_label'].upper()} {key}"
                    )
            f.seek(0)
            f.write(s)
            f.close()
    else:
        print(f"File {labels_file} not found. Will not substitute labels.")




def get_dict(keyspace=KEYSPACE_NAME, kind="concepts"):
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


def get_concept_embeddings(concept_list, model, embedding_type="entity"):
    import numpy as np
    concept_arr = np.array(concept_list, dtype=object)
    # print(arr.size)
    # uniq_array = np.unique(arr)
    concept_embedding = dict(
        zip(concept_arr, model.get_embeddings(concept_arr, embedding_type))
    )
    return concept_embedding


def get_all_concept_embeddings(model):
    list_concepts_ids = {}
    print("get dict ids")
    concept_dict_ids = get_dict(keyspace=KEYSPACE_NAME, kind="ids")
    
    concepts_name = list(concept_dict_ids.keys())
    
    for concept in concepts_name:
        concept_list = concept_dict_ids[concept]
        print(f"get {concept} embedding")
        embedding_dict = get_concept_embeddings(
            concept_list, model, embedding_type="entity"
        )
        list_concepts_ids[concept] = embedding_dict
        # print(concept_embedding)
    return list_concepts_ids

def get_save_all_concepts_emnb(model):
    all_concepts_emb = get_all_concept_embeddings(model)

    import pickle

    dict_emb = KEYSPACE_NAME + "_dict_embeddings.pkl"
    dict_file = os.path.join(_dictionay_directory, dict_emb)
    output = open(dict_file, "wb")
    pickle.dump(all_concepts_emb, output)
    output.close()
    return all_concepts_emb







#%%
if __name__ == "__main__":
    csv_directory = _3ples_directory
    csv_file = "triples_" + KEYSPACE_NAME + ".csv"

    from ampligraph.datasets import load_from_csv

    data = load_from_csv(csv_directory, csv_file, sep=",")

    #    data = pd.read_csv(csv_directory + '/' + csv_file, sep=',', header=None, names=None, dtype=str)
    #    data = data.drop_duplicates()
    #    data = data.values

    embed_triples(data)
    

    #dict_concepts = get_dict()
else:
    print("I am being imported from another module")



