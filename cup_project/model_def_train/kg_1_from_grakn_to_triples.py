import datetime
import warnings
import os
from grakn.client import GraknClient

#import sys
#sys.path.append(file_dir)


#%% GLOBAL VARIABLES
KEYSPACE_NAME = "cup_1747_1" # cup_1747_1 cup_6484_1  cup_100_1 cup_200_2  cup_1000_1
# KEYSPACE_NAME = "cup_1_new"
# KEYSPACE_NAME = "phone_calls"

#  192.168.1.20:48555
#  localhost:48555
GRAKN_URI = "localhost:48555"

triples = list()
dict_concepts = dict()
dict_ids = dict()
time_limit = "2018-01-01"
#time_limit = None

_local_dir=False  # True False

#%% Dirs variables
file_dir = os.path.dirname(__file__)
if _local_dir:
    working_dir = os.path.join(file_dir, os.pardir)
else:
    working_dir = os.path.abspath(os.path.join(file_dir, f"../../../../Pycodes/cup_kg/"))
_3ples_directory = os.path.abspath(os.path.join(working_dir, "3ples"))
os.makedirs(_3ples_directory, exist_ok=True)
_dictionay_directory = os.path.abspath(os.path.join(working_dir, "dicts"))
os.makedirs(_dictionay_directory, exist_ok=True)





#%% main method definition
def get_triples_dictionary(keyspace=KEYSPACE_NAME, uri=GRAKN_URI):
    client = GraknClient(uri=uri)
    session = client.session(keyspace=keyspace)

    with session.transaction().read() as tx:
        # Change the terminology here onwards from thing -> node and role -> edge

        entities = get_all_entities(tx=tx)
        print(f"Found node types: {entities}")
        print("Generating triples from entities: \n")
        start_time = datetime.datetime.now()
        for entity in entities:
            print(f"\tGenerating triples from entity: {entity}\n")
            temp_triples, temp_dict_entities, temp_dict_ids = get_triples_from_entity(
                entity, tx=tx
            )
            triples.extend(temp_triples)
            dict_concepts.update(temp_dict_entities)
            dict_ids.update(temp_dict_ids)
            print(
                f"\tGenerated {len(temp_triples)} triples with {len(temp_dict_entities)} dic entities \n"
            )
        end_time = datetime.datetime.now()
        print(f"Finished in: {end_time-start_time}\n")

        relations = get_all_relations(tx=tx)
        print(f"Found node types: {relations}")
        print("Generating triples from relations: \n")
        start_time = datetime.datetime.now()
        for relation in relations:
            print(f"\tGenerating triples from relation: {relation}\n")
            temp_triples, temp_dict_entities, temp_dict_ids = get_triples_from_relation(
                relation, tx=tx
            )
            triples.extend(temp_triples)
            dict_concepts.update(temp_dict_entities)
            dict_ids.update(temp_dict_ids)
            print(
                f"\tGenerated {len(temp_triples)} triples with {len(temp_dict_entities)} dic entities\n"
            )
        end_time = datetime.datetime.now()
        print(f"Finished in: {end_time-start_time}.\n")

        print(
            f"In total: Generated {len(triples)} triples with {len(dict_concepts)} dic entities\n"
        )

        return triples, dict_concepts, dict_ids


#%% tools method definitions
def get_n_save_triples_dict(keyspace=KEYSPACE_NAME, uri=GRAKN_URI,csv_dir=_3ples_directory,dict_dir= _dictionay_directory):
    triples, dict_concepts, dict_ids = get_triples_dictionary(keyspace, uri)
    save_triples_dict(triples, dict_concepts, dict_ids, keyspace=KEYSPACE_NAME, csv_dir=_3ples_directory, dict_dir= _dictionay_directory)
    return triples, dict_concepts, dict_ids





def save_triples_dict(triples, dict_concepts, dict_ids, keyspace=KEYSPACE_NAME, csv_dir=_3ples_directory, dict_dir= _dictionay_directory):
    import csv
    triples_file_name = "triples_" + KEYSPACE_NAME + ".csv"
    triples_file = os.path.join(csv_dir, triples_file_name)
    with open(triples_file, "w") as writeFile:
        print(f"Writing: {triples_file} to {csv_dir}.\n")
        writer = csv.writer(writeFile)
        writer.writerows(triples)
    writeFile.close()

    import pickle
    # write python dict to a file
    dict_file_name = KEYSPACE_NAME + "_dict_concepts.pkl"
    dict_file = os.path.join(dict_dir, dict_file_name)
    output = open(dict_file, "wb")
    print(f"Writing: {dict_file_name} to {dict_dir}.\n")
    pickle.dump(dict_concepts, output)
    output.close()

    dict_file_name = KEYSPACE_NAME + "_dict_ids.pkl"
    dict_file = os.path.join(dict_dir, dict_file_name)
    output = open(dict_file, "wb")
    print(f"Writing: {dict_file_name} to {dict_dir}.\n")
    pickle.dump(dict_ids, output)
    output.close()
    
    return True




def open_keyspace(func):
    def wrapper(*args, **kwargs):
        # print(f"Ran with args: {args}, kwargs: {kwargs}")
        if "tx" in kwargs:
            result = func(tx=kwargs["tx"], *args)
        elif "uri" in kwargs and "keyspace" in kwargs:
            client = GraknClient(uri=kwargs["uri"])
            session = client.session(keyspace=kwargs["keyspace"])
            with session.transaction().read() as tx:
                result = func(tx=tx, *args)
        else:
            raise Warning("No (tx) or (uri and keyspace) specified")
        return result

    return wrapper


# def get_all_entities(keyspace=KEYSPACE_NAME, uri="localhost:48555"):
#     client = GraknClient(uri=uri)
#     session = client.session(keyspace=keyspace)

#     with session.transaction().read() as tx:
#         get_all_entities(tx)
#     return thing_types


@open_keyspace
def get_all_entities(**tr_arg):
    # print(f"Ran with tr_arg: {tr_arg}")
    schema_concepts = tr_arg["tx"].query("match $x sub entity; get;").collect_concepts()
    thing_types = [schema_concept.label() for schema_concept in schema_concepts]
    thing_types.remove("entity")
    return thing_types


@open_keyspace
def get_all_relations(**tr_arg):
    # print(f"Ran with tr_arg: {tr_arg}")
    schema_concepts = (
        tr_arg["tx"].query("match $x sub relation; get;").collect_concepts()
    )
    thing_types = [schema_concept.label() for schema_concept in schema_concepts]
    thing_types.remove("relation")
    thing_types = [x for x in thing_types if not x.startswith("@")]
    return thing_types


@open_keyspace
def get_triples_from_entity(entity_name, **tr_arg):
    from collections import defaultdict

    fun_triples = []
    fun_dict_entities = {}
    fun_dict_ids = defaultdict(list)
    concept_maps = tr_arg["tx"].query(f"match $x isa {entity_name}; get $x;")
    for entity in concept_maps.collect_concepts():
        fun_dict_ids[entity_name].append(entity.id)
        fun_dict_entities[entity.id] = {"type_label": entity.type().label()}
        for attribute in entity.attributes():
            fun_dict_entities[attribute.id] = {
                "type_label": attribute.type().label(),
                "value": attribute.value(),
            }
            fun_triples.append(
                [entity.id, "@has-" + attribute.type().label(), attribute.id]
            )

    return fun_triples, fun_dict_entities, fun_dict_ids


@open_keyspace
def get_triples_from_relation(relation_name, **tr_arg):
    from collections import defaultdict

    fun_triples = []
    fun_dict_relations = {}
    fun_dict_ids = defaultdict(list)
    # fun_dict_ids = {}
    if time_limit:
        if relation_name == "reservation":
            concept_maps = tr_arg["tx"].query(
                f"match $res isa reservation, has reservation-date $date; $date < {time_limit}; get $res;"
            )
        elif relation_name == "referral":
            concept_maps = tr_arg["tx"].query(
                f"match $res (booked-referral: $ref) isa reservation, has reservation-date $date; $date < {time_limit}; get $ref;"
            )
        elif relation_name == "health-care":
            concept_maps = tr_arg["tx"].query(
                f"match $x isa health-care, has booked-date $date; $date < {time_limit}; get $x;"
            )
        elif relation_name == "provision":
            concept_maps = tr_arg["tx"].query(
                f"match $x isa {relation_name}; get $x;"
            )
        else:
            warnings.warn(
                "ATTENTION: there is a time_limit without a proper relation.", Warning
            )
    else:
        concept_maps = tr_arg["tx"].query(f"match $x isa {relation_name}; get $x;")
    # fun_dict_ids[relation] = []
    for relation in concept_maps.collect_concepts():
        fun_dict_ids[relation_name].append(relation.id)
        fun_dict_relations[relation.id] = {"type_label": relation.type().label()}
        for attribute in relation.attributes():
            fun_dict_relations[attribute.id] = {
                "type_label": attribute.type().label(),
                "value": attribute.value(),
            }
            fun_triples.append(
                [relation.id, "@has-" + attribute.type().label(), attribute.id]
            )
        for entity, role in zip(relation.role_players(), relation.role_players_map()):
            fun_triples.append([relation.id, role.label(), entity.id])

    return fun_triples, fun_dict_relations, fun_dict_ids


@open_keyspace
def get_id_from_concept(concept, **tr_arg):
    fun_triples = []
    concept_maps = tr_arg["tx"].query(f"match $x isa {concept}; get; offset 0;")
    for concept in concept_maps.collect_concepts():
        fun_triples.append(concept.id)

    return fun_triples


#%% extra tools
def get_all_concepts(tx):
    schema_concepts = tx.query("match $x sub thing; get;").collect_concepts()
    thing_types = [
        schema_concept.is_relation_type() for schema_concept in schema_concepts
    ]
    return thing_types


def get_all_playing(tx):
    schema_concepts = tx.query("match $x sub thing; get;").collect_concepts()
    thing_types = [schema_concept.playing() for schema_concept in schema_concepts]
    return thing_types


def get_thing_types(tx):
    schema_concepts = tx.query("match $x sub thing; get;").collect_concepts()
    thing_types = [schema_concept.label() for schema_concept in schema_concepts]
    return thing_types


def get_role_types(tx):
    schema_concepts = tx.query("match $x sub role; get;").collect_concepts()
    role_types = ["has"] + [role.label() for role in schema_concepts]
    return role_types


#%%  main
if __name__ == "__main__":
    get_n_save_triples_dict()
    #get_triples_dictionary()
