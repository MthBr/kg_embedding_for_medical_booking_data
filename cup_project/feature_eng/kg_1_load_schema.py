from grakn.client import GraknClient
import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

#%%
HOW_MANY = 1747  #1747 100 200(2) 500 1000 2000   #1 * 10 ** 3
SEED = 1

#KEYSPACE_NAME = "cup_1"
KEYSPACE_NAME = f"cup_{int(HOW_MANY)}_{int(SEED)}"
SCHEMA_PATH = "schemas/cup-rule2.gql"   # cup-rules   cup-network-schema_v3

# KEYSPACE_NAME = "cup_i_one"
# SCHEMA_PATH = 'schemas/schema_cup_i_one.gql'

# KEYSPACE_NAME = "phone_calls"
# SCHEMA_PATH = '../../GraknExamples/schemas/phone-calls-schema.gql'


# 192.168.1.20  localhost
GRAKN_URI = "localhost:48555"

#%%

print("Starting... ")
with GraknClient(uri=GRAKN_URI) as client:
    print("Searching for schema: " + KEYSPACE_NAME)
    ks_list = client.keyspaces().retrieve()
    print(ks_list)
    if KEYSPACE_NAME in ks_list:
        print("Deleting keyspace " + KEYSPACE_NAME + " ....")
        #client.keyspaces().delete(KEYSPACE_NAME)
        print("Deleted keyspace: " + KEYSPACE_NAME)
    else:
        print("no deletion ")

#%%

with GraknClient(uri=GRAKN_URI) as client:
    print("Opening schema " + KEYSPACE_NAME + " ...")
    with open(os.path.join(file_dir, SCHEMA_PATH), "r") as schema:
        with client.session(keyspace=KEYSPACE_NAME) as session:
            define_query = schema.read()
            print("Loading schema " + KEYSPACE_NAME + " ...")
            with session.transaction().write() as transaction:
                transaction.query(define_query)
                transaction.commit()
                print(KEYSPACE_NAME + " schema loaded")


print("... Ended")


# from kglib.kgcn.examples.diagnosis import diagnosis

