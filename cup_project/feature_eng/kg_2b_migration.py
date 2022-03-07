# Inspired by
# Copyright 2019 Grakn Labs Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import datetime
import multiprocessing
import os
import sys
import ast
import time
import numpy as np

from grakn.client import GraknClient

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

#%%PARAMS
HOW_MANY = 1747  # 100 200(2) 500 1000 2000   #1 * 10 ** 3
SEED = 1
Patient_GOOD = True #True


#%% GLOBAL VARIABLES
_cup_datasets_path = os.path.abspath(
    os.path.join(file_dir, f"../../../../Pycodes/csv/random_{HOW_MANY}_{SEED}_{Patient_GOOD}")
)


KEYSPACE_NAME = f"cup_{int(HOW_MANY)}_{int(SEED)}"
# 192.168.1.20  localhost
GRAKN_URI = "localhost:48555"

entity_queries = {
    "patient": [],
    "practitioner": [],
    "booking-staff": [],
    "medical-branch": [],
    "health-service-provision": [],
    "appointment-provider": [],
}
relation_queries = {"referral": [], "reservation": []}

#%% ENTITY - RELATION TEMPLATE


def generate_has(k_type, k_value):
    query = ""
    val = str(k_value).strip('"')
    if len(val) != 0:
        query += ", has " + k_type + " " + str(k_value)
    return query


def entity_template(data):
    query = "insert $x isa " + data["type"]
    for attribute in data["attributes"]:
        if isinstance(attribute["value"], list):
            for elem in attribute["value"]:
                query += generate_has(attribute["type"], str(elem))
        else:
            query += generate_has(attribute["type"], str(attribute["value"]))
    query += ";"
    return query


def relation_template(data):
    query = "match "
    for r, roleplayer in enumerate(data["roleplayers"]):
        query += "$" + str(r) + " isa " + roleplayer["type"]
        for a, roleplayer_attribute in enumerate(roleplayer["roleplayer_attributes"]):
            if isinstance(roleplayer_attribute["key_value"], list):
                for b, elem in enumerate(roleplayer_attribute["key_value"]):
                    query += generate_has(roleplayer_attribute["key_type"], str(elem))
                    if b < len(roleplayer_attribute["key_value"]) - 1:
                        query += (
                            "; $"
                            + str(r)
                            + "-"
                            + str(b + 1)
                            + " isa "
                            + roleplayer["type"]
                        )
            else:
                query += generate_has(
                    roleplayer_attribute["key_type"],
                    str(roleplayer_attribute["key_value"]),
                )
                # if a < len(roleplayer["roleplayer_attributes"]) - 1:
                #     query += ","
        query += "; "

    if "roleplayers_relation" in data:
        data_rel = data["roleplayers_relation"][0]
        for r, roleplayer in enumerate(data_rel["roleplayers"]):
            query += "$rel" + str(r) + " isa " + roleplayer["type"]
            for a, roleplayer_attribute in enumerate(
                roleplayer["roleplayer_attributes"]
            ):
                if isinstance(roleplayer_attribute["key_value"], list):
                    for b, elem in enumerate(roleplayer_attribute["key_value"]):
                        query += generate_has(
                            roleplayer_attribute["key_type"], str(elem)
                        )
                        if b < len(roleplayer_attribute["key_value"]) - 1:
                            query += (
                                "; $rel"
                                + str(r)
                                + "-"
                                + str(b + 1)
                                + " isa "
                                + roleplayer["type"]
                            )
                else:
                    query += generate_has(
                        roleplayer_attribute["key_type"],
                        str(roleplayer_attribute["key_value"]),
                    )
                    # if a < len(roleplayer["roleplayer_attributes"]) - 1:
                    #     query += ","
            query += "; "

        query += "$r ("
        for r, roleplayer in enumerate(data_rel["roleplayers"]):
            query += roleplayer["role_name"] + ": $rel" + str(r)

            for roleplayer_attribute in roleplayer["roleplayer_attributes"]:
                if isinstance(roleplayer_attribute["key_value"], list):
                    for b, elem in enumerate(roleplayer_attribute["key_value"]):
                        if b < len(roleplayer_attribute["key_value"]) - 1:
                            query += "-" + str(b + 1)
                            query += ", " + roleplayer["role_name"] + ": $rel" + str(r)

            if r < len(data_rel["roleplayers"]) - 1:
                query += ", "

        query += ") isa " + data_rel["type"]

        if "attributes" in data_rel:
            # query += "; $r"
            for a, attribute in enumerate(data_rel["attributes"]):
                query += generate_has(attribute["type"], str(attribute["value"]))
                # if a < len(data_rel["attributes"]) - 1:
                #     query += ","
        query += "; "

    # match the relation if required
    # if "relation_key_attributes" in data:
    #     query += "$x"
    #     for a, relation_key_attribute in enumerate(data["relation_key_attributes"]):
    #         query += (
    #             " has "
    #             + relation_key_attribute["key_type"]
    #             + " "
    #             + str(relation_key_attribute["key_value"])
    #         )
    #         if a < len(data["relation_key_attributes"]) - 1:
    #             query += ","
    #     query += "; "

    query += "insert $x ("
    for r, roleplayer in enumerate(data["roleplayers"]):
        query += roleplayer["role_name"] + ": $" + str(r)

        for roleplayer_attribute in roleplayer["roleplayer_attributes"]:
            if isinstance(roleplayer_attribute["key_value"], list):
                for b, elem in enumerate(roleplayer_attribute["key_value"]):
                    if b < len(roleplayer_attribute["key_value"]) - 1:
                        query += "-" + str(b + 1)
                        query += ", " + roleplayer["role_name"] + ": $" + str(r)

        if r < len(data["roleplayers"]) - 1:
            query += ", "

    if "relation_key_attributes" in data:
        query += ")"
    elif "roleplayers_relation" in data:
        query += (
            ", "
            + data["roleplayers_relation"][0]["role_name"]
            + ": $r) isa "
            + data["type"]
        )
    else:
        query += ") isa " + data["type"]

    if "attributes" in data:
        for a, attribute in enumerate(data["attributes"]):
            query += generate_has(attribute["type"], str(attribute["value"]))

    query += ";"
    return query


#%% OTHER FUNCTIONS


def string(value):
    return '"' + value + '"'


def unique_append(lst, key, item):
    #    if item not in lst[key]:
    #        lst[key].append(item)
    lst[key].append(item)


# def zone_already_added(zone_name):
#    zone_already_added = False
#    for zone_query in relation_queries["zone"]:
#        if 'isa zone; $x has name "' + zone_name + '"' in zone_query:
#            zone_already_added = True
#            break
#    return zone_already_added


def parse_data_to_dictionaries(template):
    """
      1. reads the file through a stream,
      2. adds the dictionary to the list of items
      :param input.file as string: the path to the data file, minus the format
      :returns items as list of dictionaries: each item representing a data item from the file at input.file
    """
    items = []
    with open(os.path.join(_cup_datasets_path, template["file"] + ".csv")) as data:  # 1
        for row in csv.DictReader(
            data, skipinitialspace=True, delimiter=",", quotechar='"'
        ):
            item = {key: value for key, value in row.items()}
            items.append(item)  # 2
    return items


#%% TEMPLATES CSV


def patient_template(patient):
    # insert patient
    unique_append(
        entity_queries,
        "patient",
        entity_template(
            {
                "type": "patient",
                "attributes": [
                    {"type": "encrypted-nin-id", "value": string(patient["sa_ass_cf"])},
                    {"type": "gender", "value": int(float(patient["sa_sesso_id"]))},
                    {"type": "nuts-istat-code", "value": patient["sa_comune_id"]},
                ],
            }
        ),
    )


def practitioner_template(practitioner):
    # insert practitioner
    unique_append(
        entity_queries,
        "practitioner",
        entity_template(
            {
                "type": "practitioner",
                "attributes": [
                    {"type": "practitioner-id", "value": practitioner["sa_med_id"]}
                ],
            }
        ),
    )


def booking_staff_template(booking_staff):
    # insert booking_staff
    unique_append(
        entity_queries,
        "booking-staff",
        entity_template(
            {
                "type": "booking-staff",
                "attributes": [
                    {"type": "booking-agent-id", "value": booking_staff["sa_ut_id"]},
                    {
                        "type": "local-health-department-id",
                        "value": string(booking_staff["sa_asl"]),
                    },
                ],
            }
        ),
    )


def medical_branch_template(medical_branch):
    # insert medical_branch
    unique_append(
        entity_queries,
        "medical-branch",
        entity_template(
            {
                "type": "medical-branch",
                "attributes": [
                    {
                        "type": "medical-branch-id",
                        "value": string(medical_branch["id_branca"]),
                    },
                    {
                        "type": "refined-medical-branch-id",
                        "value": medical_branch["indx"],
                    },
                    {
                        "type": "branch-description",
                        "value": string(medical_branch["descrizione_y"]),
                    },
                    {
                        "type": "official-branch-description",
                        "value": string(medical_branch["descrizione_x"]),
                    },
                ],
            }
        ),
    )


def health_service_provision_template(health_service_provision):
    # insert health_service_provision
    unique_append(
        entity_queries,
        "health-service-provision",
        entity_template(
            {
                "type": "health-service-provision",
                "attributes": [
                    {
                        "type": "health-service-id",
                        "value": [
                            string(x)
                            for x in ast.literal_eval(
                                health_service_provision["list_sa_pre_id"]
                            )
                        ],
                    },
                    {
                        "type": "refined-health-service-id",
                        "value": health_service_provision["indx_prestazione"],
                    },
                    {
                        "type": "refined-medical-branch-id",
                        "value": ast.literal_eval(
                            health_service_provision["list_branca"]
                        ),
                    },
                    {
                        "type": "health-service-description",
                        "value": string(health_service_provision["descrizione"]),
                    },
                ],
            }
        ),
    )


def appointment_provider_template(appointment_provider):
    # insert appointment_provider
    unique_append(
        entity_queries,
        "appointment-provider",
        entity_template(
            {
                "type": "appointment-provider",
                "attributes": [
                    {
                        "type": "referral-centre-id",
                        "value": string(appointment_provider["sa_uop_codice_id"]),
                    },
                    {
                        "type": "nuts-istat-code",
                        "value": appointment_provider["struttura_comune"],
                    },
                    {
                        "type": "local-health-department-id",
                        "value": string(appointment_provider["struttura_codasl"]),
                    },
                ],
            }
        ),
    )


def referral_template(referral):
    # insert referral
    unique_append(
        relation_queries,
        "referral",
        relation_template(
            {
                "type": "referral",
                "roleplayers": [
                    {
                        "type": "patient",
                        "roleplayer_attributes": [
                            {
                                "key_type": "encrypted-nin-id",
                                "key_value": string(referral["sa_ass_cf"]),
                            },
                            {
                                "key_type": "gender",
                                "key_value": int(float(referral["sa_sesso_id"])),
                            },
                            {
                                "key_type": "nuts-istat-code",
                                "key_value": referral["sa_comune_id"],
                            },
                        ],
                        "role_name": "referred-patient",
                    },
                    {
                        "type": "practitioner",
                        "roleplayer_attributes": [
                            {
                                "key_type": "practitioner-id",
                                "key_value": referral["sa_med_id"],
                            }
                        ],
                        "role_name": "referrer",
                    },
                    {
                        "type": "health-service-provision",
                        "roleplayer_attributes": [
                            {
                                "key_type": "refined-health-service-id",  # TODO sicuro? non melgio sa_pre_id
                                "key_value": ast.literal_eval(
                                    referral["list_indx_prestazione"]
                                ),
                            }
                        ],
                        "role_name": "prescribed-health-service",
                    },
                    {
                        "type": "medical-branch",
                        "roleplayer_attributes": [
                            {
                                "key_type": "medical-branch-id",
                                "key_value": string(referral["sa_branca_id"]),
                            }
                        ],
                        "role_name": "referred-medical-branch",
                    },
                ],
                "attributes": [
                    {"type": "referral-id", "value": referral["sa_impegnativa_id"]},
                    {
                        "type": "referral-modified-id",
                        "value": referral["indx_impegnativa"],
                    },
                    {"type": "referral-date", "value": referral["sa_data_prescr"]},
                    {"type": "patient-age", "value": referral["sa_eta_id"]},
                    {
                        "type": "priority-code",
                        "value": string(referral["sa_classe_priorita"]),
                    },
                    {
                        "type": "exemption-code",
                        "value": string(referral["sa_ese_id_lk"]),
                    },
                ],
            }
        ),
    )


# TODO
def reservation_template(reservation):
    # insert reservation
    unique_append(
        relation_queries,
        "reservation",
        relation_template(
            {
                "type": "reservation",
                "roleplayers": [
                    {
                        "type": "appointment-provider",
                        "roleplayer_attributes": [
                            {
                                "key_type": "referral-centre-id",
                                "key_value": string(reservation["sa_uop_codice_id"]),
                            },
                            {
                                "key_type": "local-health-department-id",
                                "key_value": string(reservation["sa_asl"]),
                            },
                        ],
                        "role_name": "referring-centre",
                    },
                    {
                        "type": "booking-staff",  # TODO attenzione doppio booking staff
                        "roleplayer_attributes": [
                            {
                                "key_type": "booking-agent-id",
                                "key_value": reservation["sa_utente_id"],
                            },
                            {
                                "key_type": "local-health-department-id",
                                "key_value": string(reservation["sa_asl"]),
                            },
                        ],
                        "role_name": "booking-agent",
                    },
                    {
                        "type": "booking-staff",  # TODO attenzione doppio booking staff
                        "roleplayer_attributes": [
                            {
                                "key_type": "booking-agent-id",
                                "key_value": reservation["sa_ut_id"],
                            },
                            {
                                "key_type": "local-health-department-id",
                                "key_value": string(reservation["sa_asl"]),
                            },
                        ],
                        "role_name": "updating-agent",
                    },
                    {
                        "type": "health-service-provision",
                        "roleplayer_attributes": [
                            {
                                "key_type": "refined-health-service-id",  #  TODO indx_prestazione sa_pre_id   refined-health-service-id
                                "key_value": reservation[
                                    "indx_prestazione"
                                ],  # TODO indx_prestazione
                            }
                        ],
                        "role_name": "reserved-health-service",
                    },
                ],
                "roleplayers_relation": [
                    {
                        "type": "referral",
                        "roleplayers": [
                            {
                                "type": "patient",
                                "roleplayer_attributes": [
                                    {
                                        "key_type": "encrypted-nin-id",
                                        "key_value": string(reservation["sa_ass_cf"]),
                                    },
                                    {
                                        "key_type": "gender",
                                        "key_value": int(
                                            float(reservation["sa_sesso_id"])
                                        ),
                                    },
                                    {
                                        "key_type": "nuts-istat-code",
                                        "key_value": reservation["sa_comune_id"],
                                    },
                                ],
                                "role_name": "referred-patient",
                            },
                            {
                                "type": "practitioner",
                                "roleplayer_attributes": [
                                    {
                                        "key_type": "practitioner-id",
                                        "key_value": reservation["sa_med_id"],
                                    }
                                ],
                                "role_name": "referrer",
                            },
                            {
                                "type": "health-service-provision",
                                "roleplayer_attributes": [
                                    {
                                        "key_type": "refined-health-service-id",  # TODO sicuro? non melgio sa_pre_id
                                        "key_value": reservation["indx_prestazione"],
                                    }
                                ],
                                "role_name": "prescribed-health-service",
                            },
                            {
                                "type": "medical-branch",
                                "roleplayer_attributes": [
                                    {
                                        "key_type": "medical-branch-id",
                                        "key_value": string(
                                            reservation["sa_branca_id"]
                                        ),
                                    }
                                ],
                                "role_name": "referred-medical-branch",
                            },
                        ],
                        "attributes": [
                            {
                                "type": "referral-id",
                                "value": reservation["sa_impegnativa_id"],
                            },
                            {
                                "type": "referral-modified-id",
                                "value": reservation["indx_impegnativa"],
                            },
                            {
                                "type": "referral-date",
                                "value": reservation["sa_data_prescr"],
                            },
                            {"type": "patient-age", "value": reservation["sa_eta_id"]},
                            {
                                "type": "priority-code",
                                "value": string(reservation["sa_classe_priorita"]),
                            },
                            {
                                "type": "exemption-code",
                                "value": string(reservation["sa_ese_id_lk"]),
                            },
                        ],
                        "role_name": "booked-referral",
                    }
                ],
                "attributes": [
                    {
                        "type": "appointment-encoding",
                        "value": string(reservation["sa_dti_id"]),
                    },
                    {"type": "booked-date", "value": reservation["sa_data_app"]},
                    {"type": "reservation-date", "value": reservation["sa_data_pren"]},
                    # {
                    #     "type": "last-reservation-change-date",
                    #     "value": reservation["sa_data_ins"],
                    # },
                    {"type": "booking-type", "value": string(reservation["sa_is_ad"])},
                    {
                        "type": "number-of-health-services",
                        "value": int(float(reservation["sa_num_prestazioni"])),
                    },
                ],
            }
        ),
    )


#%%CONSTRUCT QUERIES


def construct_queries(entity_queries, relation_queries, verbose=0):
    for template in templates:
        inputs = parse_data_to_dictionaries(template)
        if verbose:
            import numpy as np
            import time

            print("template = '{}'".format(template["file"]))
            INTERVAL = 10000
            print("num_of_inputs =", len(inputs))
            print("num_bins_float =", len(inputs) / INTERVAL)
            bins = np.array_split(inputs, int(np.ceil(len(inputs) / INTERVAL)))
            print("Number of bins = {}".format(len(bins)))
            all_count = 0
            for inputs_part in bins:
                t_partial = time.time()
                for row in inputs_part:
                    template["template"](row)
                all_count += len(inputs_part)
                print(
                    "{} Ended with time {}".format(all_count, time.time() - t_partial)
                )
        else:
            for row in inputs:
                template["template"](row)


#%% INSERT
def insert(queries, n_batch=1000, cpu=None):
    with GraknClient(uri=GRAKN_URI) as client:
        with client.session(keyspace=KEYSPACE_NAME) as session:
            transaction = session.transaction().write()
            if cpu:
                print(
                    "Using CPU core "
                    + str(cpu)
                    + "\n"
                    + "\tInserting: "
                    + str(len(queries))
                    + " queries, like: "
                    + str(queries[0])
                    + "\n"
                )
            else:
                print(
                    "Inserting: "
                    + str(len(queries))
                    + " queries, like: "
                    + str(queries[0])
                    + "\n"
                )

            queries_time = []

            for i, query in enumerate(queries):
                starting_time = time.time()
                transaction.query(query)
                queries_time.append(time.time() - starting_time)
                if cpu:
                    print(
                        "Using CPU core "
                        + str(cpu)
                        + " at "
                        + str(datetime.datetime.now())
                        + "\n"
                        + "\tQuery made in "
                        + str(datetime.timedelta(seconds=(time.time() - starting_time)))
                        + "\n"
                        + "\tNumber of query made "
                        + str(i + 1)
                        + "\n"
                        + "\tNumber of query to make "
                        + str(len(queries) - 1 - i)
                        + "\n"
                        + "\t Remaining query time "
                        + str(datetime.timedelta(seconds=round(np.mean(queries_time) * (len(queries) - 1 - i))))
                        + " seconds"
                        + "\n"
                    )
                else:
                    print(
                        "at "
                        + str(datetime.datetime.now())
                        + "\n"
                        + "\tQuery made in "
                        + str(datetime.timedelta(seconds=(time.time() - starting_time)))
                        + "\n"
                        + "\tNumber of query made "
                        + str(i + 1)
                        + "\n"
                        + "\tNumber of query to make "
                        + str(len(queries) - 1 - i)
                        + "\n"
                        + "\t Remaining query time "
                        + str(datetime.timedelta(seconds=round(np.mean(queries_time) * (len(queries) - 1 - i))))
                        + " seconds"
                        + "\n"
                    )

                if i % n_batch == 0:
                    print(i + 1)
                    print(query + "\n")
                    # if i % 1000 == 0:
                    #     print("- - - - - - - - - - - - - -")
                    # else:
                    #     print("♡ ♡ ♡ ♡ ♡ ♡ ♡ ♡ ♡ ♡ ♡ ♡ ♡")

                    if cpu:
                        print(
                            "Using CPU core "
                            + str(cpu)
                            + "\n"
                            + "\tStarting commit"
                            + "\n"
                        )
                    else:
                        print("Starting commit" + "\n")
                    starting_time = time.time()
                    transaction.commit()
                    if cpu:
                        print(
                            "Using CPU core "
                            + str(cpu)
                            + "\n"
                            + "\tFinished commit in "
                            + str(datetime.timedelta(seconds=(time.time() - starting_time)))
                            + " seconds"
                            + "\n"
                        )
                    else:
                        print(
                            "Finished commit in "
                            + str(datetime.timedelta(seconds=(time.time() - starting_time)))
                            + " seconds"
                            + "\n"
                        )
                    transaction = session.transaction().write()
            if cpu:
                print("Using CPU core " + str(cpu) + "\n" + "\tStarting commit" + "\n")
            else:
                print("Starting commit" + "\n")
            starting_time = time.time()
            transaction.commit()
            if cpu:
                print(
                    "Using CPU core "
                    + str(cpu)
                    + "\n"
                    + "\tFinished commit in "
                    + str(datetime.timedelta(seconds=(time.time() - starting_time)))
                    + " seconds"
                    + "\n"
                )
            else:
                print(
                    "Finished commit in "
                    + str(datetime.timedelta(seconds=(time.time() - starting_time)))
                    + " seconds"
                    + "\n"
                )


def insert_concurrently(queries, processes, n_batch=1000):
    cpu_count = int(multiprocessing.cpu_count() / 2)
    chunk_size = int(len(queries) / cpu_count)

    for i in range(cpu_count):
        if i == cpu_count - 1:
            chunk = queries[i * chunk_size :]
        else:
            chunk = queries[i * chunk_size : (i + 1) * chunk_size]

        process = multiprocessing.Process(
            target=insert, args=(chunk, n_batch, int(i + 1))
        )
        process.start()
        processes.append(process)

    print("Starting Process Join" + "\n")
    starting_time = time.time()
    for process in processes:
        process.join()
    print(
        "Finished Process Join in "
        + str(datetime.timedelta(seconds=round(time.time() - starting_time)))
        + " seconds"
        + "\n"
    )


#%% TEMPLATES
templates = [
    {"file": "patient", "template": patient_template},
    {"file": "practitioner", "template": practitioner_template},
    {"file": "booking-staff", "template": booking_staff_template},
    {"file": "medical-branch", "template": medical_branch_template},
    {"file": "health-service-provision", "template": health_service_provision_template},
    {"file": "appointment-provider", "template": appointment_provider_template},
    {"file": "referral", "template": referral_template},
    {"file": "reservation", "template": reservation_template},
]

#%% INIT


def init():
    start_time = datetime.datetime.now()

    construct_queries(entity_queries, relation_queries, verbose=1)

    entities = []
    for _, v in entity_queries.items():
        entities += v
    entity_processes = []
    print("Inserting entities ...")
    insert_concurrently(entities, entity_processes, n_batch=1000)

    relations = relation_queries["referral"]
    #    for _, v in relation_queries['referral'].items():
    #        relations += v
    print("Inserting referral ...")
    relation_processes = []
    insert_concurrently(relations, relation_processes)
    #    insert(relations, n_batch=1)

    print("Inserting reservation ...")
    relations = relation_queries["reservation"]
    #    for _, v in relation_queries['reservation'].items():
    #        relations += v
    relation_processes = []
    insert_concurrently(relations, relation_processes)
    #    insert(relations, n_batch=1)

    # insert(entities)
    # insert(relations)

    end_time = datetime.datetime.now()
    print("- - - - - -\nTime taken: " + str(end_time - start_time))
    print("\n" + str(len(entity_processes)) + " processes used to insert Entities.")
    print(str(len(relation_processes)) + " processes used to insert Relationships.")


if __name__ == "__main__":
    init()
#    construct_queries(entity_queries, relation_queries, verbose=1)
