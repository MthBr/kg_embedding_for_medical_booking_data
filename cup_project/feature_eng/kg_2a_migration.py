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

from grakn.client import GraknClient

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)


#%% GLOBAL VARIABLES
_cup_datasets_path = os.path.abspath(os.path.join(file_dir, "../../../../Pycodes/csv"))

KEYSPACE_NAME = "cup_1"
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
def entity_template(data):
    query = "insert $x isa " + data["type"]
    for attribute in data["attributes"]:
        if isinstance(attribute["value"], list):
            for elem in attribute["value"]:
                query += ", has " + attribute["type"] + " " + str(elem)
        else:
            query += ", has " + attribute["type"] + " " + str(attribute["value"])
    query += ";"
    return query

# "roleplayers_relation"

def match_roleplayers(data, typ):
    query = ""
    for r, roleplayer in enumerate(data["roleplayers"]):
        query += "$" + str(typ) + str(r) + " isa " + roleplayer["type"]
        for roleplayer_attribute in roleplayer["roleplayer_attributes"]:
            if isinstance(roleplayer_attribute["key_value"], list):
                for b, elem in enumerate(roleplayer_attribute["key_value"]):
                    query += (
                        ", has " + roleplayer_attribute["key_type"] + " " + str(elem)
                    )
                    if b < len(roleplayer_attribute["key_value"]) - 1:
                        query += (
                            "; $"
                            + str(typ) + str(r)
                            + "-"
                            + str(b + 1)
                            + " isa "
                            + roleplayer["type"]
                        )
            else:
                query += (
                    ", has "
                    + roleplayer_attribute["key_type"]
                    + " "
                    + str(roleplayer_attribute["key_value"])
                )
    query += "; "
    return query

def match_relation(data, typ, reltype='x'):
    query = ""
    query += "$"+str(reltype)+" ("
    for r, roleplayer in enumerate(data["roleplayers"]):
        query += roleplayer["role_name"] + ": $" + str(typ) + str(r)

        for roleplayer_attribute in roleplayer["roleplayer_attributes"]:
            if isinstance(roleplayer_attribute["key_value"], list):
                for b, elem in enumerate(roleplayer_attribute["key_value"]):
                    if b < len(roleplayer_attribute["key_value"]) - 1:
                        query += "-" + str(b + 1)
                        query += ", " + roleplayer["role_name"] + ": $" + str(typ) + str(r)

        if r < len(data["roleplayers"]) - 1:
            query += ", "

    if "relation_key_attributes" in data:
        query += ")"
    else:
        query += ") isa " + data["type"]

    if "attributes" in data:
        query += "; $"+str(reltype)
        for a, attribute in enumerate(data["attributes"]):
            query += " has " + attribute["type"] + " " + str(attribute["value"])
            if a < len(data["attributes"]) - 1:
                query += ","
    query += ";"
    return query



def relation_template(data):
    query = "match "
    query += match_roleplayers(data, typ = "e")
    
    if "roleplayers_relation" in data:
        query += match_roleplayers(data, typ = "r")
        query += match_relation(data, typ = "r", reltype='y')
        
        
    # match the relation if required
    if "relation_key_attributes" in data:
        query += "$x"
        for a, relation_key_attribute in enumerate(data["relation_key_attributes"]):
            query += (
                " has "
                + relation_key_attribute["key_type"]
                + " "
                + str(relation_key_attribute["key_value"])
            )
            if a < len(data["relation_key_attributes"]) - 1:
                query += ","
        query += "; "

    query += "insert "
    match_relation(data, typ = "e")
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
                                "key_value": reservation["indx_prestazione"],  # TODO indx_prestazione
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
                                        "key_value": string(reservation["sa_branca_id"]),
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
                        "role_name": "booked-referall",
                    }
                ],
                "attributes": [
                    {
                        "type": "rappointment-encoding",
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

            print("template = {}".format(template))
            INTERVAL = 1000
            bins = np.array_split(inputs, int(np.ceil(len(inputs) / INTERVAL)))
            print("Number of bins = {}".format(len(bins)))
            for count, inputs_part in enumerate(bins):
                t_partial = time.time()
                for ct, row in enumerate(inputs_part):
                    template["template"](row)
                print("{} Ended with time {}".format(count, time.time() - t_partial))
        else:
            for row in inputs:
                template["template"](row)


#%% INSERT
def insert(queries):
    with GraknClient(uri=GRAKN_URI) as client:
        with client.session(keyspace=KEYSPACE_NAME) as session:
            transaction = session.transaction().write()
            for i, query in enumerate(queries):
                transaction.query(query)

                if i % 500 == 0:
                    print(i)
                    print(query)
                    print("- - - - - - - - - - - - - -")
                    transaction.commit()
                    transaction = session.transaction().write()
            transaction.commit()


def insert_concurrently(queries, processes):
    cpu_count = int(multiprocessing.cpu_count()/2)
    chunk_size = int(len(queries) / cpu_count)

    for i in range(cpu_count):
        if i == cpu_count - 1:
            chunk = queries[i * chunk_size :]
        else:
            chunk = queries[i * chunk_size : (i + 1) * chunk_size]

        process = multiprocessing.Process(target=insert, args=(chunk,))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


#%% TEMPLATES
templates = [
    # {"file": "patient", "template": patient_template},
    # {"file": "practitioner", "template": practitioner_template},
    # {"file": "booking-staff", "template": booking_staff_template},
    # {"file": "medical-branch", "template": medical_branch_template},
    # {"file": "health-service-provision", "template": health_service_provision_template},
    # {"file": "appointment-provider", "template": appointment_provider_template},
    # {"file": "referral", "template": referral_template},
    {"file": "reservation", "template": reservation_template},
]

#%% INIT


def init():
    start_time = datetime.datetime.now()

    construct_queries(entity_queries, relation_queries, verbose=0)

    entities, relations = [], []
    for _, v in entity_queries.items():
        entities += v
    for _, v in relation_queries.items():
        relations += v

    entity_processes = []
    relation_processes = []

    insert_concurrently(entities, entity_processes)
    insert_concurrently(relations, relation_processes)

    # insert(entities)
    # insert(relations)

    end_time = datetime.datetime.now()
    print("- - - - - -\nTime taken: " + str(end_time - start_time))
    print("\n" + str(len(entity_processes)) + " processes used to insert Entities.")
    print(str(len(relation_processes)) + " processes used to insert Relationships.")


if __name__ == "__main__":
    # init()
    construct_queries(entity_queries, relation_queries, verbose=1)
