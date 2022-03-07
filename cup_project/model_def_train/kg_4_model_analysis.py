# import numpy as np
import os
import pandas as pd
#%%PARAMS

LOG_FILE_NAME = "test_CompeX.log"
KEYSPACE_NAME = "cup_100_1" # cup_100_1 cup_200_2  cup_1000_1
# KEYSPACE_NAME = "cup_1"
#KEYSPACE_NAME = "football_results"
#KEYSPACE_NAME = "phone_calls"
#KEYSPACE_NAME = "GoT"
# KEYSPACE_NAME = "GoT_false"


_local_dir=True

#%% Dirs variables
file_dir = os.path.dirname(__file__)
if _local_dir:
    working_dir = os.path.join(file_dir, os.pardir)
else:
    working_dir = os.path.abspath(os.path.join(file_dir, f"../../../../Pycodes/cup_kg/"))
_model_path = os.path.abspath(os.path.join(working_dir, 'models'))
os.makedirs(_model_path, exist_ok=True)

_log_dir = os.path.join(file_dir, 'logs')   #file_dir



#%% main method definition
def get_model_df(file_name="application.log", log_dir = _log_dir):

    models_log = load_log(log_dir, file_name)
    
    models_df = extract_models(models_log)
    return models_df

#%% tools method definition
def extract_models(models_log):
    
    import re
    subs = "{'batches_count"
    models_list = [x for x in models_log if re.search(subs, x)] 
    
    import ast
    model_types = []; model_output = []; param_dict =[]
    for string in models_list:
        model_types.append(string.split()[0])
        content = ' '.join(string.split()[2:])
        content_splt = content.split('{',1)
        model_output.append(content_splt[0])
        param_dict.append(ast.literal_eval('{'+content_splt[1]))
        
             
    models_df = pd.DataFrame({
            'model_types': model_types,
            'model_output': model_output,
            'param_dict': param_dict
            })
    

    for idx in models_df[models_df['model_types']=='INFO'].index:
        #parse string sequence to dict
        splitted_params = models_df.loc[idx]['model_output'].split(' ')
        dict_preformance_string = str(#' '.join(splitted_params[13:15]) + ' ' +\
                                           ''.join(splitted_params[4:7])+ ', ' + \
                                           ''.join(splitted_params[7:10])+ ', ' + \
                                           ''.join(splitted_params[10:13])  + \
                                           ' '.join(splitted_params[0:2])+ ', ' + \
                                           ' '.join(splitted_params[2:4]) )
        dict_preformance = dict((x.strip(), float(y.strip())) for x, y in (element.split(':') for element in dict_preformance_string.split(','))) 
        models_df.loc[idx]['model_output'] = dict_preformance
  
    print(models_df)
    return models_df

def load_log(log_dir = _log_dir, file_name="application.log"):

    log_file = os.path.join(log_dir, file_name)
    if os.path.isfile(log_file):
        application_log = open(log_file, "r")
        models_log = application_log.read().splitlines()
        application_log.close()
    else:
        models_log = None

    return models_log

#%%




if __name__ == "__main__":
    result = get_model_df(file_name = LOG_FILE_NAME, log_dir = _log_dir)
