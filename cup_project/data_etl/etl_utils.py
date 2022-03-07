#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:33:39 2019
Set of functions for data loading and cleaning
@author: modal
"""
from cup_project.custom_funcs import benchmark


import os
os.environ["MODIN_ENGINE"] = "ray"  # Modin will use Ray
#os.environ["MODIN_ENGINE"] = "dask"  # Modin will use Dask


#import modin.pandas as pd
#import pandas
#print(pd.__version__)
#print(pandas.__version__)

import pandas as pd
import pickle




@benchmark
def load_dataset(file, chunksize=None, column_types=None, parse_dates=False, sep=','):
    df = pd.DataFrame()
    if chunksize!=None:
        for chunk in pd.read_csv(file, chunksize=chunksize, dtype = column_types, parse_dates = parse_dates, sep=sep):
            df = pd.concat([df, chunk], ignore_index=True)
    
        del chunk
    else:
        df = pd.read_csv(file, dtype = column_types, parse_dates = parse_dates, sep=sep)
    
    return df

def save_on_pickle(df,file_name):
    pickling_on = open(file_name,"wb")
    pickle.dump(df, pickling_on)
    pickling_on.close()
    
def load_pickle(file_name):
    pickle_off = open(file_name,"rb")
    df = pickle.load(pickle_off)
    return df


@benchmark
def load_describe_save(file_name, sep, raw_data_dir, describe_data_dir, interm_data_dir, dates_name=False, column_types=None):
    print(f'Generating Pickle {file_name.upper()}...')
    print(f'Loading {file_name.upper()} dataset...',end='')
    data_frame = load_dataset(file=raw_data_dir / (file_name+'.csv'), chunksize=1000000, column_types = column_types, parse_dates = dates_name, sep=sep)
    print('ended loading')

    print(f'Generating {file_name.upper()} description...',end='')
    description=data_frame.describe(include='all')
    description.to_csv(describe_data_dir/ (file_name+'_describe.csv')) #, sort=True
    description.to_latex(describe_data_dir/ (file_name+'_describe.tex'), index=False)
    print('ended!')

    print(f'Generating Pickle {file_name.upper()}...',end='')
    save_on_pickle(data_frame, interm_data_dir/(file_name+'.pickle'))
    print('ended!')
    del data_frame




