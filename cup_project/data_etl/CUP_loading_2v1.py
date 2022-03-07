#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:33:39 2019

@author: modal
"""

import pandas as pd
import pickle

def load_dataset(file, chunksize=None, column_types=None, parse_dates=False):
    df = pd.DataFrame()
    if chunksize!=None:
        for chunk in pd.read_csv(file, chunksize=chunksize, dtype = column_types, parse_dates = parse_dates):
            df = pd.concat([df, chunk], ignore_index=True)
    
        del chunk
    else:
        df = pd.read_csv(file, dtype = column_types, parse_dates = parse_dates)
    
    return df

def save_on_pickle(df,file_name):
    pickling_on = open(file_name,"wb")
    pickle.dump(df, pickling_on)
    pickling_on.close()
    
def load_pickle(file_name):
    pickle_off = open(file_name,"rb")
    df = pickle.load(pickle_off)
    return df
