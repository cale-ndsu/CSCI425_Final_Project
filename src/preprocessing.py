'''
CSCI425 Final Project
Analyzing Traffic Disruptions in Relation to Cold Temperatures
By Maison Kasprick and Cale Voglewede

preprocessing.py
Responsible for preparing data for machine learning analysis.
'''

import pandas as pd
import os
import sys

def preprocessing():    
    DATA_FILE_PATH = '../data/US_Accidents_March23_sampled_500k.csv'
    BASE_PATH = os.path.join(os.path.dirname(__file__), '.')
    
    os.chdir(BASE_PATH)

    if ((os.path.isfile(DATA_FILE_PATH)) == False):
        print(f"Error: The file '{DATA_FILE_PATH}' was not found.")
        sys.exit()
        
    data_file = pd.read_csv(DATA_FILE_PATH)
    
    # filter to independent variable and dependent variable desired
    data_file = data_file[["Temperature(F)","Severity"]]
    
    # split the dataset into two datasets
    data_cold = data_file[data_file['Temperature(F)'] <= 32]
    data_warm = data_file[data_file['Temperature(F)'] > 32]

    data_container = []
    data_container.append(data_cold)
    data_container.append(data_warm)
    return data_container