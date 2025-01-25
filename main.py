# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 20:15:16 2025

@author: reini
"""

import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

import os 

#arguements
parser = argparse.ArgumentParser(description='Stellar Classification')
parser.add_argument('-f', default='', type=str)
parser.add_argument('--data_path',type=str, default="data", 
                    help="Data location")
parser.add_argument('--dataset',type=str, default="star_classification.csv", 
                    help="Data file")
args = parser.parse_args()


#Get dataset
dataset= os.path.join(args.data_path, args.dataset)

df= pd.read_csv(dataset)
mini_df=df[0:100] #for testing

#plots
plt.scatter(mini_df['alpha'],mini_df['delta'])
plt.plot()
plt.show()


