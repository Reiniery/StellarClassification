# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 20:15:16 2025

@author: reini
"""

import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

import os 

#arguements
parser = argparse.ArgumentParser(description='Stellar Classification')
parser.add_argument('-f', default='', type=str)
parser.add_argument('--data_path',type=str, default="data", 
                    help="Data location")
parser.add_argument('--dataset',type=str, default="star_classification", 
                    help="Data file")
args = parser.parse_args()


#Get dataset
dataset= os.path.join(args.data_path, args.dataset+".csv")

df= pd.read_csv(dataset)
mini_df=df[0:5000]#for testing

# scatter plots - Locate Position of observed object
   
colors = {'GALAXY': 'blue', 'STAR': 'yellow', 'QSO': 'green'} # define each color by class
plt.figure(figsize=(8, 6))
for obj_class, group in mini_df.groupby('class'):
    plt.scatter(group['alpha'], group['delta'], label=obj_class, color=colors[obj_class])


plt.xlabel('Right Ascension (alpha)')
plt.ylabel('Declination (delta)')
plt.title('Scatter Plot of Celestial Objects by Class')
plt.legend(title='Object Class')
plt.grid()
plt.show()
#bar graph of types ofd observations
categories = ['GALAXY', 'STAR', 'QSO']

num_classes = df['class'].value_counts()
plt.bar(categories, num_classes)
plt.title("Data Bar Graph")

#correlation heatmap
numeric_df = mini_df[["u", "g", "r", "i", "z", "redshift"]]
corr_matrix = numeric_df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="viridis", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


