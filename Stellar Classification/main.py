# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 21:05:36 2025

@author: reini
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os 
from model import  StellarClassifier
import torch.optim as optim

parser = argparse.ArgumentParser(description='Stellar Classification')
parser.add_argument('-f', default='', type=str)

#tasks
parser.add_argument('--dataset', type=str, default='star_classification',
                    help='dataset to use (default: stellar _classification)')
parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')

#training paramters
parser.add_argument('--train_size', type=float, default=0.8,
                    help='data used for training model (0:80%)')
parser.add_argument('--val_size', type=float , default=0.9,
                    help='valid percentage for data (train_size%:val_size)')

args = parser.parse_args()

#load data
dataset= os.path.join(args.data_path, args.dataset+".csv")
df= pd.read_csv(dataset)

#preprocess  - not needed since this data is clean

#choose what we will be classifying
label_encoder = LabelEncoder()
df["class"] = label_encoder.fit_transform(df["class"])

#split data

X = df[["u", "g", "r", "i", "z", "redshift"]]
y= df['class']

#scale data
scaler=StandardScaler()
X_scaled = scaler.fit_transform(X)

#split data - we will use 80 10 10
length = len(y)
train_size=int(length*args.train_size)
val_size= int(length*args.val_size)
X_train, y_train = X_scaled[:train_size], y[:train_size]
X_val, y_val = X_scaled[train_size:val_size], y[train_size:val_size]
X_test, y_test = X_scaled[val_size:], y[val_size:]

#convert to tensors
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

#define model, loss function and optimizer
model = StellarClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#train model
for epoch in range(700):
   optimizer.zero_grad()
   outputs = model(X_train_tensor)
   loss = criterion(outputs, y_train_tensor)
   loss.backward()
   optimizer.step()
   print(f"Epoch {epoch+1}, Loss: {loss.item()}")

#check how good model is 
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f"Accuracy: {accuracy * 100:.2f}%")

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generate the confusion matrix
cm = confusion_matrix(y_test, predicted)

# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
