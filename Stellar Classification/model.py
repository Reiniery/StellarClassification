# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 22:46:00 2025

@author: reini
"""
import torch
import torch.nn as nn

class StellarClassifier(nn.Module):
    def __init__(self):
        super(StellarClassifier, self).__init__()
        #connected laters
        self.fc1 = nn.Linear(6, 64)  # Input features: 6
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)  # Output classes: 3

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        