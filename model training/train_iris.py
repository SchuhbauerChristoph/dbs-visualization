# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import utils.models as models
from Training_NN import Trainer
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from sklearn.datasets import load_iris

def determine_uncertain_values(complete_data, p):
    shape = complete_data.shape
    copy_ds = complete_data.copy()
    copy_ds = copy_ds.astype('float')
    uncertain_data = np.random.binomial(1, p, shape)
    copy_ds[uncertain_data.astype('bool')] = np.nan
    return copy_ds

# load data
# Hier für andere Datensätze oft andere Ladevorgänge. Oft als csv Datei vorhanden.
# Dann hier entsprechend ändern.

data = load_iris()

X = data['data']
y = data['target']

# Diese Speichervorgänge möglichst unberührt lassen. Dieses Format wird genau so
# im Hauptcode wieder geladen.

with open('../datasets/iris/data/iris_data.npy', 'wb') as f:
    np.save(f, X)
    np.save(f, y)


# scale and split data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# save the split
with open('../datasets/iris/data/iris_data_train.npy', 'wb') as f:
    np.save(f, X_train)
    np.save(f, y_train)
with open('../datasets/iris/data/iris_data_test.npy', 'wb') as f:
    np.save(f, X_test)
    np.save(f, y_test)


# train and save the model
# Hier einzustellen: Epochenanzahl des Trainings. Name des Models beim Speichern. 
# "model_number" legt fest welches der Models aus utils/models.py verwendet werden soll.
# Dort kann dann Input- und Outputdimension sowie Anzahl und Schichten Neuronen festgelegt werden.
trainer = Trainer(epochs=1000, model_name="model_iris", model_number = "4") 
trainer.initialize_data(train_data = (X_train, y_train), test_data = (X_test, y_test))
trainer.prepare_for_training()
trainer.train_and_save()
print(f"Accuracy auf Testdaten:", trainer.test())