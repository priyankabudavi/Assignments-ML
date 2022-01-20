
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
# For classification purposes use RidgeClassifier instead of Ridge
from sklearn.linear_model import RidgeClassifier


# Read a feature space
input_data = pd.read_csv("C:/Desktop/Test1.csv ,header=None")
NN = 2
# Label/Response set
y = input_data[NN]
# Drop the labels and store the features
input_data.drop(NN,axis=1,inplace=True)
X = input_data
# Generate feature matrix using a Numpy array
tmp = np.array(X)
X1 = tmp[:,0:NN] #tmp[:,0:4]
# Generate label matrix using Numpy array
Y1 = np.array(y)
# Split the data into 80:20
row, col = X.shape
TR = round(row*0.8)
TT = row-TR
# Training with 80% data
X1_train = X1[0:TR-1,:]
Y1_train = Y1[0:TR-1]