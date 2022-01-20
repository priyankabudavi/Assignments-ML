# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 14:42:25 2021

@author: p_bud
"""

import pandas as pd
import numpy as np

# Read a feature space
input_data = pd.read_csv("C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/Priyanka_Budavi_Submission_ML/Data/CSV files/Image01.csv" , header=None)

#Store the value of the CSC into X 
X = input_data
row, col = X.shape

#Split the data into 80% - Training Dataset 
TR = round(row*0.8)
X1 = np.array(X)

#remaining is the Testing Dataset
TT = row-TR

#Training and Testing Dataset
X_train = X1[0:TR,:]
X_test = X1[TR:row,:]

#savetTraining dataset
data_train = pd.DataFrame(X_train)
data_train.to_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/Priyanka_Budavi_Submission_ML/Assignment2/Data/Train_Image01_NonOverlap.csv')

#save testing Dataset
data_test = pd.DataFrame(X_test)
data_test.to_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/Priyanka_Budavi_Submission_ML/Assignment2/Data/Test_Image01_NonOverlap.csv', index=False)

