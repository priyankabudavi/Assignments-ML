# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:09:13 2021

@author: p_bud
"""

import pandas as pd
import numpy as np

merged_data = pd.read_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/Priyanka_Budavi_Submission_ML/Assignment3/Data/Image012_Overlap.csv',header=None)


X = merged_data
row, col = X.shape
TR = round(row*0.50)
X1 = np.array(X)
TT = row-TR
X_train = X1[0:TR,:]
X_test = X1[TR:row,:]


data_train = pd.DataFrame(X_train)
data_train.to_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/Priyanka_Budavi_Submission_ML/Assignment3/Data/Reduceddatatrain012_O.csv')
data_test = pd.DataFrame(X_test)
data_test.to_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/Priyanka_Budavi_Submission_ML/Assignment3/Data/Reduceddatatest012_O.csv')