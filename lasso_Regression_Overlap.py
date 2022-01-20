

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 08:13:24 2021

@author: p_bud
"""


import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score

#read the training and the testing dataset
x_train = pd.read_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/Priyanka_Budavi_Submission_ML/Assignment2/Data/Training Dataset/Train_Image012Overlap.csv' , header = None)
x_test = pd.read_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/Priyanka_Budavi_Submission_ML/Assignment2/Data/Testing Dataset/Test_Image012Overlap.csv', header=None)


#read the label and store in y
y = x_train[64]
Y = np.array(y)

#drop the label column 
x_train.drop(64,axis=1,inplace=True)
X = x_train
X1 = np.array(X)

#value at 150 the accuracy decreases , thus max value of lamda can be 150
lamda = 0.01

# X' 
X2 = X1.transpose()

#XX'
XX_transpose = np.matmul( X2 , X1)

# XX' inverse
IX = inv(XX_transpose)

#Ydash = Y.transpose()
ymulX = np.matmul(X2, Y)

# Inverse of the square of the feature matrix
first_parameter = np.matmul(ymulX,IX)

S= np.sign(first_parameter)
second_parameter = (S*(lamda/2))

sub_value = ymulX-second_parameter
A = np.matmul(IX , sub_value)

y_hat = np.matmul(X1, A)
ZZ2 = y_hat > y_hat.mean()
yhatTrain = ZZ2.astype(int)

# Save the predicted values
yhatTrain_saved = pd.DataFrame(yhatTrain)
#yhatTrain_saved.to_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/Priyanka_Budavi_Submission_ML/Assignment2/Data/Training Dataset/LassoTrain_Image01_Overlap_result.csv', index=False)

#Confusion matrix
CC = confusion_matrix(Y, yhatTrain)
TN = CC[1,1]
FP = CC[1,0]
FN = CC[0,1]
TP = CC[0,0]
FPFN = FP+FN
TPTN = TP+TN


Accuracy = 1/(1+(FPFN/TPTN))
print("Train_Accuracy_Score:",Accuracy)
Precision = 1/(1+(FP/TP))
print("Train_Precision_Score:",Precision)
Sensitivity = 1/(1+(FN/TP))
print("Train_Sensitivity_Score:",Sensitivity)
Specificity = 1/(1+(FP/TN))
print("Train_Specificity_Score:",Specificity)
print('------------------------------------')

#Testing the model
test_y = x_test[64]
Y_test = np.array(test_y)
x_test.drop(64,axis=1,inplace=True)
x_test_np = np.array(x_test)

Z1_test = np.matmul(x_test_np, A)
Z2_test = Z1_test > Z1_test.mean()

# Test data accuracy
yhat_test = Z2_test.astype(int)
yhat_saved = pd.DataFrame(yhat_test)
yhat_saved.to_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/Priyanka_Budavi_Submission_ML/Assignment2/Data/Testing Dataset/LassoTest_Image01_Overlap_result.csv', index=False)

CC_test = confusion_matrix(test_y, yhat_test)
TN = CC_test[0,0]
FP = CC_test[0,1]
FN = CC_test[1,0]
TP = CC_test[1,1]
FPFN = FP+FN
TPTN = TP+TN


Accuracy = 1/(1+(FPFN/TPTN))
print("Test_Accuracy_Score:",Accuracy)
Precision = 1/(1+(FP/TP))
print("Test_Precision_Score:",Precision)
Sensitivity = 1/(1+(FN/TP))
print("Test_Sensitivity_Score:",Sensitivity)
Specificity = 1/(1+(FP/TN))
print("Test_Specificity_Score:",Specificity)
print('------------------------------------')


#Inbuilt performance metrics
print('sklearn.metrics Accuracy',accuracy_score(test_y, yhat_test ))
print('sklearn.metrics precision',precision_score(test_y, yhat_test, average='macro'))
print('sklearn.metrics sensitivity',recall_score(test_y, yhat_test , average='macro'))
