# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 11:17:18 2021

@author: p_bud
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import matplotlib.pyplot as plt


x_train = pd.read_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/Priyanka_Budavi_Submission_ML/Assignment2/Data/Training Dataset/Train_Image012Overlap.csv' , header = None)
x_test = pd.read_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/Priyanka_Budavi_Submission_ML/Assignment2/Data/Testing Dataset/Test_Image012Overlap.csv' , header = None)
    
y = x_train[64]
Y = np.array(y)
x_train.drop(64,axis=1,inplace=True)
X = x_train
X1 = np.array(X)

# Train the model
randomFClassifier = RandomForestClassifier(random_state=0,n_estimators=50,oob_score=True, n_jobs=1)
randomFClassifier = randomFClassifier.fit(X1, Y)


#Testing the model using Trained results0
test_y = x_test[64]
Y_test = np.array(test_y)
x_test.drop(64,axis=1,inplace=True)
x_test_np = np.array(x_test)
y_pred = randomFClassifier.predict(x_test_np)
yhat_saved = pd.DataFrame(y_pred)
#yhat_saved.to_csv('Predicted_012_O.csv', index=False)


CC_test = confusion_matrix(test_y, y_pred)
# 0 as my positive class
TN = CC_test[1,1]
FP = CC_test[1,0]
FN = CC_test[0,1]
TP = CC_test[0,0]

FPFN = FP+FN
TPTN = TP+TN


#Qualitative measures of performance calculation using Confusion matrix
Accuracy = 1/(1+(FPFN/TPTN))
print("Test_Accuracy_Score:",Accuracy)
Precision = 1/(1+(FP/TP))
print("Test_Precision_Score:",Precision)
Sensitivity = 1/(1+(FN/TP))
print("Test_Sensitivity_Score:",Sensitivity)
Specificity = 1/(1+(FP/TN))
print("Test_Specificity_Score:",Specificity)


print("-----------------------------------------------------------")
print("Measures using Library")

print("-----------------------------------------------------------")


#Qualitative measures of performance calculation using sklearn metrics
print('sklearn.metrics Accuracy',accuracy_score(test_y, y_pred))
print('sklearn.metrics precision',precision_score(test_y, y_pred , average='macro'))
print('sklearn.metrics sensitivity',recall_score(test_y, y_pred , average='macro'))




#Entropy measure
import math
-(7/12)*math.log(7/12, 2) - (5/12)*math.log(5/12, 2)


#generate the random forest tree
plt.figure(figsize=(100,100))
tree.plot_tree(randomFClassifier.estimators_[2], filled=True)
#plt.savefig('RandomForest_012_O.pdf') 






