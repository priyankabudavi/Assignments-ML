import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import time

x_train = pd.read_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/Priyanka_Budavi_Submission_ML/Assignment3/Data/Local RF Data/trainoverlap012.csv',header=None)
x_test = pd.read_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/Priyanka_Budavi_Submission_ML/Assignment3/Data/Local RF Data/testoverlap012.csv',header=None)

    
y = x_train[64]
Y = np.array(y)
x_train.drop(64,axis=1,inplace=True)
X = x_train
X1 = np.array(X)

#Training the model
t0= time.time()
randomFClassifier = RandomForestClassifier(random_state=0,n_estimators=1000,oob_score=True, n_jobs=1)
randomFClassifier = randomFClassifier.fit(X1, Y)
t1 = time.time() - t0
print("Time elapsed: ", t1)

importance = randomFClassifier.feature_importances_
indices = importance.argsort()[::-1]

oob_error = 1- randomFClassifier.oob_score_

#Testing the model using Trained results0
test_y = x_test[64]
Y_test = np.array(test_y)
x_test.drop(64,axis=1,inplace=True)
x_test_np = np.array(x_test)
y_pred = randomFClassifier.predict(x_test_np)
#yhat_saved = pd.DataFrame(y_pred)
#yhat_saved.to_csv('C:/Users/deepa/Desktop/BigData_ML/Assignment2/files/Randomforest/overlap/predicted_01.csv', index=False)
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
 
#plt.figure(figsize=(100,100))
#tree.plot_tree(randomFClassifier.estimators_[2], filled=True)
#plt.savefig('RandomForest_01_nonO.pdf') 
#dot_data = tree.export_graphviz(randomFClassifier, out_file=None) 
#graph = graphviz.Source(dot_data) 
#graph.render('C:/Users/deepa/Desktop/BigData_ML/Assignment2/files/Randomforest/nonoverlap/tree')

#Qualitative measures of performance calculation using sklearn metrics
print('sklearn.metrics Accuracy', accuracy_score(test_y, y_pred))
print('sklearn.metrics precision', precision_score(test_y, y_pred ))
print('sklearn.metrics sensitivity', recall_score(test_y, y_pred)) 
#Entropy measure
import math
-(7/12)*math.log(7/12, 2) - (5/12)*math.log(5/12, 2)
