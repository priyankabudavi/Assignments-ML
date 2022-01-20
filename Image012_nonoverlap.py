# Databricks notebook source
import pyspark

# COMMAND ----------


df = sqlContext.sql("Select * FROM image012_nonoverlap_5_csv ")

# COMMAND ----------

input_data = df.select("*").toPandas()

# COMMAND ----------

input_data

# COMMAND ----------

input_data.isna().any()

# COMMAND ----------


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
train, test = train_test_split(input_data, test_size=0.2, random_state=123)
X_train = train.drop(["64"], axis=1)
X_test = test.drop(["64"], axis=1)
y_train = train['64']
y_test = test['64']
print(X_train.shape)
print(X_test.shape)

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
randomFClassifier = RandomForestClassifier(random_state=0,n_estimators=1000,oob_score=True, n_jobs=1)
randomFClassifier = randomFClassifier.fit(X_train, y_train)

# COMMAND ----------

y_pred = randomFClassifier.predict(X_test)

# COMMAND ----------

from sklearn.metrics import confusion_matrix
CC_test = confusion_matrix(y_test, y_pred)
print(CC_test)

# COMMAND ----------

pred_prob = randomFClassifier.predict_proba(X_test)

# COMMAND ----------


import matplotlib.pyplot as plt
dims = (3, 4)
f, axes = plt.subplots(dims[0], dims[1], figsize=(25, 15))
axis_i, axis_j = 0, 0
display(sns.boxplot(x=y_train, y=input_data['30'], ax=axes[axis_i, axis_j]))

# COMMAND ----------

from sklearn.metrics import roc_curve, auc
fpr = {}
tpr = {}
threshold = {}
n_classes = 3
roc_auc = {}
for i in range(n_classes):
    fpr[i], tpr[i], threshold[i] = roc_curve(y_test, pred_prob[:, i],pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

# COMMAND ----------

import pandas as pd
feature_importances = pd.DataFrame(randomFClassifier.feature_importances_, index=X_train.columns.tolist(), columns=['importance'])
feature_importances.sort_values('importance', ascending=False)

# COMMAND ----------

from sklearn.metrics import accuracy_score
auccuracy = accuracy_score(y_test, y_pred)
print(auccuracy)

# COMMAND ----------


