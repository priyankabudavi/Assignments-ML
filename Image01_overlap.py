# Databricks notebook source
import pyspark

# COMMAND ----------

df = sqlContext.sql("SELECT * FROM image01_overlap_csv")
input_data = df.select("*").toPandas()

# COMMAND ----------

import seaborn as sns
display(sns.distplot(input_data['30'],kde=False))

# COMMAND ----------

import seaborn as sns
display(sns.distplot(input_data['60'],kde=False))

# COMMAND ----------

input_data.isna().any()

# COMMAND ----------

from sklearn.model_selection import train_test_split
 
train, test = train_test_split(input_data,test_size=0.2, random_state=123)
X_train = train.drop(["64"], axis=1)
X_test = test.drop(["64"], axis=1)
y_train = train['64']
y_test = test['64']


# COMMAND ----------


import matplotlib.pyplot as plt
dims = (3, 4)
f, axes = plt.subplots(dims[0], dims[1], figsize=(25, 15))
axis_i, axis_j = 0, 0
display(sns.boxplot(x=y_train, y=input_data['30'], ax=axes[axis_i, axis_j]))

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
randomFClassifier = RandomForestClassifier(random_state=0,n_estimators=100,oob_score=True, n_jobs=1)
randomFClassifier = randomFClassifier.fit(X_train, y_train)

# COMMAND ----------

y_pred = randomFClassifier.predict(X_test)

# COMMAND ----------

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
auc_score = roc_auc_score(y_test, y_pred)
fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, marker='.', label='Random forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# COMMAND ----------


predict_probabilities = randomFClassifier.predict_proba(X_test)
fprp, tprp, thresholdp = roc_curve(y_test, predict_probabilities[:,1])
plt.plot(fprp, tprp, marker='.', label='Random forest probabilty')


# COMMAND ----------

importance = randomFClassifier.feature_importances_
indices = importance.argsort()[::-1]
print(importance)

# COMMAND ----------

from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(y_test, y_pred)
print(auc_score)

# COMMAND ----------


from sklearn.metrics import accuracy_score
print('sklearn.metrics Accuracy',accuracy_score(y_test, y_pred))

# COMMAND ----------

from sklearn.metrics import confusion_matrix
CC_test = confusion_matrix(y_test, y_pred)
print(CC_test)
