# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 13:59:51 2021

@author: p_bud
"""


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/Priyanka_Budavi_Submission_ML/Assignment2/Data/Training Dataset/Train_Image012_NonOverlap.csv')
 
f61 = df['61']
f62 = df['62']
label = df['64']


fig = plt.figure(figsize = (20 , 20))
ax1 = fig.add_subplot(221)
ax1.set_title(" Image012 Non Overlapping Train dataset")
ax1.set_xlabel("Feature 61")
ax1.set_ylabel("Feature 62")
sns.scatterplot(x = f61 , y= f62 , hue= label , palette= "tab10")
plt.show()


