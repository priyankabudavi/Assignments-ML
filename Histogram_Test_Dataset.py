# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 22:28:29 2021

@author: p_bud
"""


import matplotlib.pyplot as plt 
import pandas as pd


#read the file
input_data = pd.read_csv("C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/Priyanka_Budavi_Submission_ML/Assignment2/Data/Testing Dataset/Test_Image012_NonOverlap.csv")

f61 = input_data['61']  
f62 = input_data['62']  
  

nbins = 40
plt.title('Histogram of  Non Overlapping Feature 61 Test data set/image012')
plt.hist(f61,nbins ,  color='g', edgecolor='k')
plt.axvline(f61.mean(), color='k', linestyle='dashed', linewidth=1)
plt.text(f61.mean(), 60 ,r'mean')
plt.show()


plt.title('Histogram of Non Overlapping Feature 62 Test data set/image012')
plt.hist(f62 ,nbins ,  color='g', edgecolor='k')
plt.axvline(f62.mean(), color='k', linestyle='dashed', linewidth=1)
plt.text(f62.mean(), 60 ,r'mean')
plt.show()