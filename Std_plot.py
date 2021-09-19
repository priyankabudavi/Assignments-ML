# 
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 10:27:30 2021

@author: p_bud
"""

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

#read the csv file 
read_banana = pd.read_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/Image0_Overlap.csv')
read_apple = pd.read_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/Image1_Overlap.csv')
read_guava = pd.read_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/Image2_Overlap.csv')

#calculate the mean of the CSV files 
std_banana =  round(read_banana.std())
std_apple =  round(read_apple.std())
std_guava =  round(read_guava.std())

print(" Banana ", std_banana)
print(" Apple ", std_apple)
print(" Guava ", std_guava)

Corr = np.cov(std_banana , std_apple)
print("CoV:" , Corr)

#Plot the mean values
plt.plot(std_banana, 'r')
plt.plot(std_apple, 'b')
plt.plot(std_guava, 'g')

#Setting the Co-ordinate values
plt.title( 'Standard Deviation Plot Overlapping Set' )
plt.xlabel('Features')
plt.ylabel('Standard Deviation Values')


    