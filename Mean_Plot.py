# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 10:27:30 2021

@author: p_bud
"""

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt 
import pandas as pd

#read the csv file 
read_banana = pd.read_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/CSV files/Image0.csv')
read_apple = pd.read_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/CSV files/Image1.csv')
read_guava = pd.read_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/CSV files/Image2.csv')

#calculate the mean of the CSV files 
mean_banana =  round(read_banana.mean())
mean_apple =  round(read_apple.mean())
mean_guava =  round(read_guava.mean())

a =print(mean_banana)
b =print(mean_apple)
c =print(mean_guava)

#Plot the mean values
plt.plot(mean_banana, 'r')
plt.plot(mean_apple, 'b')
plt.plot(mean_guava, 'g')

#Setting the Co-ordinate values
plt.title( 'Mean Plot Non-Overlapping Set' )
plt.xlabel('Features')
plt.ylabel('Mean Values')

