# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 16:08:10 2021

@author: p_bud
"""


import matplotlib.pyplot as plt 
import pandas as pd


#read the fruit
read_banana = pd.read_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/CSV files/Image0_Overlap.csv')
read_apple = pd.read_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/CSV files/Image1_Overlap.csv')
read_guava = pd.read_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/CSV files/Image2_Overlap.csv')

f20 = read_apple['40']


nbins = 40
plt.hist(f20, nbins )
plt.title("Histogram : Image0/ Feature 40 Overlapping ")
plt.xlabel('Feature 40')
plt.ylabel('Values')
plt.show()
