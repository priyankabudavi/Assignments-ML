# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 12:00:09 2021

@author: p_bud
"""

import matplotlib.pyplot as plt 
import pandas as pd


#read the csv file 
read_fruit_one = pd.read_csv('Image0.csv')
read_fruit_two = pd.read_csv('Image1.csv')


f20 = read_fruit_one['20']
f40 = read_fruit_two['40']


plt.scatter(f20, f40, color =['red'] , s=1 )
plt.title("Feature Plot Banana/Apple")
plt.xlabel('Feature 20')
plt.ylabel('Feature 40')
plt.show()

