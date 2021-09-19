# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 10:24:31 2021

@author: p_bud
"""


import matplotlib.pyplot as plt 
import pandas as pd


#read the csv file 
read_banana = pd.read_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/CSV files/Image0.csv')
read_banana = pd.read_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/CSV files/Image0.csv')

read_apple = pd.read_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/CSV files/Image1.csv')
read_apple = pd.read_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/CSV files/Image1.csv')


f20 = read_banana['20']
f40 = read_banana['40']

G20 = read_apple['20']
G40 = read_apple['40']


plt.scatter(f20, f40, color =['red'] , s=1 )
plt.scatter(G20, G40, color =['green'] , s=1 )

plt.title("Feature Plot Banana/Apple")
plt.xlabel('Feature 20')
plt.ylabel('Feature 40')
plt.show()
