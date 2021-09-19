# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 14:22:14 2021

@author: p_bud
"""


import matplotlib.pyplot as plt
import pandas as pd
 

read_banana = pd.read_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/CSV files/Image0.csv')
read_apple = pd.read_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/CSV files/Image1.csv')
read_gauava = pd.read_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/CSV files/Image2.csv')

f10 = read_banana['10']
f20 = read_banana['20']
f40 = read_banana['40']

A10 = read_apple['10']
A20 = read_apple['20']
A40 = read_apple['40']
 
G10 = read_gauava['10']
G20 = read_gauava['20']
G40 = read_gauava['40']
 
# Creating figure
fig = plt.figure(figsize = (40, 7))
ax = plt.axes(projection ="3d")
 
# Creating plot
ax.scatter3D(f10, f20, f40, color = "red")
ax.scatter3D(A10, A20, A40, color = "green")
ax.scatter3D(G10,G20, G40, color = "blue")

plt.title("3D Scatter Plot of Banana, Apple, Guava")
plt.xlabel('Feature 10')
plt.ylabel('Feature 20')
ax.set_zlabel('Feature 40')
plt.show()