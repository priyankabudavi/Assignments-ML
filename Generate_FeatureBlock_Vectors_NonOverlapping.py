# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 08:41:44 2021

@author: p_bud
"""
import cv2
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#read  the image
read_fruit = cv2.imread('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/Screenshot/Image1.jpg')
#covert banana to grayscale
fruit_gray = cv2.cvtColor(read_fruit, cv2.COLOR_BGRA2GRAY)
height , width = fruit_gray.shape
std_height = 256

print("Dimension of Gray image: ", height , width)

#resize function 
def width_div8(height , width):
    print("Original Height :", height)
    print("Original Width :", width)
    aspect_ratio = round(height/width)
   
    new_fruit_width = aspect_ratio * 256
    print("My width :" , new_fruit_width)

    return round(new_fruit_width)

#function for Dividing by 8
def dimesion_div8(dimensions_value):
      remainder = dimensions_value % 8
      if remainder == 0:
          dimensions_value = dimensions_value
          #print("Value visible by 8 :" , dimensions_value)
      else:
           dimensions_value = dimensions_value + (8-remainder)
           #print("Value disible by 8 :", dimensions_value)
          
      return dimensions_value
#Values when both need to be divisible by 8 
new_fruit_width = width_div8(height,width)
new_height = std_height

#Values when both the height and width is to be divisible by 8
print("New Dimension divisible by 8 :" , new_height, new_fruit_width )
image_resized = cv2.resize(fruit_gray, dsize = (new_height , new_fruit_width))
cv2.imwrite('Fruit_resize.jpg' , image_resized)

#code for generating non overlapping blocks
dd = round((new_height) * (new_fruit_width)/64)
print(dd)
flat_image = np.full((dd , 65) , 1)
k=0

for i in range(0 , new_height, 8):
    for j in range(0, new_fruit_width ,8):
        tmp = image_resized[i : i+8 , j: j+8]
        print(tmp)
        flat_image[k,0:64] = tmp.flatten()
        k = k+ 1
    
feature_space = pd.DataFrame(flat_image)
feature_space.to_csv('Image1.csv' , index=False)
    



