# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 22:46:22 2021

@author: p_bud
"""
import cv2
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#read  the image
read_fruit = cv2.imread('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/Screenshot/Image0.jpg')
#covert banana to grayscale
fruit_gray = cv2.cvtColor(read_fruit, cv2.COLOR_BGRA2GRAY)
height , width = fruit_gray.shape
std_height = 256

def width_resize(height , width):
    print("Original Height :", height)
    print("Original Width :", width)
    aspect_ratio = width/height
    new_fruit_width = aspect_ratio * 256
    print("My width :" , new_fruit_width)
    return round(new_fruit_width)
def dimesion_div8(height , width):
      new_fruit_width = width_resize(height , width)
      remainder = new_fruit_width % 8
      if remainder == 0:
         return new_fruit_width
      else:
          new_fruit_width  = new_fruit_width - remainder
          
      return new_fruit_width
#Values when both need to be divisible by 8 
new_fruit_width = dimesion_div8(height,width)
print('Standard Height 256 dimension' , std_height , new_fruit_width)
resized_fruit = cv2.resize(fruit_gray, dsize = (new_fruit_width, std_height))
heightd , widthd = resized_fruit.shape


dd = round((heightd -7 ) * (widthd - 7))
flat_image = np.full((dd, 65), 0)

k = 0 
for i in range( 0 , heightd -7 , 1):
    for j in range(0 , widthd - 7 , 1):
        tmp = resized_fruit[ i:i + 8 , j:j + 8]
        flat_image[k, 0:64] = tmp.flatten()
        k = k + 1
        
feature_space = pd.DataFrame(flat_image)
feature_space.to_csv('Image0_Overlap' , index=False)

