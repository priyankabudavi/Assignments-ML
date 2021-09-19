# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 10:19:05 2021

@author: p_bud
"""

import cv2
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


std_height = 256
#resize function 
def width_div8(height , width):
    print("Original Height :", height)
    print("Original Width :", width)
    aspect_ratio = round(width/height)
   
    new_fruit_width = aspect_ratio * std_height
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
          
def read_multiple_files( image , filename):
    fruit_gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    height , width = fruit_gray.shape
    print("Dimension of Gray image: ", height , width)     
    new_fruit_width = width_div8(height,width)

    #Values when both the height and width is to be divisible by 8
    print("New Dimension divisible by 8 :" , std_height, new_fruit_width )
    image_resized = cv2.resize(fruit_gray, dsize = (std_height , new_fruit_width))
    heightd , widthd = image_resized.shape
    dd = round((heightd) * (widthd)/64)
    print(dd)
    flat_image = np.full((dd , 65) , 1)
    k=0
    for i in range(0 , heightd, 8):
        for j in range(0, widthd ,8):
          tmp = image_resized[i : i+8 , j: j+8]
        print(tmp)
        flat_image[k,0:64] = tmp.flatten()
        k = k+ 1
    
    feature_space = pd.DataFrame(flat_image) 
    feature_space.to_csv(filename, index=False)

    return feature_space


folder_path = 'C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/Priyanka_Budavi_Submission_ML/Data/FIDS30/apples' 
images = []
for imagename in os.listdir(folder_path):
 img = cv2.imread(os.path.join(folder_path,imagename))
 if img is not None:
    images.append(img) 
    splitted_filename = imagename.split('.')
    rename_csvfile = 'AppleDataset'+splitted_filename[0]+'.csv'
    read_multiple_files(img,rename_csvfile)
       

