# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 11:18:56 2021

@author: p_bud
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 10:23:21 2021

@author: Priyanka Budavi
"""
import cv2
#import matplotlib.pyplot as plt 

#read  the image
Grayguava = cv2.imread('Grey_Image2.jpg')
#print the original dimensions
print("Original Dimensions:" ,Grayguava.shape )

height = (Grayguava.shape[0])/2
width = (Grayguava.shape[1])/2

print("Halfed height :" , height)
print("Halfed width :" , width)

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
new_width = round(dimesion_div8(height))
new_height = round(dimesion_div8(width))

#Values when both the height and width is to be divisible by 8
print("New Dimension divisible by 8 :" , new_height, new_width )
image_resized = cv2.resize(Grayguava, dsize = (new_height , new_width))
cv2.imwrite('Guava_imagedim_div8.jpg' , image_resized)