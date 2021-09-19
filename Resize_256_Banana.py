# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 10:23:21 2021
@author: Priyanka Budavi
"""
import cv2


#read  the image
Gray_fruit = cv2.imread('Grey_Image2.jpg')
#print the original dimensions
print("Original Dimensions:" ,Gray_fruit.shape )
std_height = 256
height = Gray_fruit.shape[0]
width = Gray_fruit.shape[1]

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
image_resized = cv2.resize(Gray_fruit, dsize = (new_height , new_fruit_width))
cv2.imwrite('Gauva_256_resize.jpg' , image_resized)

                           
