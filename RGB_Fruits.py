# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 11:51:12 2021

@author: Priyanka Budavi
"""

import cv2
import matplotlib.pyplot as plt 

#read  the image
banana_image = cv2.imread('Image0.jpg')
apple_image = cv2.imread('Image1.jpg')
Guava_image = cv2.imread('Image2.jpg')

#print the dimensions of the image
print (banana_image.shape)
print (apple_image.shape)
print (Guava_image.shape)

#RGB for banana
plt.imshow(banana_image[:,:,0])
plt.imshow(banana_image[:,:,1])
plt.imshow(banana_image[:,:,2])

#RGB for apple 
plt.imshow(apple_image[:,:,0])
plt.imshow(apple_image[:,:,1])
plt.imshow(apple_image[:,:,2])

#RGB for Gauva
plt.imshow(Guava_image[:,:,0])
plt.imshow(Guava_image[:,:,1])
plt.imshow(Guava_image[:,:,2])
