# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 18:32:34 2021

@author: p_bud
"""

import pandas as pd
import numpy as np

Image0_read = pd.read_csv('Image01_Overlap.csv') 
Image1_read= pd.read_csv('Image2_Overlap.csv')


frames = [Image0_read , Image1_read]
merged = pd.concat(frames)

indx = np.arange(len(merged))
rndmged = np.random.permutation(indx)
rndmged= merged.sample(frac=1).reset_index(drop=True)
rndmged.to_csv('C:/Users/p_bud/OneDrive/Desktop/Assignment/Big Data ML/Image012_Overlap.csv', index=False)

