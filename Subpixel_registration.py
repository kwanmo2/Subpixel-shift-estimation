# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 09:03:17 2022

@author: kwanm
"""

import numpy as np
import matplotlib as plt
from pandas import Series, DataFrame
import pandas as pd
from skimage import data, io
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
import cv2
from tkinter import filedialog


file_list =  filedialog.askopenfilenames(initialdir ="C://",title = "choose your dark images")
NumberofImage = np.linspace(1,len(file_list),len(file_list),dtype=int)
NumberofImage = len(NumberofImage)
index=np.zeros((NumberofImage,5))

frame = DataFrame(index, columns = ['FileName','X offset','Y offset','error','Rotation'])



for i in range(0,NumberofImage):
    src = cv2.imread(file_list[i],cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR)
    b,g,r = cv2.split(src)    
    image = g
    offset_image = r
    shifted, error, diffphase = register_translation(image, offset_image,1000)
   
    frame.loc[i] = [file_list[i],shifted[1],shifted[0],error,"N/A"]

