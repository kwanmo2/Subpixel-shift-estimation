# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 09:03:17 2022

@author: kwanm
"""
from matplotlib.pylab import *
import matplotlib.pyplot as plt
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

def sinc_interp(x, s, u):
    if len(x) != len(s):
        raise ValueError('x and s must be the same length')
    
    # Find the period    
    T = s[1] - s[0]
    
    sincM = np.tile(u, (len(s), 1)) - np.tile(s[:, np.newaxis], (1, len(u)))
    y = np.dot(x, np.sinc(sincM/T))
    return y

"""
NumberofImage = np.linspace(1,len(file_list),len(file_list),dtype=int)
NumberofImage = len(NumberofImage)
index=np.zeros((NumberofImage,5))

frame = DataFrame(index, columns = ['FileName','X offset','Y offset','error','Rotation'])



for i in range(0,NumberofImage):
    src = cv2.imread(file_list[i],cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR)
    b,g,r = cv2.split(src)    
    image = g
    offset_image = r
    #Using sckimage module
    shifted, error, diffphase = register_translation(image, offset_image,1000)
   



    frame.loc[i] = [file_list[i],shifted[1],shifted[0],error,"N/A"]
"""
#FFT로 직접 계산

src = cv2.imread(file_list[0],cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR)
b,g,r = cv2.split(src)

reference_IMG = r
h, w = reference_IMG.shape
Target_IMG = g
reference_IMG = np.pad(reference_IMG,((100,100),(100,100)), 'constant', constant_values=0)
Target_IMG = np.pad(Target_IMG,((100,100),(100,100)), 'constant', constant_values=0)

S1 = np.fft.fftshift(np.fft.fft2(reference_IMG))
S2 = np.fft.fftshift(np.fft.fft2(Target_IMG))

Q = (S1 * np.conj(S2)) / abs(S1*conj(S2))
Qi = np.fft.ifft2(Q)

Yp, Xp = np.where(abs(Qi) == np.max(abs(Qi)))

Qi_real = abs(Qi)

deltaX  = (Qi_real[1,0]*Xp + Qi_real[0,0]*1)/(Qi_real[1,0]+Qi_real[0,0]) -1
deltaY  = (Qi_real[0,1]*Yp + Qi_real[0,0]*1)/(Qi_real[0,1]+Qi_real[0,0]) -1

y = sinc_interp(np.array((0,1,2)), np.array((Qi_real[-1,0],Qi_real[0,0],Qi_real[1,0])), 1000)

#참고용
"""
#FFT shift로 이미지 shift하기 .
input_IMG = cv2.imread(file_list[0],cv2.IMREAD_ANYDEPTH);
rows, cols = input_IMG.shape
FFT_IMG = np.fft.fft2(input_IMG)
FFT_shift = np.fft.fftshift(FFT_IMG)
#Spectrum
magnitude_spectrum = 20*np.log(np.abs(FFT_shift))
#Phase
phase = np.imag(FFT_shift)

#Define shifts
x0=-35; 
y0=-50;
xF, yF = np.meshgrid((np.arange((int)(-cols/2),(int)(cols/2),1)),(np.arange((int)(-rows/2),(int)(rows/2),1)))
FFT_shift = FFT_shift*np.exp(-1j*2*np.pi*(xF*x0+yF*y0)/420)
#Spectrum
magnitude_spectrum2 = 20*np.log(np.abs(FFT_shift))
#Phase
phase = np.imag(FFT_shift)

Shifted_Image = np.fft.ifft2(np.fft.ifftshift(FFT_shift))


img_I = Image.fromarray(input_IMG)
img_O = Image.fromarray(abs(np.real(Shifted_Image)).astype(np.uint8))
img_I.show()
img_O.show()
img_O.save("pill_shift.jpg")
plt.subplot(2,1,1)
plt.plot(magnitude_spectrum)
plt.subplot(2,1,2)
plt.plot(magnitude_spectrum2)
"""