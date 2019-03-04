import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from corner_detector import *
from anms import *
from feat_desc import *
from feat_match import *
from ransac_est_homography import *
from mymosaic import *



#SET 1
im1 = np.array(Image.open('SET1_L.jpg').convert('RGB'))
im2 = np.array(Image.open('SET1_M.jpg').convert('RGB'))
im3 = np.array(Image.open('SET1_R.jpg').convert('RGB'))

#DEFINING THE INPUT IMAGE MATRIX

img_input = np.zeros((3,),dtype='object')
img_input[0] = im1
img_input[1]= im2
img_input[2]= im3

#GETTING THE FINAL MOSAIC
final_mosaic = mymosaic(img_input)


#PLOTTING THE FINAL MOSAIC
plt.imshow(final_mosaic)
plt.show()



'''
##############################################################################
#SET 2
im1 = np.array(Image.open('SET2_L.jpg').convert('RGB'))
im2 = np.array(Image.open('SET2_M.jpg').convert('RGB'))
im3 = np.array(Image.open('SET2_R.jpg').convert('RGB'))

#DEFINING THE INPUT IMAGE MATRIX

img_input = np.zeros((3,),dtype='object')
img_input[0] = im1
img_input[1]= im2
img_input[2]= im3

#GETTING THE FINAL MOSAIC
final_mosaic = mymosaic(img_input)


#PLOTTING THE FINAL MOSAIC
plt.imshow(final_mosaic)
plt.show()
##############################################################################
'''

'''
##############################################################################
#TEST IMAGE
im1 = np.array(Image.open('1_M.jpg').convert('RGB'))
im2 = np.array(Image.open('1_R.jpg').convert('RGB'))

#TEST IMAGE INPUT MATRIX
img_input = np.zeros((2,),dtype='object')
img_input[0] = im1
img_input[1]= im2

#GETTING THE FINAL MOSAIC
final_mosaic = mymosaic(img_input)


#PLOTTING THE FINAL MOSAIC
plt.imshow(final_mosaic)
plt.show()
##############################################################################
'''
