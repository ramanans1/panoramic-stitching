
import skimage.feature
import matplotlib.pyplot as plt
import numpy as np
def corner_detector(img):

  img_gray=0.2989*(img[:,:,0])+0.5870*(img[:,:,1])+0.1140*(img[:,:,2])
  img_gray=img_gray.astype(np.uint8)
  cimg=skimage.feature.corner_harris(img_gray)

  '''
  IF THE TEST IMAGE IS BEING USED, THEN THE FEATURE HAS TO BE CHANGED FROM HARRIS TO SHI_TOMASI

  cimg = skimage.feature.corner_shi_tomasi(img_gray)
  '''


  return cimg
