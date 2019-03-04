
import numpy as np
from est_homography import est_homography


def ransac_est_homography(x1, y1, x2, y2, thresh):


  count = 0
  max = 1000


  '''
  IF THE TEST IMAGE IS BEING RUN, THEN THE MAX COUNT NEEDS TO BE CHANGED FROM 1000 TO 10000

  max = 10000
    '''


  l = x1.shape[0]

  set1 = np.zeros((l,3))
  set2 = np.zeros((l,3))

  set1[:,2] = 1
  set2[:,2] = 1

  set1[:,0] = np.reshape(x1,(x1.shape[0],))
  set1[:,1] = np.reshape(y1,(y1.shape[0],))

  set2[:,0] = np.reshape(x2,(x2.shape[0],))
  set2[:,1] = np.reshape(y2,(y2.shape[0],))

  set_tot = np.hstack([set1,set2])
  set_tot1 = np.hstack([set1,set2])


  inlier_ind = np.zeros((x1.shape[0],1))
  ic = 0
  set_tot = np.array(set_tot, dtype='int')
  set_tot1 = np.array(set_tot, dtype='int')
  set1 = np.array(set1, dtype = 'int')
  set2 = np.array(set2, dtype= 'int')

  while count<max:
      in_tmp = np.zeros((x1.shape[0],1))
      np.random.shuffle(set_tot)
      x = set_tot[0:4,0]
      y = set_tot[0:4,1]
      X = set_tot[0:4,3]
      Y = set_tot[0:4,4]
      H_temp = est_homography(x,y,X,Y)

      t1 = np.matmul(H_temp,(np.transpose(set_tot1[:,0:3])))
      t1 = t1/t1[2]

      error =  np.sqrt((t1[0]-set_tot1[:,3])**2+(t1[1]-set_tot1[:,4])**2)
      in_tmp[error<thresh,0] = 1
      in_idx = np.where(in_tmp==1)
      in_count = in_idx[0].shape[0]
      if in_count>ic:
          H = H_temp
          inlier_ind = in_tmp
          ic = in_count
      count += 1

  if ic>4:
      pos = np.where(inlier_ind==1)
      pos = pos[0]
      H = est_homography(set_tot1[pos,0],set_tot1[pos,1],set_tot1[pos,3],set_tot1[pos,4])
  else:
      print('Not enough inliers')

  return H, inlier_ind
