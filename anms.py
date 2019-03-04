'''
  File name: anms.py
  Author:
  Date created:
'''

'''
  File clarification:
    Implement Adaptive Non-Maximal Suppression. The goal is to create an uniformly distributed
    points given the number of feature desired:
    - Input cimg: H × W matrix representing the corner metric matrix.
    - Input max_pts: the number of corners desired.
    - Outpuy x: N × 1 vector representing the column coordinates of corners.
    - Output y: N × 1 vector representing the row coordinates of corners.
    - Output rmax: suppression radius used to get max pts corners.
'''
import numpy as np
from scipy.spatial import distance
def anms(harris, max_pts):
  # Your Code Here
    harris_pts=np.zeros((np.where(harris>0)[0].shape[0],2)).astype(np.int32)
    harris_pts[:,0]=np.where(harris>0)[0].reshape(-1).astype(np.int32)
    harris_pts[:,1]=np.where(harris>0)[1].reshape(-1).astype(np.int32)
    harris_val=harris[harris_pts[:,0],harris_pts[:,1]].reshape(harris_pts[:,0].shape[0],1)

    non_max=[]
    for x in range(harris_pts.shape[0]):
        supp_radius=np.Inf
        xi,yi=harris_pts[x][0],harris_pts[x][1]
        for y in range(harris_pts.shape[0]):
            xj,yj=harris_pts[y][0],harris_pts[y][1]
            if(xi!=xj and yi!=yj and harris_val[x]< 0.9*harris_val[y]):
                dist=distance.euclidean((xi,yi),(xj,yj))
                if dist<supp_radius:
                    supp_radius=dist
        non_max.append([xi,yi,dist])
    non_max.sort(key=lambda x:x[2],reverse=True)
    top_max_pts=non_max[:max_pts]
    top_max_pts=np.array(top_max_pts).reshape(len(top_max_pts),3).astype(np.int)
    x=top_max_pts[:,0].reshape(len(top_max_pts),1)
    y=top_max_pts[:,1].reshape(len(top_max_pts),1)
    r_max=top_max_pts[:,2].reshape(len(top_max_pts),1)


    return x,y,r_max
