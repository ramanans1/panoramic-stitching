
import numpy as np
from scipy.spatial.distance import cdist


def feat_match(descs1, descs2):


    d1 = descs1
    d2 = descs2

    d1 = np.transpose(d1)

    d2 = np.transpose(d2)

    comp_dist = cdist(d1,d2)
    srted1 = np.argsort(comp_dist,1)[:,0]     #index of best comparison
    srted2 = np.argsort(comp_dist,1)[:,1]     #index of second best comparison

    ratio = comp_dist[np.arange(d1.shape[0]),srted1]/comp_dist[np.arange(d1.shape[0]),srted2]

    matches = np.hstack([np.argwhere(ratio < 0.9),srted1[np.argwhere(ratio<0.9)]]).astype(int)

    '''
    IF THE TEST IMAGE IS BEING USED, THEN THE RATIO HAS TO BE CHANGED FROM 0.9 TO 0.95

    matches = np.hstack([np.argwhere(ratio < 0.95),srted1[np.argwhere(ratio<0.95)]]).astype(int)
    '''


    match = np.ones((d1.shape[0],1),dtype='int')*(-1)
    match[matches[:,0],0] = matches[:,1]

    return match
