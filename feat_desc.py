'''
  File name: feat_desc.py
  Author:
  Date created:
'''

'''
  File clarification:
    Extracting Feature Descriptor for each feature point. You should use the subsampled image around each point feature,
    just extract axis-aligned 8x8 patches. Note that it’s extremely important to sample these patches from the larger 40x40
    window to have a nice big blurred descriptor.
    - Input img: H × W matrix representing the gray scale input image.
    - Input x: N × 1 vector representing the column coordinates of corners.
    - Input y: N × 1 vector representing the row coordinates of corners.
    - Outpuy descs: 64 × N matrix, with column i being the 64 dimensional descriptor (8 × 8 grid linearized) computed at location (xi , yi) in img.
'''
import numpy as np
import scipy.signal as signal
from skimage.measure import block_reduce

def feat_desc(img, x, y):
# Your Code Here
    pts=np.zeros((x.shape[0],2))
    pts[:,0]=x.reshape(-1)
    pts[:,1]=y.reshape(-1)
    pts[:,0]=pts[:,0]+20
    pts[:,1]=pts[:,1]+20

    def GaussianPDF_1D(mu, sigma, length):
        half_len = length / 2
        if np.remainder(length, 2) == 0:
            ax = np.arange(-half_len, half_len, 1)
        else:
            ax = np.arange(-half_len, half_len + 1, 1)
        ax = ax.reshape([-1, ax.size])
        denominator = sigma * np.sqrt(2 * np.pi)
        nominator = np.exp( -np.square(ax - mu) / (2 * sigma * sigma) )
        return nominator / denominator

    def GaussianPDF_2D(mu, sigma, row, col):
        g_row = GaussianPDF_1D(mu, sigma, row)
        g_col = GaussianPDF_1D(mu, sigma, col).transpose()
        return signal.convolve2d(g_row, g_col, 'full')

    def blur(gauss_mtx,img):
        scale_mtx=signal.convolve2d(img,gauss_mtx,'same')
        return scale_mtx

    img_pad=np.pad(img, ((20,20),(20,20)), mode="symmetric")
    npatch=pts.shape[0]
    descriptor=np.zeros((64,npatch))

    for i in range(npatch):
        patch=img_pad[int(pts[i,0]-20):int(pts[i,0]+20),int(pts[i,1]-20):int(pts[i,1]+20)]
        gauss=GaussianPDF_2D(0, 1.4, 5,5)
        patch_blurred=blur(gauss,patch)
        patch_blurred = patch
        sub_patch=block_reduce(patch_blurred,(5,5),np.max).reshape(64)
        sub_patch=(sub_patch-np.mean(sub_patch))/np.std(sub_patch)

        descriptor[:,i]=sub_patch

    

    return descriptor
