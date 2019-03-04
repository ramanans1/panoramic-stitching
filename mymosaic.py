
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from corner_detector import *
from anms import *
from feat_desc import *
from feat_match import *
from ransac_est_homography import *
from skimage.feature import plot_matches



def mymosaic(img_input):


    im1 = img_input[0]

    for i in np.arange(1,img_input.shape[0],1):

        im2 = img_input[i]

        im1_c=corner_detector(im1)
        im2_c=corner_detector(im2)


        im1_c[im1_c<0.29]=0
        im2_c[im2_c<0.2]=0

        #Adaptive Non Maximum suppresion
        x_im1,y_im1,r_max_im1=anms(im1_c,max_pts=500)
        x_im2,y_im2,r_max_im2=anms(im2_c,max_pts=500)

        #######################################################################
        ############### PLOTTING FEATURE POINTS AFTER ANMS ##################
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(im1, interpolation='nearest', cmap=plt.cm.gray)
        ax1.plot(y_im1, x_im1, '.b', markersize=10)
        ax2.imshow(im2, interpolation='nearest', cmap=plt.cm.gray)
        ax2.plot(y_im2,x_im2, '.b', markersize=5)

        plt.show()


        #######################################################################


        #Feature_Descriptor
        im1_gray=0.2989*(im1[:,:,0])+0.5870*(im1[:,:,1])+0.1140*(im1[:,:,2])
        im1_gray=im1_gray.astype(np.uint8)
        im1_desc=feat_desc(im1_gray,x_im1,y_im1)

        im2_gray=0.2989*(im2[:,:,0])+0.5870*(im2[:,:,1])+0.1140*(im2[:,:,2])
        im2_gray=im2_gray.astype(np.uint8)
        im2_desc=feat_desc(im2_gray,x_im2,y_im2)

        #Feature Matching
        match=feat_match(im1_desc, im2_desc)

        #Ransac
        lpos=np.arange(match.shape[0]).reshape(match.shape[0],1)
        lpos=lpos[match!=-1]
        x_1,y_1=x_im1[lpos],y_im1[lpos]
        x_2,y_2=x_im2[match[match!=-1]],y_im2[match[match!=-1]]
        Homo_lm,inlier_idx=ransac_est_homography(x_1, y_1,x_2, y_2, 0.5)
        inlier_idx=inlier_idx.astype(np.int32)

        x_1_post=x_1*inlier_idx
        x1_final=x_1_post[x_1_post>0]
        x1_final=x1_final.reshape(-1,1)

        y_1_post=y_1*inlier_idx
        y1_final=y_1_post[y_1_post>0]
        y1_final=y1_final.reshape(-1,1)

        x_2_post=x_2*inlier_idx
        x2_final=x_2_post[x_2_post>0]
        x2_final=x2_final.reshape(-1,1)

        y_2_post=y_2*inlier_idx
        y2_final=y_2_post[y_2_post>0]
        y2_final=y2_final.reshape(-1,1)

        ######################################################################
        ############## PLOTTING MATCHED FEATURES AFTER RANSAC #################

        l = x_1.shape[0]

        set1 = np.zeros((l,2))
        set2 = np.zeros((l,2))

        set1[:,0] = np.reshape(x_1,(x_1.shape[0],))
        set1[:,1] = np.reshape(y_1,(y_1.shape[0],))

        set2[:,0] = np.reshape(x_2,(x_2.shape[0],))
        set2[:,1] = np.reshape(y_2,(y_2.shape[0],))

        set1 = np.array(set1, dtype = 'int')
        set2 = np.array(set2, dtype= 'int')

        locs = np.where(inlier_idx==1)
        locs = locs[0]

        kp1 = set1[locs,:]
        kp1 = kp1.astype('int64')
        kp2 = set2[locs,:]
        kp2 = kp2.astype('int64')

        mmtche = np.zeros((kp1.shape[0],2))
        mmtche[:,0] = np.arange(kp1.shape[0])
        mmtche[:,1] = np.arange(kp1.shape[0])

        mmtche = mmtche.astype('int64')

        fig, ax = plt.subplots(1,1)

        plt.gray()
        plot_matches(ax,im1,im2,kp1,kp2,mmtche)

        plt.show()

        #####################################################################

        blending_factor=0.8
        x=np.array([0,im1.shape[0]-1,0,im1.shape[0]-1])
        y=np.array([0,im1.shape[1]-1,im1.shape[1]-1,0])
        x = np.array(x,dtype=int)
        y= np.array(y,dtype=int)

        def homography(H,x,y):
            out=np.vstack((x,y,np.ones((y.shape))))
            out=np.matmul(H,out)
            out[0]=out[0]/out[2]
            out[1]=out[1]/out[2]
            return out[0],out[1]
        x_lim,y_lim=homography(Homo_lm,x,y)

        min_x,max_x,min_y,max_y=round(np.min(x_lim)),round(np.max(x_lim)),round(np.min(y_lim)),round(np.max(y_lim))

        x_trans=np.arange(min_x,max_x)
        y_trans=np.arange(min_y,max_y)
        x_trans,y_trans=np.meshgrid(x_trans,y_trans)

        x_trans=np.transpose(np.reshape(x_trans,(x_trans.shape[0]*x_trans.shape[1],1)))
        y_trans=np.transpose(np.reshape(y_trans,(y_trans.shape[0]*y_trans.shape[1],1)))
        x_source,y_source=homography(np.linalg.inv(Homo_lm),x_trans,y_trans)

        x_trans=np.transpose(x_trans)
        y_trans=np.transpose(y_trans)
        y_trans.shape

        x_lower,y_lower,x_high,y_high=int(min(min_x,0)),int(min(min_y,0)),int(max(max_x,im1.shape[0])),int(max(max_y,im1.shape[1]))
        img_canvas=np.zeros((x_high-x_lower+1,y_high-y_lower+1,3))
        img_canvas= np.array(img_canvas,dtype='uint8')

        stitch_x,stitch_y=int(max(1,1-x_lower)),int(max(1,1-y_lower))

        img_canvas[stitch_x:stitch_x+im2.shape[0],stitch_y:stitch_y+im2.shape[1],:]=im2

        id1= np.logical_and(x_source>=0 , x_source<im1.shape[0]-1)
        id2= np.logical_and(y_source>=0 , y_source<im1.shape[1]-1)
        id=  np.logical_and(id1,id2)


        x_source=x_source[id]
        y_source=y_source[id]
        x_trans=x_trans[id]
        y_trans=y_trans[id]


        for i in range(x_trans.shape[0]-1):
            ceilPixelx=int(np.ceil(x_source[i]))
            ceilPixely=int(np.ceil(y_source[i]))
            floorPixelx=int(np.floor(x_source[i]))
            floorPixely=int(np.floor(y_source[i]))

            y_1=0.5*(im1[floorPixelx,ceilPixely,:])+0.5*(im1[floorPixelx,floorPixely,:])
            y_2=0.5*(im1[ceilPixelx,ceilPixely,:])+0.5*(im1[ceilPixelx,floorPixely,:])
            x_avg=(0.5*y_1)+(0.5*y_2)

            if np.all(img_canvas[int(x_trans[i]-x_lower+1),int(y_trans[i]-y_lower+1),:])==0:
                img_canvas[int(x_trans[i]-x_lower+1),int(y_trans[i]-y_lower+1),:]=x_avg
            else:
                img_canvas[int(x_trans[i]-x_lower+1),int(y_trans[i]-y_lower+1),:]=0.7*(img_canvas[int(x_trans[i]-x_lower+1),int(y_trans[i]-y_lower+1),:])+(0.3)*(x_avg)

        img_canvas=img_canvas.astype(np.uint8)
        im1 = img_canvas
        plt.imshow(img_canvas)
        plt.show()


    img_mosaic = img_canvas
    return img_mosaic
