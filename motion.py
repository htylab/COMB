import matplotlib.pyplot as plt
import numpy as np
from .ahaseg import get_seg
from scipy import ndimage
import dipy.align.imwarp as imwarp
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric

def get_seg_xyt(heart_xyt, nseg=6):

    #step 1: collect dia_frame and sys_frame
    curve = np.sum(heart_xyt==1, axis=(0, 1))
    #print(curve)
    dia_frame = np.argmax(curve)
    curve[curve==0] = 1e8
    sys_frame = np.argmin(curve)
    #print(dia_frame, sys_frame)


    heart_sys = heart_xyt[..., sys_frame]
    heart_dia = heart_xyt[..., dia_frame]

    if np.sum(np.unique(heart_dia)) + np.sum(np.unique(heart_sys))< 12:
        return heart_sys*0, heart_dia*0

    if dia_frame==sys_frame:
        return heart_sys*0, heart_dia*0    

    heart_seg_sys = get_seg(heart_sys, nseg)
    heart_seg_dia = get_seg(heart_dia, nseg)

    return heart_seg_sys, heart_seg_dia

def get_center(label_mask):
    center_list = []
    for ii in range(np.max(label_mask)):
        center = ndimage.center_of_mass(label_mask==(ii+1))
        center_list.append(center)
        
    return np.array(center_list)


def get_motion_field(static, moving):

    dim = static.ndim
    metric = SSDMetric(dim)
    level_iters = [100, 50, 25]
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

    mapping = sdr.optimize(static, moving)
    warped = mapping.transform(moving)
    backward_field = mapping.get_backward_field()
    motion_magnitude = np.sqrt(backward_field[..., 0]**2 + backward_field[..., 1]**2)
    return warped, motion_magnitude, backward_field



def motion_ana_xyt(heart_xyt, nseg=6, display=False):

    heart_seg_sys, heart_seg_dia = get_seg_xyt(heart_xyt, nseg)
    if (np.sum(heart_seg_sys) == 0) or (np.sum(heart_seg_dia) == 0):
        got_seg = False
        motion_map = heart_seg_sys * 0
        motion_vector = -1
    else:
        got_seg = True
        sys_center = get_center(heart_seg_sys)
        dia_center = get_center(heart_seg_dia)
        motion_vector = (sys_center - dia_center)
        mask_dia_to_sys, motion_map, backward_field = get_motion_field(heart_seg_sys, heart_seg_dia)
        #union_mask = heart_seg_sys | heart_seg_dia
        #motion_map = motion_map * (union_mask > 0)

    if display and got_seg:
        plt.figure(figsize=(15,5))
        plt.subplot(131)
        plt.imshow(heart_seg_dia)
        plt.scatter(dia_center[:, 1], dia_center[:, 0], color='red')
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(heart_seg_sys)
        plt.scatter(sys_center[:, 1], sys_center[:, 0], color='red')
        plt.axis('off')

        #fig = plt.figure(figsize=(10, 10))

        plt.subplot(133)
        #fig.subplots_adjust(left=0,right=0,bottom=0,top=0)
        #ax = plt.axes([0.0, 0.0, 1.0, 1.0])
        plt.imshow(heart_seg_dia, cmap='gray')
        index = np.arange(6)
        plt.quiver(dia_center[index, 1], dia_center[index, 0] , 
                motion_vector[index, 1], motion_vector[index, 0]*-1,
                color='r', headlength=6, scale=50)
        plt.axis('off')

        #plt.subplot(122)
        #plt.imshow(motion_map)
        #plt.axis('off')
        #plt.show()

    return motion_vector, motion_map
