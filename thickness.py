
import numpy as np
from .ahaseg import get_heartmask, get_seg, circular_sector, get_angle, get_sweep360
import numpy as np
from scipy import ndimage
from .cine import get_frame

def get_thick(heart_mask, nseg):
    LVbmask, LVwmask, RVbmask = get_heartmask(heart_mask)
    
    _, mask360, _ = get_angle(heart_mask, nseg)
    sweep360 = get_sweep360(LVwmask, LVwmask)
    thick_list = []
    for ii in range(nseg):
        thick_list.append(np.mean(sweep360[mask360 == (ii + 1)]))
    
    return np.array(thick_list)

def get_thickmap(LVwmask):

    LV_center = ndimage.center_of_mass(LVwmask)
    rr = np.min(np.abs(LVwmask.shape-np.array(LV_center))).astype(np.int)
    thickmap = LVwmask * 0

    sweep360 = []
    for theta in range(360):
        #print(theta)
        xall, yall = circular_sector(np.arange(0, rr, 0.5),
                                     theta, LV_center)
        projection = ndimage.map_coordinates(LVwmask, [xall, yall], order=0).sum()
        thickmap[xall.astype(np.int), yall.astype(np.int)] = projection
        
    thickmap = LVwmask * thickmap
    return thickmap

def get_thickmap_mean(label_mask, thick):
    thickmap_mean = label_mask.copy()
    for ii in range(thick.size):
        thickmap_mean[thickmap_mean == (ii+1)] = thick[ii]
        
    return thickmap_mean

def thick_ana_xy(heart_mask_xy, nseg=6):

    thick_result = dict()

    LVbmask, LVwmask, RVbmask = get_heartmask(heart_mask_xy)

    if np.sum(LVbmask) >0 and np.sum(LVwmask) >0 and np.sum(RVbmask) >0:

        label_mask = get_seg(heart_mask_xy, nseg)
        thick = get_thick(heart_mask_xy, nseg)
        thickmap = get_thickmap(LVwmask)
        thickmap_mean = get_thickmap_mean(label_mask, thick)

        thick_result['thickness'] = thick
        thick_result['thickmap'] = thickmap
        thick_result['thickmap_mean'] = thickmap_mean
        thick_result['ok'] = True
    
    else:
        thick_result['thickness'] = -1
        thick_result['thickmap'] = -1
        thick_result['thickmap_mean'] = -1
        thick_result['ok'] = False

    return thick_result

def thick_ana_xyt(heart_mask_xyt, nseg=6):

    thick_result = dict()

    sys_frame, dia_frame = get_frame(heart_mask_xyt)

    sys_thick = thick_ana_xy(heart_mask_xyt[..., sys_frame], nseg)
    dia_thick = thick_ana_xy(heart_mask_xyt[..., dia_frame], nseg)

    if sys_thick['ok'] and sys_thick['ok']:

        thickening = (sys_thick['thickness'] - dia_thick['thickness'])
        thickening = thickening / dia_thick['thickness']        

    else:
        thickening = -1

    thick_result['sys'] =  sys_thick
    thick_result['dia'] =  dia_thick
    thick_result['thickening'] =  thickening

    print(thickening)



    return thick_result

    