# -*- coding: utf-8 -*-
'''
example usage:
%matplotlib inline
import matplotlib.pyplot as plt
from pymr.heart import ahaseg
heart_mask = (LVbmask, LVwmask, RVbmask)
label_mask = ahaseg.get_seg(heart_mask, nseg=4)
plt.imshow(label_mask)
'''
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
def get_heartmask(heart_mask):
    if isinstance(heart_mask, tuple):
        #backward compatibility
        LVbmask, LVwmask, RVbmask = heart_mask
    else:
        LVbmask = (heart_mask==1)
        LVwmask = (heart_mask==2)
        RVbmask = (heart_mask==3)
    return LVbmask, LVwmask, RVbmask

def degree_calcu(UP, DN, seg_num):
    anglelist = np.zeros(seg_num)
    if seg_num == 4:
        anglelist[0] = DN - 180.
        anglelist[1] = UP
        anglelist[2] = DN
        anglelist[3] = UP + 180.

    if seg_num == 6:
        anglelist[0] = DN - 180.
        anglelist[1] = UP
        anglelist[2] = (UP + DN)/2.
        anglelist[3] = DN
        anglelist[4] = UP + 180.
        anglelist[5] = anglelist[2] + 180.
    anglelist = (anglelist + 360) % 360

    return anglelist.astype(int)


def degree_calcu_new(mid, seg_num):
    anglelist = np.zeros(seg_num)
    if seg_num == 4:
        anglelist[0] = mid - 45 - 90
        anglelist[1] = mid - 45
        anglelist[2] = mid + 45
        anglelist[3] = mid + 45 + 90

    if seg_num == 6:
        anglelist[0] = mid - 120
        anglelist[1] = mid - 60
        anglelist[2] = mid
        anglelist[3] = mid + 60
        anglelist[4] = mid + 120
        anglelist[5] = mid + 180
    anglelist = (anglelist + 360) % 360

    return anglelist.astype(int)

def circular_sector(r_range, theta_range, LV_center):
    cx, cy = LV_center
    theta = theta_range/180*np.pi
    z = r_range.reshape(-1, 1).dot(np.exp(1.0j*theta).reshape(1, -1))
    xall = -np.imag(z) + cx
    yall = np.real(z) + cy
    return xall, yall



def get_theta(sweep360):
    from scipy.optimize import curve_fit
    from scipy.signal import medfilt
    
    y = sweep360.copy()
    
    def gauss(x, *p):
        A, mu, sigma = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))

    def fit(x, y):
        #print(y)
        p0 = [np.max(y), np.argmax(y)+x[0], 1.]
        try:
            coeff, var_matrix = curve_fit(gauss, x, y, p0=p0)
            A, mu, sigma = coeff
        except:
            mu = 0
            sigma = 0
        return mu, sigma
    y = medfilt(y)
    y2 = np.hstack([y, y, y])
    #y2 = medfilt(y2)
    maxv = y2.argsort()[::-1][:10]
    maxv = maxv[np.argmin(np.abs(maxv-360*3//2))]
    #print('maxv:%d' % maxv, y2.argsort()[::-1][:10])
    y2[:(maxv-90)] = 0
    y2[(maxv+90):] = 0
    #print(y2[(maxv-90):maxv])
    x = np.arange(y2.size)
    mu, sigma = fit(x[(maxv-150):maxv], y2[(maxv-150):maxv])
    uprank1 = mu - sigma*2.5
    mu, sigma = fit(x[maxv:(maxv+150)], y2[maxv:(maxv+150)])
    downrank1 = mu + sigma*2.5
    uprank2 = np.nonzero(y2 >= min(np.max(y2), 5))[0][0]
    downrank2 = np.nonzero(y2 >= min(np.max(y2), 5))[0][-1]
    uprank = int(max(uprank1, uprank2)) % 360 + 360
    downrank = int(min(downrank1, downrank2)) % 360 + 360
    #print(uprank, downrank)
    phase = np.deg2rad(np.array([uprank, downrank]))
    uprank, downrank = np.rad2deg(np.unwrap(phase)).astype(np.int) - 360
    

    #print(uprank, downrank)
    #print('=' * 20)
    return uprank, downrank


def get_sweep360(LVwmask, RVbmask):

    LV_center = ndimage.center_of_mass(LVwmask)
    rr = np.min(np.abs(LVwmask.shape-np.array(LV_center))).astype(np.int)

    sweep360 = []
    for theta in range(360):
        #print(theta)
        xall, yall = circular_sector(np.arange(0, rr, 0.5),
                                     theta, LV_center)
        projection = ndimage.map_coordinates(RVbmask, [xall, yall], order=0).sum()
        sweep360.append(projection)
    return np.array(sweep360)


def get_angle(heart_mask, nseg=4):
    def sector_mask(xall, yall, LVwmask):
        smask = LVwmask * 0
        xall = np.round(xall.flatten())
        yall = np.round(yall.flatten())

        mask = (xall >= 0) & (yall >= 0) & \
               (xall < LVwmask.shape[0]) & (yall < LVwmask.shape[1])
        xall = xall[np.nonzero(mask)].astype(int)
        yall = yall[np.nonzero(mask)].astype(int)
        smask[xall, yall] = 1
        return smask
    
    #step 1: sweep 360 and get AHA angles
    
    LVbmask, LVwmask, RVbmask = get_heartmask(heart_mask)
    #import time
    #t = time.time()
    sweep360 = get_sweep360(LVbmask, RVbmask)
    UP, DN = get_theta(sweep360)
    #anglelist = degree_calcu(UP, DN, nseg) old
    j = (-1)**0.5
    pi = np.pi
    mid_angle = np.angle(np.exp(j*UP/180*pi) + np.exp(j*DN/180*pi))/pi*180
    anglelist = degree_calcu_new(mid_angle, nseg)
      
    #step 2: calculate mask360, 360 points with AHA labels

    angles2 = np.append(anglelist, anglelist[0])
    angles2 = np.rad2deg(np.unwrap(np.deg2rad(angles2)))

    mask360 = np.zeros((360, ))
    #print(angles2)
    for ii in range(angles2.size-1):
        temp = np.arange(angles2[ii], angles2[ii + 1]).astype(np.int)
        temp[temp >= 360] = temp[temp >= 360] - 360
        mask360[temp] = ii + 1
        
    #step 3: calculating AHA sector
    LV_center = ndimage.center_of_mass(LVbmask)
    AHA_sector = LVwmask * 0
    rr = np.min(np.abs(LVwmask.shape-np.array(LV_center))).astype(np.int)


    for ii in range(angles2.size-1):
        xall, yall = circular_sector(np.arange(0, rr, 0.5),
                                     np.arange(angles2[ii], angles2[ii+1],
                                     0.1), LV_center)
        
        smask = sector_mask(xall, yall, LVwmask)
        
        #print(angles2[ii], angles2[ii+1])
        #print(xall, yall)
        #plt.figure()
        #plt.imshow(smask)
        #plt.title('%f_%f' % (angles2[ii], angles2[ii+1]))
        
        AHA_sector[smask > 0] = (ii + 1)
    
    
    #plt.figure()
    #plt.imshow(AHA_sector)
    #plt.title(angles2)
    return anglelist, mask360, AHA_sector
#print(time.time() - t)
#plt.figure()
#plt.imshow(label_mask)

def get_seg(heart_mask, nseg=4):
    LVbmask, LVwmask, RVbmask = get_heartmask(heart_mask)

    _, _, AHA_sector = get_angle(heart_mask, nseg)
    #label_mask = labelit(anglelist, LVwmask)
    label_mask = AHA_sector * LVwmask

    return label_mask
