#!/usr/bin/env python3

import numpy as np
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

def coord_to_r(coord, cen_deduct = False, cen_coord = np.zeros(3)):
    
    # Calculate distance given coordinates;
    # set cen_deduct = True to calculate distance to this one center coordinate,
    # otherwise, the center is set to (0,0,0) by default  
    if (len(coord.shape) == 1): 
        return np.sqrt(np.sum(np.square(coord-cen_coord)))
    elif (coord.shape[1]==3):
        return np.sqrt(np.sum(np.square(coord-cen_coord),axis=1))
    else:
        return np.sqrt(np.sum(np.square(coord.T-cen_coord),axis=1))

def cal_vr_vt(coord, vel):
    #Calculate radial velcity and tangential velocity

    if coord.shape!=vel.shape:
        print('Coordinates shape does not match velocity shape!')
        return np.nan, np.nan
    else:
        if (coord.shape[1]==3):
            vr = np.sum(coord*vel, axis = 1)/coord_to_r(coord)
        else:
            vr = np.sum(coord*vel, axis = 0)/coord_to_r(coord)
        vt = np.sqrt( np.square(coord_to_r(vel)) - np.square(vr) )
        return vr, vt


def calculate_ang_mom(mass,coord,vel):

    # Calculate angular momentum given mass, coordinates and velocities
    if coord.shape[1]!=3:
        coord = coord.T
        vel = vel.T
    mom = np.zeros((coord.shape[0],3))
    mom[:,0] = mass * ( (coord[:,1]*vel[:,2]) - (coord[:,2]*vel[:,1]) )
    mom[:,1] = mass * ( (coord[:,2]*vel[:,0]) - (coord[:,0]*vel[:,2]) )
    mom[:,2] = mass * ( (coord[:,0]*vel[:,1]) - (coord[:,1]*vel[:,0]) )

    mom1 = np.sum(mom[:,0])
    mom2 = np.sum(mom[:,1])
    mom3 = np.sum(mom[:,2])

    return np.array((mom1,mom2,mom3))/np.sum(mass)

def norm_vec(v):

    # Normalize a 1D vector

    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    else:
        return v / norm

def vrrotvec(a,b):

    # Calculate to rotation vector that can rotate vector a to vector b

    an = norm_vec(a)
    bn = norm_vec(b)
    axb = np.cross(an,bn)
    ac = np.arccos(np.dot(an,bn))
    return np.append(axb,ac)

def vrrotvec2mat(r):

    # Convert rotation vector r to rotation matrix

    s = np.sin(r[3])
    c = np.cos(r[3])
    t = 1-c

    n = norm_vec(r[0:3])

    x = n[0]
    y = n[1]
    z = n[2]
    m = np.array( ((t*x*x + c, t*x*y - s*z, t*x*z + s*y),\
        (t*x*y + s*z, t*y*y + c, t*y*z - s*x),\
        (t*x*z - s*y, t*y*z + s*x, t*z*z + c)) )
    return m

def cal_rotation_matrix(a,b):

    # Calculate rotation matrix that can rotate vector a to vector b

    return vrrotvec2mat(vrrotvec(a,b))

def rotate_matrix(data,r):

    # Rotate data with rotation matrix r

    if data.shape[1]!=3:
        return np.dot(r,data)
    else:
        return np.dot(r,data.T).T

def rotate_2axis(data,a,b):

    # Rotate data in such a way that can rotate from vector a to vector b

    r = rotation_matrix(a,b)
    if data.shape[1]!=3:
        return np.dot(r,data)
    else:
        return np.dot(r,data.T).T

def smooth_hist(data_n, smooth_param): #smooth_param: calculate from left/right n data points
    mean = np.zeros(data_n.shape[0])
    dispersion = np.zeros(data_n.shape[0])
    for ii in range(0,data_n.shape[0]):
        if ii<smooth_param:
            sample = data_n[:ii+smooth_param+1]
        elif ii>=(data_n.shape[0]-smooth_param):
            sample = data_n[ii-smooth_param:]
        else:
            sample = data_n[ii-smooth_param:ii+smooth_param+1]
        mean[ii] = np.mean(sample)
        dispersion[ii] = np.sqrt( np.sum( np.square(sample-mean[ii] ) ) / (sample.shape[0]-1) )
    return mean, dispersion

def add_at(ax, t, loc=1, s=14):
    fp = dict(size=s)
    _at = AnchoredText(t, loc=loc, prop=fp)
    ax.add_artist(_at)
    return _at