import numpy as np
from matplotlib import pyplot as plt

def nan_average(vals, wts):
    '''
    Compute weighted average with nans
    '''
    if np.all(np.isnan(wts)):
        return np.nan
    else:
        vals = np.array(vals)
        wts = np.array(wts)

        nan_idx = np.isnan(vals) | np.isnan(wts)

        return np.average(vals[~nan_idx], weights = wts[~nan_idx])

def weighted_std(vals, wts):
    '''
    Compute weighted standard deviation
    '''
    if np.all(np.isnan(wts)):
        return np.nan
    else:
        wtd_avg = nan_average(vals, wts = wts)
        return np.sqrt(nan_average((vals - wtd_avg)**2, wts = wts))

def angle_mean(angles_in_rad, wts):
    '''
    Compute the mean angle using directional statistics
    '''
    cosmean = nan_average(np.cos(angles_in_rad), wts = wts)
    sinmean = nan_average(np.sin(angles_in_rad), wts = wts)

    return np.arctan2(sinmean, cosmean)

def angle_median(angles_in_rad):
    '''
    Compute the median angle using directional statistics
    '''
    cosmean = np.nanmedian(np.cos(angles_in_rad))
    sinmean = np.nanmedian(np.sin(angles_in_rad))

    return np.arctan2(sinmean, cosmean)

def normalize_image(image):
    '''
    Normalize an image from [0, 255] to [0, 1]
    '''
    imin = np.min(image[:,:,0])
    imax = np.max(image[:,:,0])

    image = np.int16(((image - imin)/(imax - imin)) * 255)
    return image

def get_ax(rows=1, cols=1, size=16):
    _, ax = pp.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

# code from https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
def reject_outliers(data, m = 2.5):
    '''
    Reject outliers of data defined relative to the median
    '''
    d = np.abs(data - angle_median(data))
    mdev = angle_median(d)
    s = d/mdev if mdev else 0.
    return s < m

def wrapTo180(angle):
    '''
    Wrap angle to -180,180
    '''
    angle = np.radians(angle)
    return np.degrees(np.arctan2(np.sin(angle), np.cos(angle)))

def find_nearest(array, val):
    '''
    Find the value closest to the input argument 'val' in 'array'
    '''
    array = np.asarray(array)
    idx = np.argmin(np.abs(array - val))
    return idx

def gaussian(x, *params):
    '''
    A gaussian function f = A exp (-(x-a)^2/b^2)
    '''
    A, mu, sigma = params


def rotation_matrix(theta):
    '''
    Create a 2D rotation matrix by angle theta
    '''
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def rotate_2d_vector(xin, yin, theta, center_pt_x = 0, center_pt_y = 0):
    '''
    Rotate a 2-d vector whose coordinates are x, y by an angle theta
    '''
    
    if center_pt_x == 0:
        x = xin - np.nanmean(xin)
        xflag = False
    else:
        x = xin - center_pt_x
        xflag = True
        
    if center_pt_y == 0:
        y = yin - np.nanmean(yin)
        yflag = False
    else:
        y = yin - center_pt_y
        yflag = True
    
    R = rotation_matrix(theta)
    xp = list()
    yp = list()
    
    for (xi, yi) in zip(x, y):
        (xbuff, ybuff) = np.matmul(np.array([xi, yi]), R)
        xp.append(xbuff)
        yp.append(ybuff)
    
    if xflag:
        xp = np.array(xp) + center_pt_x
    else:
        xp = np.array(xp) + np.mean(xin)
        
    if yflag:
        yp = np.array(yp) + center_pt_y
    else:
        yp = np.array(yp) + np.mean(yin)
    
    return xp.flatten(), yp.flatten()


def rotate_contour(contour, normal_to_slipface_angle, x_c = np.nan, y_c = np.nan):
    '''
    Rotate the contour such that the slipface is oriented "down".
    Returns the rotated contour.
    Optional: x_c, y_c the coordinates of the point to rotate
    the contour around
    '''
    x = contour[:,0] - np.mean(contour[:,0]) if np.isnan(x_c) else contour[:,0] - x_c
    y = contour[:,1] - np.mean(contour[:,1]) if np.isnan(y_c) else contour[:,1] - y_c
    
    if np.isnan(normal_to_slipface_angle):
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    # Rotate s.t. the dune is facing down (angle starts from 3 * pi /2)
    R = rotation_matrix(- normal_to_slipface_angle)

    xp = list()
    yp = list()

    for (xi, yi) in zip(x, y):
        (xbuff, ybuff) = np.matmul(np.array([xi, yi]), R)
        xp.append(xbuff)
        yp.append(ybuff)

    xp = np.array(xp) + np.mean(contour[:,0])
    yp = np.array(yp) + np.mean(contour[:,1])
    
    rot_contour = np.array([xp, yp]).T.reshape(-1,2)
    
    return rot_contour


def calculate_ratio(a, b):
    '''
    Calculate the ratio (> 1) between two values, a and b
    '''
    if np.any(np.isnan([a,b])) or np.any(np.array([a,b]) == 0):
        return np.nan
    
    return np.max((a, b)) / np.min((a, b))