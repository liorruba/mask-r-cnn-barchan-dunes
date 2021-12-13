import numpy as np
import cv2
from matplotlib import pyplot as plt
import circle_fit as cf
from scipy.signal import find_peaks, peak_widths, peak_prominences, savgol_filter, argrelextrema
from scipy import optimize
from scipy.interpolate import interp1d
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from skimage.draw import line
from scipy.optimize import curve_fit

from myutils import *


def get_contours(img, prediction_data,
                    only_barchans = True, confidence = 0.6,
                    is_draw_contour = False, cnt_color = (255, 255, 255)):
    '''
    Gets the contours of an image with prediction data from mask_rcnn
    pass only_barchans to only return the contours of barchan dunes
    pass confidence to only return the contours of dunes with score > confidence
    pass is_draw_contour to also draw the contours on the image
    '''
    
    contour_list = list()

    # Filter by feature type and confidence:
    logical_idxs = (prediction_data['scores'] > confidence)

    if only_barchans:
        logical_idxs = np.logical_and(logical_idxs, prediction_data['class_ids'] == 1)

    masks = prediction_data['masks'][:,:,logical_idxs]

    # Iterate over all masks to get their contours
    for mask_idx in range(np.shape(masks)[2]):
        cnt, hierarchy = cv2.findContours(
                np.uint8(masks[:,:,mask_idx]),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE
            )

        # If object answers the criteria:
        if np.ndim(np.squeeze(cnt)) == 1:
            # Self-intersecting contour, skip:
            continue
        else:
            if np.shape(cnt)[1] > 4 and np.ndim(np.squeeze(cnt)) == 2:
                contour_list.append(np.squeeze(cnt))
            else:
                continue

        if is_draw_contour:
            img = cv2.drawContours(img, cnt, -1, cnt_color, 2)

    contour_list = np.array(contour_list)
    return contour_list


def best_fit_ellipse(contours, image, is_return_image = False):
    '''
    DEPRECATED:
    Fit ellipses to given contours
    pass image and is_return_image = True to draw the ellipses
    outputs a cv2 ellipse object
    '''
    ellipse_list = list()

    if np.ndim(contours) == 2:
        elp = cv2.fitEllipse(np.squeeze(contours))
        ellipse_list.append(elp)
        return ellipse_list

    for cnt in contours:
        # Check the contour is valid (not zero sized, 2xN)
        if np.size(cnt) > 3:
            elp = cv2.fitEllipse(np.squeeze(cnt))
            ellipse_list.append(elp)
        else:
            ellipse_list.append([np.nan])

        if is_return_image:
            image = cv2.ellipse(image, elp, (255, 255, 255), 2)

    return ellipse_list


def best_fit_triangle(contours, image, is_return_image = False):
    '''
    DEPRECATED:
    Fit triangles to given contours
    pass image and is_return_image to draw triangles
    outputs the tirangles vertices
    '''
    triangles_list = list()

    for cnt in contours:
        bf_triangle = cv2.minEnclosingTriangle(np.expand_dims(cnt, axis=1))
        triangles_list.append(bf_triangle[1])

        if is_return_image:
            image = cv2.polylines(image, [np.int32(bf_triangle[1])],
                      True, (0, 255, 0), 2)

    return triangles_list

# Detect the convex defects of a geometric shape
# from a contour and the shape convex hull
def detect_convex_defects(contour, cvx_hull, num_of_concave_pts = 1):
    defects = cv2.convexityDefects(contour, cvx_hull)

    # If there are defects in the convex contour
    if np.all(defects) != None:
        # Clear lists
        depth_list = list()
        far_list = list()

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            far_list.append(tuple(contour[f]))
            depth_list.append(d)

        far_list = np.array(far_list)
        depth_list = np.array(depth_list)

        # If the number of concave points is smaller than the max possible,
        # set it to the max possible
        num_of_concave_pts = depth_list.size if num_of_concave_pts > depth_list.size else num_of_concave_pts

        # Sort the distances by the deepest defect
        depth_list_idx = np.argsort(depth_list)[-num_of_concave_pts:][::-1]
        depth_list = depth_list[depth_list_idx]

        # Sort to return the deepest defects
        convex_defects = list(tuple(sub) for sub in far_list[depth_list_idx])

    else:
        convex_defects = list()

    return convex_defects, depth_list

# Gets the coordinates of the center of the dune contour
# Returns a tuple (center_x, center_y)
def get_dune_center(contour):
    '''
    Compute the center of the dune contour from
    the contour moments
    '''
    M = cv2.moments(contour)
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

def detect_slipface_from_shape(contours, image, is_return_image,
                               is_return_horns = False, a_priori_azimuth = (-180, 180),
                               num_of_defects = 10):
    '''
    detect the crest from the dune shape using the convex defect method
    recieves the dune contours. pass a_priori_azimuth tuple (min,max)
    to only search for convex defects in some angle range
    Also pass image and is_return_image to draw the defects on the image.
    '''

    contour_defects_list = list()
    azimuth_list = list()
    depth_list = list()
    cvx_defect_ratio_list = list()
    horns_apexes_list = list()
    
    # If a_priori_azimuth is a scalar, make it a range:
    if np.size(a_priori_azimuth) == 1:
        a_priori_azimuth = (
            a_priori_azimuth - 45, a_priori_azimuth + 45
            )

    # For each contour, determine the deepest convex defect
    for cnt in contours:
        hull = cv2.convexHull(cnt, returnPoints = False)
        hull[::-1].sort(axis=0)
        defects, defect_depths = detect_convex_defects(cnt, hull, num_of_defects)

        # Search for the deepest convex defect in some azimuth range
        cx, cy = get_dune_center(cnt)

        # Calculate the defect point coordinate relative to the dune center
        rel_max_concave = defects - np.array([cx, cy])

        # Get the point whose azimuth is closest to the a-priori direction
        azimuth = np.degrees(np.arctan2(rel_max_concave[:,0], rel_max_concave[:,1]))

        # Estimate the likelihood dune is barchan
        elp = best_fit_ellipse(cnt, image)

        cvx_defect_ratio = calculate_cvx_defect_depth_ratio(defect_depths)

        # Select only the deepest defect within the chosen azimuth
        azimuth_in_range = np.where(
                                    (azimuth >= a_priori_azimuth[0]) &
                                    (azimuth <= a_priori_azimuth[1])
                                    )[0]

        # If no convex defects in range or the dune is not barchan (maximum
        # convex defect is not p times greater than second deepest, append nan:
        if np.size(azimuth_in_range) > 0:
            # The deepest defect in range
            deepest_defect_idx = np.argmax(defect_depths[azimuth_in_range])

            # The maximum defect that corresponds to that point
            max_defect = np.array(defects)[azimuth_in_range][deepest_defect_idx]
            # The depth of the maximum defect:
            max_concave_depth = np.array(defect_depths)[azimuth_in_range][deepest_defect_idx]
            # The azimuth that corresponds to that point
            max_concave_az = (np.array(azimuth)[azimuth_in_range][deepest_defect_idx]) % 360


            # If the new maximum convex defect (the one within the angle range)
            # is much smaller than the deepest convex defect, switch back to
            # the deepest convex defect. 
            if (max_concave_depth / np.max(defect_depths) < 0.75):
                max_idx = np.argmax(defect_depths)
                max_defect = np.array(defects)[max_idx]
                max_concave_depth = np.array(defect_depths)[max_idx]

            # Add to lists
            contour_defects_list.append(max_defect)
            azimuth_list.append(max_concave_az)
            depth_list.append(max_concave_depth)
            cvx_defect_ratio_list.append(cvx_defect_ratio)
            
            # Calculate horns
            if is_return_horns:
                max_defect_index = np.where(np.all(cnt == max_defect, 1))[0]
                horn_apexes = get_dune_horn_apexes(cnt, max_defect_index, hull)

                horns_apexes_list.append(horn_apexes)
            
            
            # If the user opted to return an image, render it
            if is_return_image:
                # DEBUG: Uncomment to draw dunes contours
                # image = cv2.drawContours(image, [cnt], -1, (255, 255, 255), 3)
                # Uncomment to draw the convex hull polygon
                # image = cv2.polylines(image, [cnt[np.squeeze(hull)]], True, (255,255,255))
                # Uncomment to draw contour center
                # image = cv2.circle(image, (cx, cy), 5,(0, 255, 255), -1)
                # Uncomment to show slipfaces
                # image = cv2.circle(image, tuple(max_defect), 5 ,(255, 255, 0), 3)
                pass

            else:
                image = {}

        else:
            contour_defects_list.append(np.nan)
            azimuth_list.append(np.nan)
            depth_list.append(np.nan)
            cvx_defect_ratio_list.append(np.nan)
            
            if is_return_horns:
                horns_apexes_list.append(np.ones((2, 2)) * np.nan)
                
    if is_return_horns:
        return contour_defects_list, azimuth_list, depth_list, cvx_defect_ratio_list, horns_apexes_list
    
    else:
        return contour_defects_list, azimuth_list, depth_list, cvx_defect_ratio_list

def get_dune_albedo_gradient(image, gauss_smooth = 5):
    '''
    DEPRECATED: Find the direction of the illumination gradient
    across the dune stoss-slipface line
    '''
    buff = np.double(image)
    buff = cv2.GaussianBlur(buff, (gauss_smooth, gauss_smooth), 0)
    gx,gy = np.gradient(buff)

    return gx, gy

def normalized_convex_defect_depth(ellipses, depths):
    '''
    Normalize the convexity defect by the ellipese fit to the dune
    '''
    # Return a list of normalize defect depths (proxy for certainty):
    normalized_depth_list = list()

    if np.size(depths) == 1:
        maj_ax_len = ellipses[0][1][1]
        # fix a bug related to depths sometimes being passed as a list
        depths = np.squeeze(np.array([depths]))
        normalized_depth = depths / 256 / (maj_ax_len)
        normalized_depth_list.append(normalized_depth)

        return normalized_depth_list

    # Fit ellipses to contours to determine their sizes
    for elp, depth in zip(ellipses, depths):
        maj_ax_len = elp[1][1]

        # Normalize depth of convex defect:
        # cv2.convexityDefects, for some reason, multiplies the depth of the
        # defects by 256. I don't really understand why, but to get the right
        # number, need to divide by 256.
        normalized_depth = depth / 256 / maj_ax_len

        # Add the normalized depth of the defect as the uncertainty.
        # If this depth is large, it is more likely this is a dune.
        normalized_depth_list.append(normalized_depth)

    return normalized_depth_list

def calculate_cvx_defect_depth_ratio(depth_list):
    '''
    Compute the convexity defect ratio
    '''
    if np.size(depth_list) == 1:
        return 1

    return depth_list[1] / depth_list[0]


def horns_bisector_vec(vecs_to_horns):
    '''
    Find the bisector vector to the head angle of the triangle defined
    by the two horns apexes and the slipface
    '''    
    # Find the bisector vector
    bisector_vector = np.linalg.norm(vecs_to_horns[0]) * vecs_to_horns[1] + np.linalg.norm(vecs_to_horns[1]) * vecs_to_horns[0]
    
    if np.any(np.isnan(bisector_vector)):
        return np.array([np.nan, np.nan])
    
    return bisector_vector
    
def tail_to_slipface_vec(tail, slipface):
    '''
    Find the vector from the dune tail to the slipface
    '''
    if np.any(np.isnan(tail)) or np.any(np.isnan(slipface)):
        return np.array([np.nan, np.nan])
    
    return slipface - tail
    

def calculate_migration_dir_sand_flux(contours, slipfaces, horn_apexes, img, is_draw_normals = False, 
                                      is_draw_tails = False, is_draw_idxs = False, horns_threshold = 1.5):
    '''
    Calculate the migration direction (vectors normal to the slipface)
    and the sand flux direction (vector from the slipface to the loner horn)
    also returns the horns' lengths and location of the tails
    '''
    elongation_direction_list = list()
    migration_dir_vector_list = list()
    tail_list = list()
    horns_lengths_euc_list = list()

    for dune_idx, (cnt, slipface, horn_apex) in enumerate(zip(contours, slipfaces, horn_apexes)):     
        # Find the vectors from the slipface to the horns
        vecs_to_horns = np.double(np.array(horn_apexes[dune_idx]) - np.array(slipface))
        
        # Make an initial guess regarding the migration direction, as
        # the bisector of the opening angle of the dune:
        migration_dir_guess = horns_bisector_vec(vecs_to_horns)

        # Find the coordinate of the tail:        
        tail = get_dune_tail(cnt, 
                             slipface, 
                             horn_apexes[dune_idx],  
                             slipface_dir = np.arctan2(-migration_dir_guess[1], migration_dir_guess[0]), 
                             is_draw_tail = is_draw_tails)
        
        ######################
        # Migration direction:
        ######################
        # migration direction as the bisector
        migration_direction_bisector = migration_dir_guess
        # migration direction as the stoss ("tail") to slipface vector
        migration_direction_stoss_slipface = tail_to_slipface_vec(tail, slipface)

        ######################
        # Elongation dir
        ######################
        # For the elongation direction, look at the longest euclidean distance:
        horns_lengths_euc = calc_horns_lengths_euclidean(tail, horn_apex)
        elongation_direction = vecs_to_horns[np.argmax(horns_lengths_euc)]
        
        # Store in vectors
        elongation_direction_list.append((elongation_direction[0], -elongation_direction[1]))
        migration_dir_vector_list.append([(migration_direction_bisector[0], -migration_direction_bisector[1]), (migration_direction_stoss_slipface[0], -migration_direction_stoss_slipface[1])])
        tail_list.append(tail)
        
        horns_lengths_euc_list.append(horns_lengths_euc)
    
        # Draw normals?
        if is_draw_normals and not np.any(np.isnan(slipface)):
            plt.quiver(slipface[0], slipface[1], migration_direction_stoss_slipface[0], -migration_direction_stoss_slipface[1], headwidth = 3,
                        headlength = 3, headaxislength = 2, minshaft = 1,
                        width = 5e-6 * img.shape[0], color = 'r')
            
            plt.quiver(slipface[0], slipface[1], migration_direction_bisector[0], -migration_direction_bisector[1], headwidth = 3,
                        headlength = 3, headaxislength = 2, minshaft = 1,
                        width = 5e-6 * img.shape[0], color = 'w')

        # Draw indexes?
        if is_draw_idxs and not np.any(np.isnan(slipface)):
            plt.text(slipface[0] + 5, slipface[1] + 5, str(dune_idx), color = 'r', fontsize = 14)
    
    return elongation_direction_list, migration_dir_vector_list, tail_list, horns_lengths_euc_list

def calculate_dune_length(tail_coords, slipface_coords):
    '''
    Calculate the length of the dune from its tail and
    slipface coordinates
    '''
    (tail_x, tail_y) = tail_coords
    dune_length = np.linalg.norm(np.array([tail_x, tail_y]) - slipface_coords)
#     dune_length = np.sqrt((tail_x - slipface_coords[0]) ** 2 + (tail_y - slipface_coords[1]) ** 2)
    
    return dune_length
    
def calculate_dune_width(cnt, migration_direction_vec):
    '''
    Calculate the width of the dune as the distance
    between the two farthest points along a line
    perpendicular to the migration direction
    '''
    # Rotate the dune contour so that the slipface is oriented "down":
    xp, yp = rotate_2d_vector(cnt[:,0], cnt[:,1], migration_direction_vec + np.pi/2)
    
    # Get the further points along the horizontal axis:
    return(np.abs(np.max(xp) - np.min(xp)))

def calculate_horns_width_and_length(cnt, left_horn_apex, right_horn_apex, migration_direction_vec, slipface):
    '''
    1. Calculate the width of the dune's horns
    as the distance between the apexes and the 
    dunes extrema along the horizontal direction.
    2. Calculate the length of the horns as the distance between
    a line passting through the slipface, perpendicular to the migration direction
    and the horns' apexes
    '''
    # Rotate the dune contour so that the slipface is oriented "down":
    xp, yp = rotate_2d_vector(cnt[:,0], cnt[:,1], migration_direction_vec + np.pi/2)
    
    left_horn_xp, left_horn_yp = rotate_2d_vector([left_horn_apex[0]], [left_horn_apex[1]], migration_direction_vec + np.pi/2, 
                                                  center_pt_x = np.nanmean(cnt[:,0]), center_pt_y = np.nanmean(cnt[:,1]))

    right_horn_xp, right_horn_yp = rotate_2d_vector([right_horn_apex[0]], [right_horn_apex[1]], migration_direction_vec + np.pi/2,
                                                    center_pt_x = np.nanmean(cnt[:,0]), center_pt_y = np.nanmean(cnt[:,1]))
    
    slipface_xp, slipface_yp = rotate_2d_vector([slipface[0]], [slipface[1]], migration_direction_vec + np.pi/2,
                                                center_pt_x = np.nanmean(cnt[:,0]), center_pt_y = np.nanmean(cnt[:,1]))
    
    # Dunes horns' widths:
    left_horn_width = np.abs(np.min(xp) - left_horn_xp)
    right_horn_width = np.abs(np.max(xp) - right_horn_xp)

    # Horns' lengths:
    left_horn_length = slipface_yp - left_horn_yp
    right_horn_length = slipface_yp - right_horn_yp
    
    if left_horn_length[0] < 0:
        left_horn_length = [1e-16]
    if right_horn_length[0] < 0:
        right_horn_length = [1e-16]
        
    return ((left_horn_width[0], right_horn_width[0]), (left_horn_length[0], right_horn_length[0]))

def calculate_normal_to_contour_at_point(pt_idx, contour, dune_length, tail, slipface):
    '''
    Compute the normal to the contour at some point
    '''
    pt = contour[pt_idx % len(contour)]
    
    a = contour[(pt_idx + 3) % len(contour)]
    b = contour[(pt_idx - 3) % len(contour)]
    u = (a - b)
    
    # The perpendicular unit vector is:
    uperp = np.array([-u[1], u[0]])
    uperp = uperp / np.sqrt(np.sum(uperp**2))
    
    # Make sure the transect is facing downwind:
    tail_slp_vec = tail - slipface
    
    if np.dot(uperp, tail_slp_vec) < 0:
        uperp = uperp * (-1)
    
    # Use the unit vector to get points that are a little
    # before and after the input point
    if np.isnan(dune_length):
        return np.nan
    else:
        before_pt = pt + np.int8(uperp * dune_length / 2)
        after_pt = pt - np.int8(uperp * dune_length / 2)
    
    return (before_pt, after_pt)

def slipface_width_from_brightness(x, img):
    '''
    Compute the width (length) of the slipface from its 
    brightness gradient
    '''
    # Return nan if the line exceeds the image:
    if np.any(x >= np.shape(img)[0]):
        return (np.nan, np.nan)
    
    # Get the pixel value of the points along the line:
    
    xx = np.sort(np.sqrt((x[:,1] - x[0,1])**2 + (x[:,0] - x[0,0])**2))
    
    y = img[x[:,1], x[:,0]]
    
    # If the number of pixels across the transect is small, return nan
    if np.max(xx) < 5:
        return (np.nan, np.nan)
    
    yy = gaussian_filter1d(y, 1)    
    pp = np.polyfit(xx, yy, 1)
    yy = yy - np.polyval(pp, xx)
    yy = np.abs(yy)
    
    # Find peaks
    max_idxs, _ = find_peaks(yy)

    # If could not find peaks, return nan:
    if (np.size(max_idxs) <= 1):
        return (np.nan, np.nan)
    
    # Find the highest prominence, bright or dark
    all_prom = peak_prominences(yy, max_idxs)[0]
    
    max_prom_idx = max_idxs[np.argmax(all_prom)] 

    # If the highest peak is dark, flip the signal to fit a gaussian:
    if yy[max_prom_idx] < 0:
        yy = -yy 

    # Calculate the width of the highest prominence as the initial guess for the slipface width
    widths = peak_widths(yy, [max_prom_idx], rel_height = 1 - np.nanmedian(yy) / yy[max_prom_idx])
#     print( np.median(yy))
    std_init_guess = widths[0]
    base_height = (widths[1] + widths[2]) / 2

    # Initial guess for gaussian
    p0 = [1, xx[max_prom_idx], std_init_guess[0]]
    snr = np.sqrt(np.mean(y**2))
    return (std_init_guess[0], snr)      
    
    # Dummy x axis for fit:
    y_dist = np.sqrt((x[0,0] - x[-1,0])**2 + (x[0,1] - x[-1,1])**2)
    
    xy = np.linspace(0, y_dist, len(y))
            
    smooth_y = gaussian_filter1d(y, 1)
    
    # Find brightness peaks (caused by illumination differences)
    # along this line
    max_idx, max_prop = find_peaks(smooth_y)
    min_idx, min_prop = find_peaks(-smooth_y)
    
    max_prom = peak_prominences(smooth_y, max_idx)
    min_prom = peak_prominences(-smooth_y, min_idx)
    
    all_extrema = np.concatenate([max_idx, min_idx])
    all_prom = np.concatenate([max_prom[0], min_prom[0]])
    
    # If couldn't identtify peaks, return nan:
    if np.size(all_prom) == 0:
        return np.nan
    
    # The tallest extrema (either dark or bright):
    tallest_extrema_idx = all_extrema[np.argmax(all_prom)]

    # If found an extrema:
    if np.size(all_extrema) > 0:
        # If the peak is bright:
        if tallest_extrema_idx in max_idx:
            
            slipface_length_px = peak_widths(smooth_y, [tallest_extrema_idx], rel_height = 0.95)
        # If it is dark:
        else:
            slipface_length_px = peak_widths(-smooth_y, [tallest_extrema_idx], rel_height = 0.95)
        
        # Calculate the length of the slipface by multiplyng the width of the peak
        # by the euclidean length of the line
        slipface_length = slipface_length_px[0][0] / len(smooth_y) * y_dist
#         print(tallest_extrema_idx, slipface_length)
        return slipface_length
    
    
    # If didn't find an extrema, return nan:
    else:
        return np.nan
    

def get_dune_height(img, tail, slipface, contour, dune_length, ret_height = False):
    '''
    Calculate the dune height by measuring the length of the slipface
    (in map view). To return height instead of slipface length, 
    set ret_height = True
    '''
    if len(np.shape(img)) > 2:
        img = img[:,:,0]
    
    # If the slipface was not identified, return nan
    if np.all(np.isnan(slipface)) or ~np.isfinite(dune_length):
        return (np.nan, np.nan)
    
    # The index of the slipface along the contour
    slipface_idx = closest_point_on_contour(contour, slipface)    
    before_dune_slipface, after_dune_slipface = calculate_normal_to_contour_at_point(slipface_idx, contour, dune_length, tail, slipface)
    
    # Stretch a line between these points:
    x_slipface = np.array(list(zip(*line(*before_dune_slipface, *after_dune_slipface))))
    
    slipface_length, loss = slipface_width_from_brightness(x_slipface, img)
    slipface_length_p1, loss_p1 = (np.nan, np.nan)#slipface_width_from_brightness(x_p1, img)
    slipface_length_p2, loss_p2 = (np.nan, np.nan)#slipface_width_from_brightness(x_p2, img)

    slp_faces_lengths = np.array([slipface_length, slipface_length_p1, slipface_length_p2])
    slp_faces_loss = np.array([loss, loss_p1, loss_p2])
    
    mean_slipface_length = np.nanmean(slp_faces_lengths)
    mean_loss = np.nanmean(slp_faces_loss)
    return (mean_slipface_length, mean_loss)

def get_dune_horn_apexes(dune_contour, slipface_idx, convex_hull):
    '''
    Find the apxes as the intersection points between the contour and the 
    convex hull
    '''
    
    try:
        dist = slipface_idx - convex_hull.flatten()
    except ValueError:
        dist = slipface_idx[0] - convex_hull.flatten()
        
    right_horn_idx = convex_hull[np.argwhere(np.array(dist) > 0)[0]][0][0]
    
    left_horn_idx = convex_hull[np.argwhere(np.array(dist) > 0)[0] - 1][0][0]
    
    right_horn = np.double(dune_contour[right_horn_idx])
    left_horn = np.double(dune_contour[left_horn_idx])
    
    return np.array([right_horn, left_horn])

def get_dune_tail(contour, slipface, horn_apexes, slipface_dir = 0, is_draw_tail = False):
    '''
    Find the coordinates of the dune's tail
    as the longest distance between its slipface
    the contour at its downwind (tail) part
    '''
    # If the defect point does not exist, return nan:
    if np.any(np.isnan(slipface)):
        return np.nan 

    # Get index of slipface
    slipface_idx = closest_point_on_contour(contour, slipface)
    
    # Permute contour, s.t. the slipface has an index of 0:
    contour = np.roll(contour, -slipface_idx, axis = 0)    
    
    # Find the right and left horn apexes
    horn_1_idx = closest_point_on_contour(contour, horn_apexes[0])
    horn_2_idx = closest_point_on_contour(contour, horn_apexes[1])   
    
    # Rotate the contour:
    rot_contour = rotate_contour(contour, slipface_dir + np.pi/2) 
    
    # Find which horn is the right and which is the left:
    slipface_x, slipface_y = rot_contour[0]
    horn_right = np.min([horn_1_idx, horn_2_idx])
    horn_left = np.max([horn_1_idx, horn_2_idx])
    right_horn_x, right_horn_y = rot_contour[horn_right]
    left_horn_x, left_horn_y = rot_contour[horn_left]
    
    # Remove the points on the slipface part of the dune
    rot_contour[0:horn_right,:] = np.nan
    rot_contour[horn_left:-1,:] = np.nan
    
    # Now, truncate the contour below the slipface
    # the y position of the slipface:
    rot_contour[rot_contour[:, 1] > slipface_y, :] = np.nan
    
    # Find the tail as the point furthest away from the slipface
    # on the side of the dune opposite from the slipface
    tail_idx = farthest_point_on_contour(rot_contour, [slipface_x, slipface_y])
    
    tail_idx2 = minimize_distance_from_horn_apexes(rot_contour, [left_horn_x, left_horn_y], [right_horn_x, right_horn_y])
    
    tail_x, tail_y = contour[tail_idx2]
    
    if is_draw_tail:
        plt.plot(tail_x, tail_y, 'r', markeredgewidth = 1, marker = 'x', markersize=4)
    
    return (tail_x, tail_y)

    
def calc_horns_lengths_euclidean(defect_point, horn_tips):
    '''
    Calculate the length of the horns as the euclidean distance
    between the slipface and apexes
    '''    
    horns_lengths = np.linalg.norm(defect_point - horn_tips[0]), np.linalg.norm(defect_point - horn_tips[1])
    return horns_lengths

def calc_horns_lengths_contour(contour, horn_tips, defect_point):
    '''
    Calculate the horns lengths as the distance along
    the contour between the slipface and apexes
    '''
    # Find cvx defect index:
    dist = np.sum((np.squeeze(contour) - defect_point)**2, axis = 1)
    defect_idx = np.argsort(dist)[0]

    # Find horn tips indexes:
    dist = np.sum((np.squeeze(contour) - horn_tips[0,:])**2, axis = 1)
    horn_1_idx = np.argsort(dist)[0]

    dist = np.sum((np.squeeze(contour) - horn_tips[1,:])**2, axis = 1)
    horn_2_idx = np.argsort(dist)[0]

    # Calculate horn lengths:
    if defect_idx >= horn_1_idx:
        horn_1_length = cv2.arcLength(contour[horn_1_idx:defect_idx], False)
        horn_2_length = cv2.arcLength(contour[defect_idx:horn_2_idx], False)
    else:
        horn_1_length = cv2.arcLength(contour[defect_idx:horn_1_idx], False)
        horn_2_length = cv2.arcLength(contour[horn_2_idx:defect_idx], False)

    return (horn_1_length, horn_2_length)

def wrapTo180(angle):
    angle = np.radians(angle)
    return np.degrees(np.arctan2(np.sin(angle), np.cos(angle)))

def minimize_distance_from_horn_apexes(contour, horn_apex_1, horn_apex_2):
    '''
    Minimizes the sum of distances to the horns' apexes
    to estimate the location of the dune tail
    '''
    dist1 = np.nansum((np.squeeze(contour) - horn_apex_1)**2, axis = 1)
    dist2 = np.nansum((np.squeeze(contour) - horn_apex_2)**2, axis = 1) 
    
    return np.argmax(dist1 + dist2)

def closest_point_on_contour(contour, point):
    '''
    Gets the position along the contour that is closest to the (x, y) 
    position of the input variable point
    '''
    dist = np.nansum((np.squeeze(contour) - point)**2, axis = 1)
    cnt_pt_idx = np.nanargmin(dist)
    
    return cnt_pt_idx

def farthest_point_on_contour(contour, point):
    '''
    Gets the position along the contour that is farthest away from
    the (x, y) position of the input variable point
    '''    
    dist = np.nansum((np.squeeze(contour) - point)**2, axis = 1)
    
    return abs(dist - np.nanpercentile(dist, 95, interpolation = "nearest")).argmin()
