import numpy as np
import cv2
from matplotlib import pyplot as plt
import circle_fit as cf
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

# Gets the contours of an image with prediction data from mask_rcnn
# pass only_barchans to only return the contours of barchan dunes
# pass confidence to only return the contours of dunes with score > confidence
# pass is_draw_contour to also draw the contours on the image
def get_contours(img, prediction_data,
                    only_barchans = True, confidence = 0.6,
                    is_draw_contour = False, cnt_color = (255, 255, 255)):
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
                cv2.CHAIN_APPROX_SIMPLE
            )

        # If object answers the criteria:
        if np.ndim(np.squeeze(cnt)) == 1:
            # Self-intersecting contour, skip:
            continue
        else:
            if np.shape(cnt)[1] > 4:
                np.array(contour_list.append(np.squeeze(cnt)))
            else:
                continue

        if is_draw_contour:
            img = cv2.drawContours(img, cnt, -1, cnt_color, 4)

    return contour_list

# Fit ellipses to given contours
# pass image and is_return_image = True to draw the ellipses
# outputs a cv2 ellipse object
def best_fit_ellipse(contours, image, is_return_image = False):
    ellipse_list = list()

    if np.ndim(contours) == 2:
        elp = cv2.fitEllipse(np.squeeze(contours))
        ellipse_list.append(elp)
        return ellipse_list

    for cnt in contours:
        if np.size(cnt) > 3:
            elp = cv2.fitEllipse(np.squeeze(cnt))
            ellipse_list.append(elp)
        else:
            ellipse_list.append([np.nan])

        if is_return_image:
            image = cv2.ellipse(image, elp, (255, 255, 255), 2)

    return ellipse_list

# Fit triangles to given contours
# pass image and is_return_image to draw triangles
# outputs the tirangles vertices
def best_fit_triangle(contours, image, is_return_image = False):
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
    M = cv2.moments(contour)
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

# detect the crest from the dune shape using the convex defect method
# recieves the dune contours. pass a_priori_azimuth tuple (min,max)
# to only search for convex defects in some angle range
# Also pass image and is_return_image to draw the defects on the image.
def detect_slipface_from_shape(contours, image, is_return_image,
                            a_priori_azimuth = (-180, 180),
                            num_of_defects = 10):
    contour_defects_list = list()
    azimuth_list = list()
    depth_list = list()
    cvx_defect_ratio_list = list()

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
        norm_cvx_defect_depth = normalized_convex_defect_depth(elp, defect_depths[0])[0]
        # Select only the deepest defect within the chosen azimuth
        azimuth_in_range = np.where(
                                    (azimuth >= a_priori_azimuth[0]) &
                                    (azimuth <= a_priori_azimuth[1])
                                    )[0]

        # If no convex defects in range or the dune is not barchan (maximum
        # convex defect is not p times greater than second deepest, append nan:
        if ((np.size(azimuth_in_range) > 0) and
            (cvx_defect_ratio < 0.75) and
            (norm_cvx_defect_depth > 0.05)):
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
            # the deepest convex defect. This difference (75%) is arbitary,
            # and is thus hardcoded.
            if (max_concave_depth / np.max(defect_depths) < 0.75):
                max_idx = np.argmax(defect_depths)
                max_defect = np.array(defects)[max_idx]
                max_concave_depth = np.array(defect_depths)[max_idx]

            # Add to lists
            contour_defects_list.append(max_defect)
            azimuth_list.append(max_concave_az)
            depth_list.append(max_concave_depth)
            cvx_defect_ratio_list.append(cvx_defect_ratio)

            # If the user opted to return an image, render it
            if is_return_image:
                # Uncomment to draw dunes contours
                # image = cv2.drawContours(image, [cnt], -1, (255, 255, 255), 3)
                # Uncomment to draw the convex hull polygon
                # image = cv2.polylines(image, [cnt[np.squeeze(hull)]], True, (255,255,255))
                # Uncomment to draw contour center
                # image = cv2.circle(image, (cx, cy), 5,(0, 255, 255), -1)
                # Uncomment to show crests
                # image = cv2.circle(image, tuple(max_defect), 5 ,(255, 255, 0), 3)
                pass

            else:
                image = {}

        else:
            contour_defects_list.append(np.nan)
            azimuth_list.append(np.nan)
            depth_list.append(np.nan)
            cvx_defect_ratio_list.append(np.nan)


    return contour_defects_list, azimuth_list, depth_list, cvx_defect_ratio_list

def get_dune_albedo_gradient(image, gauss_smooth = 5):
    buff = np.double(image)
    buff = cv2.GaussianBlur(buff, (gauss_smooth, gauss_smooth), 0)
    gx,gy = np.gradient(buff)

    return gx, gy

def normalized_convex_defect_depth(ellipses, depths):
    # Return a list of normalize defect depths (proxy for certainty):
    normalized_depth_list = list()

    if np.size(depths) == 1:
        maj_ax_len = ellipses[0][1][1]
        # fix a bug related to depths sometimes being passed as a list
        depths = np.squeeze(np.array([depths]))
        normalized_depth = depths / 256 / (maj_ax_len)
        # print(ellipses)
        # print(semi_maj_ax_len)
        # print(depths / 256)
        # print(normalized_depth)
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
    if np.size(depth_list) == 1:
        return 1

    return depth_list[1] / depth_list[0]

def get_slipface_normals(contours, convex_defects_list, img, is_draw_normals = False, horns_threshold = 1.5):
    sand_flux_vector_list = list()
    prim_wind_vector_list = list()
    horns_lengths_list = list()
    tail_list = list()

    for cnt, mc in zip(contours, convex_defects_list):
        sand_flux_vector, prim_wind_vector, horns_lengths = normal_to_slipface(mc, cnt, img, horns_threshold)

        # print(cnt)
        tail = get_dune_tail(cnt, mc, np.arctan2(-prim_wind_vector[1], prim_wind_vector[0]))

        sand_flux_vector_list.append((sand_flux_vector[0], -sand_flux_vector[1]))
        prim_wind_vector_list.append((prim_wind_vector[0], -prim_wind_vector[1]))
        horns_lengths_list.append(horns_lengths)
        tail_list.append(tail)

        if is_draw_normals and not np.any(np.isnan(mc)):
            # print(mc[0], mc[1], prim_wind_vector[0], -prim_wind_vector[1])
            plt.quiver(mc[0], mc[1], prim_wind_vector[0], -prim_wind_vector[1], headwidth = 3,
                        headlength = 3, headaxislength = 2, minshaft = 1,
                        width = 5e-6 * img.shape[0], color = 'w')

    return sand_flux_vector_list, prim_wind_vector_list, horns_lengths_list, tail_list

def get_dune_horn_apexes(dune_contour, defect_point):
    '''
    Estimate the length of the side of the dune
    on which the slipface is found
    '''
    # Iterative method to find the most approporiate contour
    # Run until the convex defect sits on (5 pixels from) the
    # approximate contour
    # Initialize the distance
    dist = 11
    epsilon = 0.1 # fraction of the length of the contour
    approx_contour = np.empty(1)

    while np.min(dist) > 10 and epsilon > 0.001 or len(approx_contour) < 5:

        approx_contour = cv2.approxPolyDP(dune_contour, epsilon * cv2.arcLength(dune_contour, True), True)
        approx_contour = np.squeeze(approx_contour)

        dist = np.sum((np.squeeze(approx_contour) - defect_point)**2, axis = 1)
        epsilon = epsilon / 2

    ##########
    # Uncomment to draw the approximated contours
    # img = cv2.drawContours(img, [approx_contour], -1, (0, 0, 255), 3)
    ##########

    # Find the point on the approximated contour closest to the defect
    closest_idx = np.argmin(dist)

    # The horn edges are the points on the approximated contour
    # before and after the defect
    crest_edges_idx_before = (closest_idx - 1) % len(approx_contour)
    crest_edges_idx_after = (closest_idx + 1) % len(approx_contour)

    right_horn = approx_contour[crest_edges_idx_before]
    left_horn = approx_contour[crest_edges_idx_after]

    return np.array([right_horn, left_horn])

def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def get_dune_tail(contour, defect_point, normal_to_slipface_angle):
    # First, rotate the contour such that the slipfact is oriented "down":
    x = contour[:,0]
    y = contour[:,1]

    if np.isnan(normal_to_slipface_angle):
        return (np.nan, np.nan)

    R = rotation_matrix(-normal_to_slipface_angle + np.pi / 2)

    xp = list()
    yp = list()

    for (xi, yi) in zip(x, y):
        (xbuff, ybuff) = np.matmul(np.array([xi, yi]), R)
        (x_def, y_def) = np.matmul(np.array([defect_point[0], defect_point[1]]), R)
        xp.append(xbuff)
        yp.append(ybuff)

    xp = np.array(xp)
    yp = np.array(yp)

    # Truncate "a little bit" (2 pixels) above the defect
    xp = xp[yp > y_def + 2]
    yp = yp[yp > y_def + 2]

    if np.size(xp) < 5 or np.size(yp) < 5:
        return (np.nan, np.nan)

    p = np.polyfit(xp, yp, 2)
    # The peak of the parabola is the dune tail
    tail_x = -p[1] / 2 / p[0]
    fi = interp1d(xp, yp)
    xpp = np.linspace(np.min(xp), np.min(xp))
    if (tail_x < np.min(xp)) | (tail_x > np.max(xp)):
        return (np.nan, np.nan)

    tail_y = fi(tail_x)

    return (tail_x, tail_y)


def normal_to_slipface(defect_point, contour, img, horns_threshold):
    '''
    Calculates the angle normal to the contour, but also returns the
    length of the horns, calculated as the distances of the horn edge
    to the convex defect
    '''
    contour = np.squeeze(contour)

    # If the convex defect
    if np.any(np.isnan(defect_point)):
        return (np.nan, np.nan), (np.nan, np.nan), (0, 0)

    # Find the bisector vector
    horn_tips = get_dune_horn_apexes(contour, defect_point)

    triangle_sides_vecs = horn_tips - defect_point
    triangle_sides_len = np.sqrt(np.sum( triangle_sides_vecs ** 2 , axis = 1 ))

    horns_lengths = calc_horns_lengths(contour, horn_tips, defect_point)
    long_horn_direction = triangle_sides_vecs[np.argmax(horns_lengths)]

    bisector_vector = np.linalg.norm(triangle_sides_vecs[0]) * triangle_sides_vecs[1] + np.linalg.norm(triangle_sides_vecs[1]) * triangle_sides_vecs[0]

    # The sand flux is in the direction of the long horn
    if (np.max(triangle_sides_len) / np.min(triangle_sides_len) > horns_threshold):
        sand_flux_direction_vector = long_horn_direction
    else:
        sand_flux_direction_vector = bisector_vector

    # The primary wind direction is always the bisector
    prim_win_direction_vector = bisector_vector

    # Uncomment to plot the first and last point in the detected crest
    # plt.plot(horn_tips[0,0], horn_tips[0,1], color = 'lightgreen', markeredgewidth = 3, marker='x', markersize=7)
    # plt.plot(horn_tips[1,0], horn_tips[1,1], color = 'lightgreen', markeredgewidth = 3, marker = 'x', markersize=7)

    if not np.any(np.isnan(horn_tips)):
        p = (horn_tips[0,1] - horn_tips[1,1]) / (horn_tips[0,0] - horn_tips[1,0])
    else:
        p = np.nan

    return sand_flux_direction_vector, prim_win_direction_vector, horns_lengths

def calc_horns_lengths(contour, horn_tips, defect_point):
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

# def get_dune_horn_tips_curv(dune_contour, defect_point):
#     '''
#     Iterate along the contour from the detected slipface (convexity defect)
#     until the angle between three consecutive points is greater than pi
#     '''
#
#     #
#     # u_new = np.linspace(u.min(), u.max(), 20)
#     #
#     #
#     # dune_contour = np.array([x_new, y_new]).T
#
#     dist = np.sum((np.squeeze(dune_contour) - defect_point)**2, axis = 1)
#
#     # Find the point on the approximated contour closest to the defect
#     cvx_defect_idx = np.argmin(dist)
#
#     # Iterate along the apporximate contour:
#     segment_length = 5
#     radius_list = list()
#
#     for idx in range(len(dune_contour)):
#         idxs = np.arange(idx, idx + segment_length) % len(dune_contour)
#         segment_x = dune_contour[:,0][idxs]
#         segment_y = dune_contour[:,1][idxs]
#
#         if len(segment_x) < 3:
#             rad = np.inf
#         else:
#             xc, yc, rad, _ = cf.least_squares_circle(([segment_x, segment_y]))
#
#         radius_list.append(rad)
#
#     curv = 1/np.array(radius_list)
#     x = dune_contour[:,0]
#     dx = np.gradient(x)
#     d2x = np.gradient(dx)
#     y = dune_contour[:,1]
#     dy = np.gradient(y)
#     d2y = np.gradient(dy)
#
#     # curv = np.abs(dx * d2y - dy * d2x) / (dx ** 2 + dy ** 2) ** (3/2)
#
#     # Update the convex defect
#
#     plt.close('all')
#     # plt.figure()
#     # plt.scatter(dune_contour[:,0], dune_contour[:,1], c=1/np.array(radius_list))
#     # plt.set_cmap('viridis')
#     # for i in range(0,len(dune_contour),3):
#     #     plt.text(dune_contour[i,0], dune_contour[i,1], str(i))
#
#     plt.show()
#     plt.figure()
#     smoothing = 0
#     x_curv = range(len(dune_contour))
#     curv_smooth = gaussian_filter(curv, smoothing)
#     dcurv_smooth = np.gradient(curv_smooth)
#     d2curv_smooth = np.gradient(dcurv_smooth)
#     plt.subplot(131)
#     plt.plot(x_curv, curv_smooth,'.')
#     plt.subplot(132)
#     plt.plot(x_curv, dcurv_smooth,'.')
#     plt.subplot(133)
#     plt.plot(x_curv, d2curv_smooth,'.')
#     plt.show()
#     plt.figure()
#
#     curv_maxima = find_peaks(curv)[0]
#     curv_minima = find_peaks(-curv)[0]
#     dcurv_maxima = find_peaks(dcurv_smooth)[0]
#     dcurv_minima = find_peaks(-dcurv_smooth)[0]
#     d2curv_maxima = find_peaks(d2curv_smooth)[0]
#     d2curv_minima = find_peaks(-d2curv_smooth)[0]
#     # print(d2curv_maxima)
#     #
#     # saddle_max_curvature = np.argmax(np.abs(dcurv_smooth[dcurv_minima]))
#     # contour_saddle_points = dune_contour[dcurv_minima[saddle_max_curvature]]
#     # contour_maxima = dune_contour[curv_maxima]
#     # contour_minima = dune_contour[curv_minima]
#
#     horn_tips_idx = d2curv_maxima[np.argmin(cvx_defect_idx - d2curv_maxima)]
#     horn_tips_idx = d2curv_maxima[np.argmin(d2curv_maxima - cvx_defect_idx)]
#     horn_tips = dune_contour[horn_tips_idx]
#     # print(cvx_defect_idx)
#     # plt.plot(dune_contour[cvx_defect_idx][0], dune_contour[cvx_defect_idx][1],'xr')
#     return np.array(horn_tips)
