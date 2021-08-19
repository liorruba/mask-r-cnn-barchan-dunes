from matplotlib import pyplot as plt
import numpy as np
from analyze_dune_morph import *
from myutils import *

def detect_dune_field_direction(image, p, draw_contours, draw_vectors):
    is_show_image = draw_contours or draw_vectors

    if np.size(p['class_ids']) < 2:
        print("No dunes detected.")
        return

    # Get contours:
    cnts = get_contours(image, p, True, 0.7, draw_contours)

    # Fit ellipses to get dunes' dimensions, to constrain the uncertainty:
    elp = best_fit_ellipse(cnts, image, False)

    # Start by assuming the crest can be in all directions:
    azi_range = (-180, 180)

    # Iterate until the standard deviation converges:
    # Initialize the previous std of the crest azimuth:
    prev_azi_std = np.nan
    eps = 1;

    # Initialize lists
    slipfaces = list()
    depths = list()

    while True:
        (slipfaces,
         azimuths_rel_to_center,
         depths,
         cvx_defect_depth_ratio) = detect_slipface_from_shape(cnts,
                                                              image = image,
                                                              is_return_image = True,
                                                              a_priori_azimuth = azi_range,
                                                              num_of_defects = 10)

        # If all detected slipfaces are nan (no slipfaces were detected)
        if np.all(np.isnan(depths)):
            return([np.nan], [np.nan], [np.nan], [np.nan], np.nan, (np.nan, np.nan))

        # Get uncertainty from depth of convex defect
        normalized_cvx_depth = normalized_convex_defect_depth(elp, depths)
        next_azi_std = weighted_std(azimuths_rel_to_center, normalized_cvx_depth)
        next_azi_range = np.rad2deg(angle_mean(np.deg2rad(np.array(azimuths_rel_to_center)), wts = normalized_cvx_depth))

        if (np.size(azi_range) == 1) and (np.abs(azi_range - next_azi_range) < eps):
            break

        else:
            azi_range = np.rad2deg(angle_mean(np.deg2rad(np.array(azimuths_rel_to_center)), wts = normalized_cvx_depth))
            prev_azi_std = weighted_std(azimuths_rel_to_center, normalized_cvx_depth)

            if np.all(np.isnan(azi_range)):
                return (np.nan, np.nan, np.nan)

    # Run one last time, to draw the dunes
    detect_slipface_from_shape(cnts, image = image,
                            is_return_image = is_show_image,
                            a_priori_azimuth = azi_range)



    sand_flux_vec, prim_wind_vec, dune_horns_lengths, dune_tails = get_slipface_normals(cnts,
                                                                                        slipfaces, img = image,
                                                                                        is_draw_normals = draw_vectors,
                                                                                        horns_threshold = 1.5)

    # Calculate dunes widths:
    for (c, s, t) in zip(cnts, slipfaces, dune_tails):
        (right_horn, left_horn) = get_dune_horn_apexes(c, s)

        if np.any(np.isnan(np.array([right_horn, left_horn]))):
            dune_width = np.nan
        else:
            dune_width = np.linalg.norm(right_horn - left_horn)

        if np.any(np.isnan(t)):
            dune_length = np.nan
        else:
            (tail_x, tail_y) = t
            dune_length = np.linalg.norm(np.array([tail_x, tail_y]) - s)

    # Remove nans to make all arrays the same length
    normalized_cvx_depth = np.array(normalized_cvx_depth)

    dune_dimensions = (dune_width, dune_length)
    return (prim_wind_vec,
            normalized_cvx_depth,
            dune_horns_lengths,
            elp,
            cvx_defect_depth_ratio,
            dune_dimensions)
