from matplotlib import pyplot as plt
import numpy as np
from detect_dune_field_direction import *
from analyze_dune_morph import *
from myutils import *

def calc_wind_diagram(image, predictions, draw_diagram = True,
                      draw_contours = True, draw_vectors = False, remove_outliers = True):
    # Create a new figure
    fig, ax = plt.subplots(figsize=(12,12))

    if draw_diagram:
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()

#     Get wind and sand flux directions
    (unit_vecs_prim_wind,
     normalized_cvx_depth,
     horns_length,
     dune_elp,
     cvx_defect_ratio,
     dune_dimensions) = detect_dune_field_direction(image,
                                                    predictions,
                                                    draw_contours,
                                                    draw_vectors)
    plt.imshow(image)

    # Compute the azimuths from the vectors
    azimuths_of_sand_flux = list()
    azimuths_of_prim_wind = list()

    if np.all(np.isnan(unit_vecs_prim_wind)):
        plt.close('all')

        return (np.count_nonzero(np.isnan(unit_vecs_prim_wind)), np.nan, np.nan,
                dune_dimensions, normalized_cvx_depth, horns_length, dune_elp, [np.nan], fig)

    for u in unit_vecs_prim_wind:
        azimuths_of_prim_wind.append(np.arctan2(u[1], u[0]))

    number_of_dunes_used = np.count_nonzero(~np.isnan(azimuths_of_prim_wind))
#     number_of_dunes_used = np.size(azimuths_of_prim_wind)

    azimuths_of_sand_flux = np.mod(np.array(azimuths_of_sand_flux), 2 * np.pi)
    azimuths_of_prim_wind = np.mod(np.array(azimuths_of_prim_wind), 2 * np.pi)
    normalized_cvx_depth = np.array(normalized_cvx_depth)

    if remove_outliers:
        # find indices of not-outliers:
        not_outliers_wind = reject_outliers(np.radians(azimuths_of_prim_wind))
    else:
        not_outliers_wind = np.ones_like(azimuths_of_prim_wind)

    mean_azimuth_of_prim_wind = np.rad2deg(angle_mean(azimuths_of_prim_wind[not_outliers_wind],
                                                      normalized_cvx_depth[not_outliers_wind]))


    std_azimuth_of_prim_wind = weighted_std(azimuths_of_prim_wind, normalized_cvx_depth)

    if draw_diagram:
        # Make a polar wind plot (rose diagram)
#         ax = plt.axes([0.19, 0.16, 0.2, 0.2], projection = 'polar')
        ax = plt.axes([0.1, 0.08, 0.2, 0.2], projection = 'polar')

        # Create the wind direction density function
        bins = np.linspace(0,2 * np.pi, 16)
        hy_nc, hx_nc = np.histogram(azimuths_of_prim_wind, bins, density = True)

        # Plot the density function
        ax.bar((hx_nc[:-1] + hx_nc[1:])/2, hy_nc,
               bottom=0.0, width = np.pi/8, alpha=0.25, edgecolor='black', color='blue')

        # Plot the average / mode of wind direction
        plt.arrow(np.deg2rad(mean_azimuth_of_prim_wind), 0, 0, 1, color = 'b', lw = 2)

        plt.legend(['Normal to contour\nat slipface\n(primary wind direction)'], bbox_to_anchor=(1.1,0.4), loc="upper left", fontsize = 12)
        ax.tick_params(labelsize = 20, labelcolor = 'white')
        ax.set_rticks([0, 0.5, 1])
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
        ax.set_yticklabels([0, 0.5, 1], fontsize = 10, color = 'k')

        plt.close('all')


    return (number_of_dunes_used, mean_azimuth_of_prim_wind, std_azimuth_of_prim_wind,
            dune_dimensions, normalized_cvx_depth, horns_length, dune_elp, cvx_defect_ratio, fig)
