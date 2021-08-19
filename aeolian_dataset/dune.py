import numpy as np
import cv2
from matplotlib import pyplot as plt
import circle_fit as cf
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

class dune:
    def __init__(self):
        contour = np.nan
        slipface = np.nan
        horn_tips = np.nan
        tail = np.nan
        width = np.nan
        length = np.nan
        normal_to_slipface = np.nan

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
            img = cv2.drawContours(img, cnt, -1, cnt_color, 2)

    return contour_list
