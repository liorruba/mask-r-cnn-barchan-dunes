import os
import shutil
import sys
import random
import math
import re
import time
import scipy
import pandas as pd
import numpy as np
import tensorflow as tf
from trainingImage import TrainingImage
import os, glob
import cv2
from scipy.sparse import lil_matrix
from urllib.request import urlopen
import json
from labelbox import Client

def arc_len(radius, central_angle_in_deg):
    return pi / 180 * radius * central_angle_in_deg


# Root directory of the project
ROOT_DIR = "/home/liorr/Mask_RCNN/mrcnn"
# Model weights path
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Import Mask RCNN
sys.path.append(ROOT_DIR)
import utils
import visualize
from visualize import display_images
import model as modellib
from model import log

# Import model class
import aeolian

# Import image processing functions
from analyze_dune_morph import *


config = aeolian.AeolianConfig()
AEOLIAN_DIR = os.path.join(ROOT_DIR, "aeolian_dataset/dataset")

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_CHANNEL_COUNT = 3

config = InferenceConfig()

API_KEY = ""
client = Client(API_KEY)
proj = client.get_project(")

print("Exporting label data to json...")
export_url = proj.export_labels()
print("Done.")

# Get json

with urlopen(export_url) as url:
    data = json.loads(url.read().decode())


# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:1"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)


AEOLIAN_WEIGHTS_PATH = "../mask_rcnn_aeolian_016__0113.h5"


weights_path = AEOLIAN_WEIGHTS_PATH

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)


##########
##########
#        #
# DETECT #
#        #
##########
##########


#### Range of mosaic tiles longitude and latitude to iterate over:
data_dir_path = "/data2/lapotre/liorr/detection_results/"
subtile_lon = np.arange(int(sys.argv[1]), int(sys.argv[2]), 2)
subtile_lat = np.arange(int(sys.argv[3]), int(sys.argv[4]), 2)
print(subtile_lat)

# Image size
req_size = 832

prediction_filename = os.path.join(data_dir_path, "predictions_" + str(subtile_lat[0]) + "_" + str(subtile_lat[-1] + 2) + "_" + str(subtile_lon[0]) + "_" + str(subtile_lon[-1] + 2) + ".npy")
print(prediction_filename)

# Create a list of prediction
detection_prediction_list = list()

# Restart from last subtile:
if os.path.exists(prediction_filename):
    print('Restoring detections from file.')
    predictions = np.load(prediction_filename)
    init_lon = predictions[-1][1][0]
    init_lat = predictions[-1][1][2]
    detection_prediction_list = list(predictions)

else:
    init_lon = subtile_lon[0]
    init_lat = subtile_lat[0]

for lat in subtile_lat:
    for lon in subtile_lon:
        # Skip already looked at tiles:
        if lat < init_lat and lon < init_lon:
            continue

        print('Splitting tile: latitude ' + str(lat) + ' and longitude ' + str(lon) + '.')

        # Call the class constructor on the center of the mosaic subtile
        train_image = TrainingImage(lat, lon)
        train_image.find_mosaic_tile()
        # Download the image
        train_image.download_tile_images(buffer_dir_path = data_dir_path)

        # Cut the subtile into subimages. Method returns subimages, a list of tuples
        # with the subimage, the min and max longitude and latitude of the image
        subimages = train_image.divide_subtile(req_size)

        # For each image, detect objects and save masks
        for subimg in subimages:
            # Extend image dimension
            buff = subimg[0]
            buff = buff[:,:,np.newaxis]

            # Detect:
            detection_prediction = model.detect([buff], verbose = 0)

            # Save into object only if detected something:
            if np.size(detection_prediction[0]['masks']) == 0:
                detection_prediction_list.append([[np.nan],
                                                  subimg[1:5],
                                                  [np.nan]])
            else:

                detection_prediction_list.append([cv2.resize(subimg[0], (416, 416)),
                                                  subimg[1:5],
                                                  detection_prediction[0]])

    # Save results to file a few times:
    with open(prediction_filename, "wb") as f:
        np.save(f, detection_prediction_list)
