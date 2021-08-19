"""
Use Mask R-CNN to detect dunes and TARs on Mars
by
Lior Rubanenko
Stanford University, 2020
-
Mask R-CNN by Matterport, Inc.
Licensed under the MIT License

Class and script based on https://github.com/matterport/Mask_RCNN/blob/v2.1/samples/balloon/balloon.py
---

Usage:
TBC


"""

import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import skimage
import requests
import cv2
import imgaug.augmenters as iaa
from labelbox import Client
from urllib import request
import json

# Labelbox project API_KEY
API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJja2NlMTFkMGVmbW1vMDczNXV2dndlcWZxIiwib3JnYW5pemF0aW9uSWQiOiJja2NlMTFjenlmbW1pMDczNTMycW43ZXlpIiwiYXBpS2V5SWQiOiJja2V5b29sbzI5MDhpMDcwOGZ6ZHU3d3lsIiwiaWF0IjoxNTk5ODU1NTQzLCJleHAiOjIyMzEwMDc1NDN9.qwaVnPCckrcFN6aBddSFN8PD9xVP2hKpTFr12U2TIW0'
PROJECT_ID = 'ckce2im26y3bv0749sq1k5xjw'

# Root directory of the project
PROJECT_ROOT_DIR = os.getcwd()
MASK_RCNN_ROOT_DIR = os.path.dirname(os.path.dirname(PROJECT_ROOT_DIR))

# Import Mask RCNN
sys.path.append(MASK_RCNN_ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(PROJECT_ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = os.path.join(PROJECT_ROOT_DIR, "logs")

############################################################
#  Config
############################################################

class AeolianConfig(Config):
    """
    Configure the aeolian topographic features dataset. This method derives from
    config and overrides some values.
    """
    # Give the configuration a recognizable name. NOTE: Cannot contain numbers
    # due to some regex later on in model.set_log_dir(). TODO fix:
    NAME = 'aeolian_024_'

    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 3  # Background + Barchans + TARs

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 100

    # Skip detections < some confidence
    DETECTION_MIN_CONFIDENCE = 0.6

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # Max image size after resizing
    IMAGE_MAX_DIM = 832

    # Use a shallower network since not a lot of classes
    BACKBONE = "resnet101"

    # Learning rate and mometum
    LEARNING_RATE_1 = 1e-3
    LEARNING_RATE_2 = 1e-5
    LEARNING_RATE_3 = 1e-7

    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.000001

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 2,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    # Image mean (RGB)
    MEAN_PIXEL = np.array([0])

    # Handle grayscale
    IMAGE_CHANNEL_COUNT = 3
    MEAN_PIXEL = np.array([112, 112, 112])

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    # Augmenation probability
    AUGMENTATION_PROB = 0.5

    # Epoch legs (head, +4 or just all)
    FIRST_LEG = 'all'
    SECOND_LEG = 'all'

############################################################
#  Dataset
############################################################

class AeolianDataset(utils.Dataset):

    # The class constructor calls the super's constructor and makes the dirs
    def load_aeolian(self, dataset_dir, subset, use_labelbox = False, download_data = False):
        """
        Load a subset of the aeolian features dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes (0 is reserved for background).
        self.add_class("aeolian", 1, "barchan_dune")
        self.add_class("aeolian", 2, "TAR")

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Since we used Labelbox to label the images, the masks are already stored online
        annotation_dataframe = pd.read_json(os.path.join(dataset_dir, "annotations.json"))

        if use_labelbox:
            # Load annotations
            for annot_idx in annotation_dataframe.keys():
                # Check if need to skip the image by checking the [objects] key
                # exists, which means nothing was labeled.
                if "objects" not in annotation_dataframe[annot_idx]['Label']:
                    # Skip images (no labels)
                    continue

                ## Parse the json:
                # Read image ID:
                image_ID = annotation_dataframe[annot_idx]['ID']

                # Read image URI
                image_URI = annotation_dataframe[annot_idx]['Labeled Data']

                # Get masks URI
                instance_mask_URI_list = list()
                class_id_list = list()

                for obj in annotation_dataframe[annot_idx]['Label']['objects']:
                    instance_mask_URI_list.append(obj['instanceURI'])

                    if obj['value'] == 'barchan_dune':
                        class_id_list.append(1)

                    if obj['value'] == 'tar':
                        class_id_list.append(2)

                ## Add to database:
                self.add_image(
                    "aeolian",
                    image_id = image_ID, # each image has its own id
                    path = image_URI, # each image has its own URI
                    mask_path = instance_mask_URI_list, # each image has its set of instance mask URIs
                    class_id = class_id_list # each image has its own set of instance class
                    )

        else:
            # Download data
            for annot_idx in annotation_dataframe.keys():
                # Check if need to skip the image by checking the [objects] key
                # exists, which means nothing was labeled.
                if "objects" not in annotation_dataframe[annot_idx]['Label']:
                    # Skip images (no labels)
                    continue

                ## Parse the json:
                # Read image ID:
                image_ID = annotation_dataframe[annot_idx]['ID']

                # Read image URI
                image_URI = annotation_dataframe[annot_idx]['Labeled Data']

                # Get masks URI
                instance_mask_URI_list = list()
                class_id_list = list()

                for obj in annotation_dataframe[annot_idx]['Label']['objects']:
                    instance_mask_URI_list.append(obj['instanceURI'])

                    if obj['value'] == 'barchan_dune':
                        class_id_list.append(1)

                    if obj['value'] == 'tar':
                        class_id_list.append(2)

                img_path = os.path.join(dataset_dir, image_ID + ".jpg")

                # If need to download image
                if download_data:
                    image = skimage.io.imread(image_URI)
                    print("Saving image to file " + img_path)
                    skimage.io.imsave(img_path, image)

                # Set masks
                mask_path_list = list()
                for mask_idx, mask_uri in enumerate(instance_mask_URI_list):
                    # Set image mask
                    mask_path = os.path.join(dataset_dir, image_ID + "_mask_" + str(mask_idx) + ".png")
                    mask_path_list.append(mask_path)

                    # If need to download mask
                    if download_data:
                        mask = skimage.io.imread(mask_uri)
                        skimage.io.imsave(mask_path, mask.astype(np.uint8), check_contrast=False)

                ## Add to database:
                self.add_image(
                    "aeolian",
                    image_id = image_ID, # each image has its own id
                    path = img_path, # each image has its own URI
                    mask_path = mask_path_list, # each image has its set of instance mask URIs
                    class_id = class_id_list # each image has its own set of instance class
                    )


    # Check image is grayscale
    def isgrayscale(self, img):
        if len(img.shape) < 3:
            return True
        if img.shape[2]  == 1:
            return True

        blue, green, red = img[:,:,0], img[:,:,1], img[:,:,2]

        if np.all(red == blue) and np.all(blue == green):
            return True
        return False

    # Override load_image from utils.Database to support grayscale:
    def load_image(self, image_id, config):
        """
        Load the specified image and return a [H,W,3] or a [H,W,1] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])

        # Standardize image?
        # image = (image - np.mean(image)) / np.std(image)

        # If the image is 3-channel grayscale (all channels identical)
        # only return the first channel:
        if self.isgrayscale(image):
            if config.IMAGE_CHANNEL_COUNT == 1:
                gs_image = image[:,:,0]
                return gs_image[:,:,np.newaxis]
            elif config.IMAGE_CHANNEL_COUNT == 3:
                # Convert to RGB for consistency.
                if image.ndim != 3:
                    image = skimage.color.gray2rgb(image)

                # If has an alpha channel, remove it for consistency
                if image.shape[-1] == 4:
                    image = image[..., :3]
            else:
                print("Error: IMAGE_CHANNEL_COUNT must be 1 or 3.")

        # Normalize image
        imin = np.min(image[:,:,0])
        imax = np.max(image[:,:,0])

        image = np.int16(((image - imin)/(imax - imin)) * 255)

        return image

    def load_mask(self, image_id, config):
        """Generate instance masks for an image.
        Since we are using Labelbox to label the images, the masks are already
        stored on the server. All we need to do is download the correct mask,
        which is a key in the dictionary image_info (see utils.add_image()).
        """
        # print('Loading masks for image ' + self.image_info[image_id]['path'])

        # If not an aeloian dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "aeolian":
            return super(self.__class__, self).load_mask(image_id)

        # Download instance mask, and place in a 3D numpy array with same shape as image
        for idx, mask_path in enumerate(image_info["mask_path"]):
            # Sleep to not exceed labelbox's API QPS limit
            # time.sleep(3)

            # response = requests.get(url) ## RETRIES??

            # if not response.ok:
            #     print("WARNING: Error fetching image from Labelbox.")
            #     print(url)
            #     print(response)
            #
            #     continue

            # mask = np.asarray(bytearray(response.content), dtype="uint8")
            # Load image
            mask = skimage.io.imread(mask_path)
            # mask = cv2.imdecode(mask, cv2.IMREAD_GRAYSCALE)
            mask = cv2.normalize(mask, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

            # If first iteration, prepare the dataset:
            if idx == 0:
                # [mask_width, mask_height, num_instances]
                instance_masks_array = np.empty([np.shape(mask)[0], np.shape(mask)[1], len(image_info["mask_path"])])

            # Insert into dataset:
            instance_masks_array[:,:,idx] = mask[:,:,0]

        # Return mask, and array of class IDs of each instance.
        return instance_masks_array, np.int32(np.array(image_info["class_id"]))

    def image_reference(self, image_id):
        """Return the path of the image."""
        image_info = self.image_info[image_id]
        if image_info["source"] == "aeolian":
            return image_info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def prepare_dirs(dataset_dir, is_remove_old_annotation = False):
    """
    Accepts the main dataset dir (not train/val subdirs),
    created train and val subdirs and splits the dataset randomly.
    """
    # Prepare directories
    # Make dirs if don't exist
    if not os.path.isdir(os.path.join(dataset_dir, 'train')):
        os.mkdir(os.path.join(dataset_dir, 'train'))

    # Validation dir
    if not os.path.isdir(os.path.join(dataset_dir, 'val')):
        os.mkdir(os.path.join(dataset_dir, 'val'))

    if is_remove_old_annotation:
        # If old annotation files exist, remove them:
        if os.path.isfile(os.path.join(dataset_dir, 'train', 'annotations.json')):
            os.remove(os.path.join(dataset_dir, 'train', 'annotations.json'))

        if os.path.isfile(os.path.join(dataset_dir, 'val', 'annotations.json')):
            os.remove(os.path.join(dataset_dir, 'val', 'annotations.json'))

        # Get new data from labelbox
        get_data_from_labelbox(dataset_dir)

        # Create new annotation files:
        split_annot_train_validation(dataset_dir)

def split_annot_train_validation(dataset_dir, frac_training = 0.8):
    '''
    Split the annotation file generated by Labelbox into training and validation datasets
    '''
    annotation_dataframe = pd.read_json(os.path.join(dataset_dir, 'annotations.json'))

    if not os.path.exists(os.path.join(dataset_dir, 'train', 'annotations.json')):
        annotations_training = annotation_dataframe.sample(frac = frac_training)

    if not os.path.exists(os.path.join(dataset_dir, 'val', 'annotations.json')):
        annotations_val = annotation_dataframe.drop(annotations_training.index)

    if not os.path.exists(os.path.join(dataset_dir, 'train', 'annotations.json')):
        annotations_training.transpose().to_json(os.path.join(dataset_dir, 'train', 'annotations.json'))

    if not os.path.exists(os.path.join(dataset_dir, 'val', 'annotations.json')):
        annotations_val.transpose().to_json(os.path.join(dataset_dir, 'val', 'annotations.json'))

def get_data_from_labelbox(dataset_dir):
    '''
    Get data export json from the labelbox servers
    input is the "dataset" dir
    '''
    client = Client(API_KEY)
    proj = client.get_project(PROJECT_ID)

    print("Exporting label data to json...")
    export_url = proj.export_labels()
    print("Done.")

    # Get json
    with request.urlopen(export_url) as url:
        print("Reading data from json...")
        data = json.loads(url.read().decode())

        with open(os.path.join(dataset_dir, 'annotations.json'), 'w') as f:
            print("Writing data to file...")
            json.dump(data, f)
            print("Done.")

def write_config_to_logs_dir(aeolian_config, logs_dir):
    '''
    Write the configuation file to the output dir, so that the experiment
    information could be accessed later
    '''
    import json

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    if not os.path.exists(os.path.join(logs_dir, 'config.json')):
        # Wait for logs dir
        while not os.path.exists(logs_dir):
            print('logs dir ' + logs_dir + ' de')
            time.sleep(0.5)

        with open(os.path.join(logs_dir, 'config.json'), 'w') as file:
            json.dump(AeolianConfig.__dict__.copy(), file, cls = NumpyEncoder)
    else:
        print("Configuation file already exists.")

def train(model):
    """Train the model."""
    # Training dataset.
    print("Getting test data")
    dataset_train = AeolianDataset()
    dataset_train.load_aeolian(args.dataset, 'train', use_labelbox = False, download_data = False)
    dataset_train.prepare()
    print("Done getting test data")

    print("Getting validation data")
    # Validation dataset
    dataset_val = AeolianDataset()
    dataset_val.load_aeolian(args.dataset, 'val', use_labelbox = False, download_data = False)
    dataset_val.prepare()
    print("Done getting validation data")

    print("Training network")
    augmentations = iaa.Sequential([
        iaa.Sometimes(config.AUGMENTATION_PROB, [
            iaa.OneOf( [
                iaa.ContrastNormalization((0.5, 0.9)),
                iaa.ContrastNormalization((1.1, 1.5)),
            ])
        ]),
        iaa.Sometimes(config.AUGMENTATION_PROB, [
            iaa.OneOf( [
                iaa.Affine(rotate=45),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=135),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=225),
                iaa.Affine(rotate=270),
            ])
        ])
        ], random_order = True
    )

    # Training
    model.train(dataset_train, dataset_val,
                learning_rate = config.LEARNING_RATE_1,
                epochs = 20,
                layers = 'all',
                augmentation = augmentations)

    model.train(dataset_train, dataset_val,
                learning_rate = config.LEARNING_RATE_2,
                epochs = 70,
                layers = 'all',
                augmentation = augmentations)

    model.train(dataset_train, dataset_val,
                learning_rate = config.LEARNING_RATE_3,
                epochs = 120,
                layers = 'all',
                augmentation = augmentations)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect aeolian features.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Aeolian dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.image or args.video,\
               "Provide --image to detect"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = AeolianConfig()
    else:
        class InferenceConfig(AeolianConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights " + weights_path)
    if "coco" in args.weights.lower():
        # Exclude the last layers because they require a matching
        # number of classes
        # If grayscale, don't load the first weights layer
        if config.IMAGE_CHANNEL_COUNT == 1:
            print("Loading weights for grayscale model:")
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask", "conv1"])

        else:
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])

    else:
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

    # # Write the configuation to the logs dir
    # print("Saving configuation file to disk...")
    # write_config_to_logs_dir(config, model.log_dir)
    # print("Done.")

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "detect":
        pass

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
