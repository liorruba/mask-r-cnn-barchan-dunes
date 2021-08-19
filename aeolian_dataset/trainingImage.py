import numpy as np
import wget
import zipfile
from pathlib import Path
import os
import glob
import re
import rasterio as rio

class TrainingImage:
    def __init__(self, dune_field_lat, dune_field_lon):
        self.dune_field_lat = dune_field_lat
        self.dune_field_lon = dune_field_lon
        self.tile_lat = 0
        self.tile_lon = 0
        self.subtile_lat = 0
        self.subtile_lon = 0
        self.subtile_size_deg = 2
        self.subtile_image_path = ''
        self.tile_geotif = []

    def find_mosaic_tile(self):
        ''' Find tile coordinates from dune coordinates, download the tile image file if necessary'''
        # The mosaic tiles are 4x4 degrees:
        self.tile_lat = (self.dune_field_lat - self.dune_field_lat%4)
        self.tile_lon = (self.dune_field_lon - self.dune_field_lon%4)

        # Each tile is subdivided into 4 png images, each 2x2:
        self.subtile_lat = (self.dune_field_lat - self.dune_field_lat%2)
        self.subtile_lon = (self.dune_field_lon - self.dune_field_lon%2)

    def download_tile_images(self, buffer_dir_path = '.'):
        import time

        # Convert lat and lon to strings:
        tilelat_str = '{:02.0f}'.format(self.tile_lat) if self.tile_lat >= 0 else '{:03.0f}'.format(self.tile_lat)
        tilelon_str = '{:03.0f}'.format(self.tile_lon) if self.tile_lon >= 0 else '{:04.0f}'.format(self.tile_lon)
        subtilelat_str = '{:02.0f}'.format(self.subtile_lat) if self.subtile_lat >= 0 else '{:03.0f}'.format(self.subtile_lat)
        subtilelon_str = '{:03.0f}'.format(self.subtile_lon) if self.subtile_lon >= 0 else '{:04.0f}'.format(self.subtile_lon)

        # Create dir if it does not exist
        dir_path = os.path.join(buffer_dir_path, "ctx_mosaic_tiles")
        os.makedirs(dir_path, exist_ok=True)

        # Join to make filename
        filename = os.path.join(dir_path, "Murray*Lab_CTX-Mosaic_beta01_E" + subtilelon_str + "_N" + subtilelat_str + ".tif")

        # If subtile image file already exists, return
        if glob.glob(filename):
            print("File " + filename + " already exists, skipping download.")

        else:
            file_url = "http://murray-lab.caltech.edu/CTX/tiles/beta01/E" + tilelon_str + "/Murray-Lab_CTX-Mosaic_beta01_E" + tilelon_str + "_N" + tilelat_str + "_data.zip"

            print("File " + filename + " doesn't exist. Downloading file...")
            print(dir_path)
            wget.download(file_url, out = dir_path)
            print("Done.")

            # Unzip downloaded files
            print("Unzipping file...")
            with zipfile.ZipFile(os.path.join(dir_path, "Murray-Lab_CTX-Mosaic_beta01_E" + tilelon_str + "_N" + tilelat_str + "_data.zip"), 'r') as zip_ref:
                zip_ref.extractall(dir_path)
            print("Done.")

            print("Removing unnecessary files...")
            # Remove shape files, keeping only image files
            filename_without_tif_ext = glob.glob(os.path.join(dir_path, "*[!tif]"))
            for f in filename_without_tif_ext:
                os.remove(f)

            print("Done.")

        # Load image into a rasterio object:
        self.subtile_image_path = str(glob.glob(filename)[0])
        self.tile_geotif = rio.open(glob.glob(filename)[0])

    def calculate_dune_rectangle(self, shape_string):
        ''' Returns min and max coordinates defining the rectangle containing the dune '''
        shape_lat_lon_lst = list()
        # Remove commas and paratheses:
        for s in shape_string.split():
            shape_lat_lon_lst.append(re.sub(r'[^\d.-]+',"",s))

        # Reshape to to columns, convert longitude to -180,180
        shape_lat_lon_arr = np.asarray(shape_lat_lon_lst[1:],dtype=np.float32).reshape(-1,2)
        shape_lat_lon_arr[:,0] = ((shape_lat_lon_arr[:,0] + 180) % 360) - 180

        min_lon = np.min(shape_lat_lon_arr[:,0])
        max_lon = np.max(shape_lat_lon_arr[:,0])
        min_lat = np.min(shape_lat_lon_arr[:,1])
        max_lat = np.max(shape_lat_lon_arr[:,1])

        # Set the dune rectangle to be the tile border if it exceeds it
        if min_lon < self.subtile_lon: min_lon = self.subtile_lon;
        if min_lat < self.subtile_lat: min_lat = self.subtile_lat;
        if max_lon > (self.subtile_lon + 2): max_lon = self.subtile_lon + 2;
        if max_lat > (self.subtile_lat + 2): max_lat = self.subtile_lat + 2;

        # Return rectangle corners as a tuple of tuples
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.min_lat = min_lat
        self.max_lat = max_lat

    def create_bounding_box_and_label(self, r = 0.1):
        ''' Crops the image using the rectangle and image provided as inputs  '''

        # Express the coordinates of the rectangle corners as fractions of the subtile:
        # Make sure to protect the end-case in which max_lat/lon = (self.subtile_lat/lon + 2)
        frac_min_lon = self.min_lon % 2 / 2
        frac_max_lon = self.max_lon % 2 / 2 if self.max_lon < (self.subtile_lon + 2) else 1
        frac_min_lat = self.min_lat % 2 / 2
        frac_max_lat = self.max_lat % 2 / 2 if self.max_lat < (self.subtile_lat + 2) else 1

        # x and y define a cartesian coordinate system whose origin is a the bottom left corner of the tile
        min_x = int(frac_min_lon * self.tile_geotif.width)
        max_x = int(frac_max_lon * self.tile_geotif.width)
        min_y = int(frac_min_lat * self.tile_geotif.height)
        max_y = int(frac_max_lat * self.tile_geotif.height)

        # The bounding box has to be smaller than the image. Consequently, increase
        # the size of the output image by a factor r (def= 10%), times some
        # random factor to shuffle the location of the dune in the image, for better training.
        rectangle_width = (max_x - min_x)
        rectangle_height = (max_y - min_y)

        r1 = (np.random.randint(5) + 1);
        r2 = (np.random.randint(5) + 1);
        r3 = (np.random.randint(5) + 1);
        r4 = (np.random.randint(5) + 1);
        print(r1, r2, r3, r4)
        min_x_img = int(min_x - rectangle_width * r * r1)
        max_x_img = int(max_x + rectangle_width * r * r2)
        min_y_img = int(min_y - rectangle_height * r * r3)
        max_y_img = int(max_y + rectangle_height * r * r4)

        # If the minimum or maximum coordinates (x,y) exceed the subtile image, set as the image borders:
        if min_x_img < 0: min_x_img = 0
        if min_y_img < 0: min_y_img = 0
        if max_x_img > self.tile_geotif.width: max_x_img = self.tile_geotif.width
        if max_y_img > self.tile_geotif.height: max_y_img = self.tile_geotif.height

        # Bounding box for YOLO
        # YOLO requires the normalized bounding box center, height and width in the image coordinates
        rectangle_center_x = ((max_x + min_x) / 2) - min_x + rectangle_width * r * r1
        rectangle_center_y = ((max_y + min_y) / 2) - min_y + rectangle_height * r * r3

        bounding_box_img = np.flipud(self.tile_geotif.read(1))[min_y_img:max_y_img, min_x_img:max_x_img]
        return (bounding_box_img, (rectangle_center_x, rectangle_center_y, rectangle_width, rectangle_height))

    def divide_subtile(self, req_size):
        '''
        Divide the subtile into sub-image, in order to download the
        entire CTX mosaic for image detection.

        "Sub-images" are cropped and resized to maintain aspect ratio after being
        projected in cylindrical coordinates in the original CTX mosaic.

        Input:
        req_size: the dimensions of the square image, that is the fraction of
                  the subtile used for detection.
        Output:
        image_with_coord: a tuple containing the sub-image, the min and max
        longitudes and the min and max latitudes of the sub-image:
        (the sub-image, min_lon, max_lon, min_lat, max_lat)
        '''
        # Import stuff
        import cv2
        import matplotlib.pyplot as pp

        # Read subtile image
        im = cv2.imread(self.subtile_image_path)[:,:,0]

        # Flip images s.t. south is top:
        im = np.flipud(im)

        # In order to correctly project the sub-images (fractions of the subtile),
        # need to divide the horizontal distance by cosine of the latitude of the
        # sub-image.
        # Get the subtile dimensions
        subtile_height, subtile_width = np.shape(im)

        # The sub-image height
        req_height = req_size
        # The number of subimages that fit vertically in the subtile
        num_of_images_fit_vertically = np.int(np.ceil(subtile_height / req_height))

        # Store results in a list of tuples
        res_list = list()

        # Need to keep max lat and lon in memory for later, in order to know
        # where does the next image start from:
        max_lat = self.subtile_lat

        # Iterate over all sub-images, horizontally and vertically:
        for i in np.arange(num_of_images_fit_vertically):
            # Start from 90S and work the way up to 90N
            subimage_center_lat = max_lat + self.subtile_size_deg * req_height / subtile_height / 2

            # The sub-image width is a function of the latitude (due to the projection)
            req_width = np.int(req_size / np.cos(np.deg2rad(subimage_center_lat)))

            # The number of subimages that fit horizontally in the subtile (in each latitude)
            num_of_images_fit_horizontally = np.int(np.ceil(subtile_width / req_width))

            # Need to keep max lat and lon in memory for later, in order to know
            # where does the next image start from:
            max_lon = self.subtile_lon

            for j in np.arange(num_of_images_fit_horizontally):
                # Handle corner
                if (i + 1) * req_height > subtile_height and (j + 1) * req_width > subtile_width:
                    subimage = im[i * req_height : subtile_height, j * req_width : subtile_width]
                    subimage_width = subtile_width - j * req_width
                    subimage_height = subtile_height - i * req_height

                # Handle horizontal edge:
                elif (j + 1) * req_width > subtile_width:
                    subimage = im[i * req_height : (i + 1) * req_height, j * req_width : subtile_width]
                    subimage_width = subtile_width - j * req_width
                    subimage_height = req_height

                # Handle vertical edge:
                elif (i + 1) * req_height > subtile_height:
                    subimage = im[i * req_height : subtile_height, j * req_width : (j + 1) * req_width]
                    subimage_width = req_width
                    subimage_height = subtile_height - i * req_height

                else:
                    subimage = im[i * req_height : (i + 1) * req_height, j * req_width : (j + 1) * req_width]
                    subimage_width = req_width
                    subimage_height = req_height

                # Pad if needed, and resize to req_size:
                if subimage_height < req_height:
                    subimage = cv2.copyMakeBorder(subimage, 0, req_height - subimage_height, 0,
                                                 0, cv2.BORDER_CONSTANT,
                                                 value = 0)
                if subimage_width < req_width:
                    subimage = cv2.copyMakeBorder(subimage, 0, 0, 0,
                                                 req_width - subimage_width, cv2.BORDER_CONSTANT,
                                                 value = 0)

                subimage = cv2.resize(subimage, (req_size, req_size), interpolation = cv2.INTER_LINEAR) # Resize to mainain aspect ratio

                # Calculate the minimum and maximum longitudes and latitudes:
                subimage_width_deg = subimage_width / subtile_width * self.subtile_size_deg
                subimage_height_deg = subimage_height / subtile_height * self.subtile_size_deg

                subimage_center_lon = max_lon + self.subtile_size_deg * subimage_width / subtile_width / 2

                min_lon = subimage_center_lon - subimage_width_deg / 2
                max_lon = subimage_center_lon + subimage_width_deg / 2
                min_lat = subimage_center_lat - subimage_height_deg / 2
                max_lat = subimage_center_lat + subimage_height_deg / 2

                # Append tuple to list of results
                res_list.append((subimage, min_lon, max_lon, min_lat, max_lat))

                ## Optional: save as image:
                # subtile_name = str(self.subtile_lat) + "_" + str(self.subtile_lon)
                # pp.imsave('./buff/' + subtile_name + '_' + str(i) + '_' + str(j) + '.jpg', subimage, format = 'jpg', cmap = 'gray')

        # Remove tile file from buffer directory:
        if os.path.exists(self.subtile_image_path):
            os.remove(self.subtile_image_path)

        # Return tuple:
        return res_list
