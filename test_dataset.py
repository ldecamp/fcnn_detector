""" Create a class to hold the Dataset generation  """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np

DEBUG = False

class TestDataset(object):

    def __init__(self, root_dir):
        """
        Creates an instance of Dataset holder.
        Store all training and test data into memory

        root_dir: Specifies the directory to load the data from
        """
        self.images_paths = []
        self.init_dataset(root_dir)
    
    def init_directory(self, images_dir):
        """ Read all image content from input directory
        """
        for dir_name, _, file_list in os.walk(images_dir):
            for file_name in file_list:
                image_file = os.path.join(dir_name, file_name)
                self.images_paths.append(image_file)
        print("Loaded {} images.".format(len(self.images_paths)))
    
    def init_dataset(self, root_dir):
        """ Load content from all classes and initialise each samples
        """
        if not root_dir.endswith('/'):
            root_dir = root_dir + '/'
        self.init_directory(root_dir)
    
    @staticmethod
    def get_image(path):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    @staticmethod
    def normalize(mat):
        return mat.astype(np.float32) / 255.0
    
    def iterate_set(self, input_size):
        in_h, in_w, in_c = input_size
        inputs = np.empty((1, in_h, in_w, in_c), dtype=np.float32)

        for path in self.images_paths:
            image = self.get_image(path)
            image = cv2.resize(image, (in_h, in_w), interpolation=cv2.INTER_AREA)
            inputs[0, :, :, 0] = self.normalize(image)
            yield inputs