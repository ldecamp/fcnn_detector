""" Create a class to hold the Dataset generation  """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from operator import itemgetter 

import os
import cv2
import math
import random
import numpy as np

DEBUG = False

class TrainDataSet(object):

    def __init__(self, root_dir, input_size, target_size, train_split=0.7, batch_size=15, receptive_field_mask=25):
        """
        Creates an instance of Dataset holder.
        For now store all training and test data into memory

        root_dir: Specifies the directory to load the data from
        input_size: 3d tuple - contains the (height, width, channels) of the input image
        target_size: 3d tuple - contains the (height, width, nclasses) of the output mask
        train_split: Percentage of data kept for training
        batch_size: Set the minibatch size
        receptive_field_mask: Number of pixels extracted around the detection center to create the mask
        """
        self.target_recept_field = receptive_field_mask
        self.train_split = train_split
        self.batch_size = batch_size
        # Structure of data is tuple (image,label)
        self.images_train = []
        self.images_test = []
        self.in_h, self.in_w, self.in_c = input_size
        self.out_h, self.out_w, self.out_c = target_size
        
        self.init_dataset(root_dir)
    
    def init_directory(self, images_dir):
        """ Read all image content from input directory
        """
        content = []
        for dir_name, _, file_list in os.walk(images_dir):
            for file_name in file_list:
                image_file = os.path.join(dir_name, file_name)
                image = self.get_image(image_file)

                mask = np.zeros(shape=image.shape, dtype=np.uint8)
                im_label = 0
                lbl_center = None
                if file_name.find('_') != -1:
                    im_label = 1
                    # if contains label center, then positive so create mask
                    x, y = file_name.split('_')[1:]
                    x, y = int(x), int(y.split('.')[0])
                    lbl_center = (x, y)
                    rc = self.target_recept_field
                    mask[(y-rc):(y+rc), (x-rc):(x+rc)] = 255*np.ones(shape=(2*rc,2*rc), dtype=np.uint8)
                
                # Resize input and targets to fit CNN dimensions
                image = cv2.resize(image, (self.in_h, self.in_w), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, (self.out_h, self.out_w), interpolation=cv2.INTER_AREA)
                mask[np.where(mask != 0)] = 255 # set all pixels within interpolation to be in mask
                content.append((image, mask, im_label, lbl_center))

        return content
    
    def init_dataset(self, root_dir):
        """ Load content from all classes and initialise each samples
        """
        if not root_dir.endswith('/'):
            root_dir = root_dir + '/'
        neg_path = os.path.join(root_dir, 'negative')
        pos_path = os.path.join(root_dir, 'positive')
        
        neg_images = self.init_directory(neg_path)
        pos_images = self.init_directory(pos_path)

        self.split_train_test(pos_images, neg_images)

    
    def split_train_test(self, ds_pos, ds_neg):
        """ Ensure that we split the data into a training and test set.
        """
        # Make sure we always generate the same train/test split.
        random.seed(123) # Fix seed
        # Ensure We balance neg/pos in train and test set.
        self.images_train, self.images_test = [], []

        n_pos, n_neg = len(ds_pos), len(ds_neg)
        n_pos_train =  int(math.floor(n_pos * self.train_split))
        n_neg_train = int(math.floor(n_neg * self.train_split))

        pos_idx, neg_idx = range(n_pos), range(n_neg)
        random.shuffle(pos_idx)
        random.shuffle(neg_idx)

        pos_train_idx, pos_test_idx = pos_idx[:n_pos_train], pos_idx[n_pos_train:]
        neg_train_idx, neg_test_idx = neg_idx[:n_neg_train], neg_idx[n_neg_train:]

        self.images_train = itemgetter(*pos_train_idx)(ds_pos) + itemgetter(*neg_train_idx)(ds_neg)
        self.images_test = itemgetter(*pos_test_idx)(ds_pos) + itemgetter(*neg_test_idx)(ds_neg)
        random.seed(None) # Remove fix seed
        # Print dataset stats
        print("Loaded {} images into the training set".format(len(self.images_train)))
        print("N pos Train: {}     ||  N neg Train: {}".format(n_pos_train, n_neg_train))
        print("Loaded {} images into the test set".format(len(self.images_test)))
        print("N pos Test: {}     ||  N neg Test: {}".format(n_pos-n_pos_train, n_neg-n_neg_train))
    
    @staticmethod
    def get_image(path):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    @staticmethod
    def normalize(mat):
        return mat.astype(np.float32) / 255.0
    
    def iterate_train_minibatch(self, val=False):
        train_set = self.images_train
        if val:
            indices = range(len(self.images_train))
            random.shuffle(indices)
            indices = indices[0:int(len(indices) / 5)]
            train_set = itemgetter(*indices)(self.images_train)
        return self.iterate_minibatch(train_set)
    
    def iterate_validation_minibatch(self):
        return self.iterate_train_minibatch(val=True)

    def iterate_test_minibatch(self):
        return self.iterate_minibatch(self.images_test)
    
    @staticmethod
    def revert_mask(mask):
        return np.abs(mask.astype(np.float32)-255)

    def iterate_minibatch(self, data):
        indices = range(len(data))
        random.shuffle(indices)

        for batch_idx in range(int(len(data) / self.batch_size)):
            inputs = np.empty((self.batch_size, self.in_h, self.in_w, self.in_c), dtype=np.float32)
            masks = np.empty((self.batch_size, self.out_h, self.out_w, self.out_c), dtype=np.float32)
            labels = np.empty((self.batch_size), dtype=np.int32)

            for im_idx in range(self.batch_size):
                sidx = indices[self.batch_size*batch_idx+im_idx]
                image, mask, label, _ = data[sidx]

                neg_mask = self.revert_mask(mask)

                if DEBUG:
                    self.show('image', image, waitKey=False, destroy=False)
                    self.show('mask', mask, waitKey=False, destroy=False)
                    self.show('revert', neg_mask)

                inputs[im_idx, :, :, 0] = self.normalize(image)
                masks[im_idx, :, :, 0] = self.normalize(neg_mask)
                masks[im_idx, :, :, 1] = self.normalize(mask)
                labels[im_idx] = label
            yield inputs, masks, labels

    def show(self, name, image, waitKey=True, destroy=True):
        cv2.imshow(name, image)
        cv2.moveWindow(name, 0, 0)
        if waitKey:
            cv2.waitKey()
        if destroy:
            cv2.destroyAllWindows()

# Test Dataset minibatch iteration code.
if __name__ == "__main__":
    ds = DataSet("./debug", 180, 45, batch_size=2)
    print("Iterate Training")
    for (dtrain, masks, labels) in ds.iterate_train_minibatch():
        print(labels)
