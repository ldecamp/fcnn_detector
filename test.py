# Copyright 2016 Laurent Decamp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Training script for the SnapRapid Logo detection challenge  """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, csv
import tensorflow as tf

from time import time
from options import FLAGS
from test_dataset import TestDataset
from detector import LogoDetector
from model import SnapRapidCNNModelBuilder

def test():
    # Loading dataset
    print("Loading dataset")
    model_builder = SnapRapidCNNModelBuilder((FLAGS.input_size, FLAGS.input_size, 1), n_classes=2)
    h, w, c = model_builder.get_output_shape()
    dataset = TestDataset(FLAGS.test_dir)
    detector = LogoDetector(min_area=FLAGS.min_area, iou_threshold=FLAGS.iou_thres)

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, (None, FLAGS.input_size, FLAGS.input_size, 1), name='x-input')
        y = tf.placeholder(tf.float32, (None, h, w, c), name='y-truth')
    
    # Build Model
    with tf.name_scope('model'):
        model = model_builder.get_model()
    
    # Create prediction function
    with tf.name_scope('output'):
        y = tf.arg_max(tf.nn.sigmoid(model(x)), 3)

    labels = []
    avg_t = 0.0
    with tf.Session() as sess:
        # Initialise all variables
        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        if os.path.exists(FLAGS.model_path):
            print("Restoring model from last snapshot")
            saver = tf.train.import_meta_graph(FLAGS.model_path + ".meta")
            saver.restore(sess, FLAGS.model_path)
        else:
            print("Could not find model at: {}. Training from scratch".format(FLAGS.model_path))
            exit()
        
        for image in dataset.iterate_set((FLAGS.input_size, FLAGS.input_size, 1)):
            # For each image predict label
            t_start = time()
            pred = sess.run([y], feed_dict={x: image})[0]
            predicted = detector.detect(pred[0, :, :])
            t_end = time()
            avg_t = (avg_t + float(t_end-t_start)) / 2.0
            if len(predicted) > 0:
                labels.append(1)
            else:
                labels.append(0)
    
    print("Average classification time in ms: {}".format(avg_t))

    # Write predictions into output csv file.
    if not os.path.exists(FLAGS.pred_path):
        os.makedirs(os.path.dirname(FLAGS.pred_path))
    with open(FLAGS.pred_path, 'wb') as csvfile:
        pred_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        pred_writer.writerow(["prediction"])
        for prediction in labels:
            pred_writer.writerow([prediction])


def main(_):
    test()

if __name__ == "__main__":
    tf.app.run()