""" Design of the CNN Model  """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorlib.modelbase import ModelBuilderBase
from tensorlib.layers import Conv2D, MaxPool2D, SeqLayer

class SnapRapidCNNModelBuilder(ModelBuilderBase):
    """ Class Helper to build the CNN Model
    """

    def __init__(self, input_shape, n_classes=1):
        """
        input_shape: a 3 dimensional tuple containing the (height, width, channels) of the input data
        """
        super(SnapRapidCNNModelBuilder, self).__init__(ns="snap_cnn")
        self.height, self.width, self.channels = input_shape
        self.n_classes = n_classes
        self.h0_filters = 32
        self.h1_filters = 64
        self.h2_filters = 128
        self.h3_filters = 256

    def get_model(self):
        with tf.name_scope(self.ns):
            layers = [ 
                Conv2D(self.channels, self.h0_filters, f_kernel=(3, 3), activation_fn=tf.nn.relu, ns="Conv1_1"),
                Conv2D(self.h0_filters, self.h0_filters, f_kernel=(3, 3), activation_fn=tf.nn.relu, ns="Conv1_2"), 
                MaxPool2D(kernel=(2, 2), ns="Pool1"),

                Conv2D(self.h0_filters, self.h1_filters, f_kernel=(3, 3), activation_fn=tf.nn.relu, ns="Conv2_1"),
                Conv2D(self.h1_filters, self.h1_filters, f_kernel=(3, 3), activation_fn=tf.nn.relu, ns="Conv2_2"),
                MaxPool2D(kernel=(2, 2), ns="Pool2"),
                
                Conv2D(self.h1_filters, self.h2_filters, f_kernel=(3, 3), activation_fn=tf.nn.relu, ns="Conv3_1"),
                Conv2D(self.h2_filters, self.h2_filters, f_kernel=(3, 3), activation_fn=tf.nn.relu, ns="Conv3_2"),
                
                Conv2D(self.h2_filters, self.h3_filters, f_kernel=(1, 1), activation_fn=tf.nn.relu, ns="Conv5_1"),
                Conv2D(self.h3_filters, self.n_classes, f_kernel=(1, 1) , ns="Conv_Final"),
            ]
            m = SeqLayer(layers, ns=self.ns)
        return m
    
    def get_output_shape(self):
        """
        Work out calculation for the size of the output mask based on input size
        """
        ow = int(self.width / 2 / 2)
        oh = int(self.height / 2 / 2)
        return (oh, ow, self.n_classes)

# Make sure model compiles and run OK on dummy data.
import numpy as np

if __name__ == '__main__':
    i_shape = (180, 180, 1)
    XX = np.ones(shape=[2, 180, 180, 1], dtype=np.float32)
    X = tf.placeholder(tf.float32, shape=[2, 180, 180, 1], name="input")
    model_builder = SnapRapidCNNModelBuilder(i_shape, n_classes=1)
    model = model_builder.get_model()
    output = model(X)
    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)        
        print(sess.run([output], feed_dict={X: XX}))
