""" Create a Set of Helper layers Enabling for easy network design + Full Logging """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class Layer(object):
    def __init__(self, ns):
        self.ns = ns
    
    def get_variables(self):
        return []

    def attach_summaries(self, var, name):
        """ Attach Ouput debug info to tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)

class Conv2D(Layer):

    def __init__(self, in_C, n_filters, f_kernel=(2, 2),
                 stride=(1, 1), activation_fn=tf.identity, padding='SAME', ns="Conv"):
        """
        Defines a 2D Convolution Layer
        in_C: Number of incomming channels
        n_filters: Number of Convolution Filters
        f_kernel: Size of convolution kernel (default: (2,2))
        stride: Stride applied to Convolution
        padding: Type of Padding applied to convolution: SAME | VALID
        ns: Namespace in graph
        """
        super(Conv2D, self).__init__(ns)
        self.in_C, self.out_C = in_C, n_filters
        # Validate Convolution Kernel
        if not isinstance(f_kernel, list) and len(f_kernel) is not 2:
            raise Exception("Invalid Convolution kernel")
        self.kernel = f_kernel
        if not isinstance(stride, list) and len(stride) is not 2:
            raise Exception("Invalid Stride definition")
        self.stride = [1] + list(stride) + [1]
        self.padding = padding
        self.activation_fn = activation_fn

        input_size = self.kernel[0] * self.kernel[1] * self.in_C
        
        w_shape = list(self.kernel + (self.in_C, self.out_C))
        with tf.variable_scope(self.ns):
            w_initializer=tf.contrib.layers.xavier_initializer_conv2d()
            self.W = tf.get_variable("W", shape=w_shape, initializer=w_initializer, trainable=True)
            self.attach_summaries(self.W, self.ns + '/weights')

            b_initializer=tf.constant(0.1, shape=[self.out_C])
            self.b = tf.get_variable("b", initializer=b_initializer, trainable=True)
            self.attach_summaries(self.b, self.ns + '/biases')

    def __call__(self, X):
        with tf.variable_scope(self.ns):
            pre_act = tf.nn.conv2d(X, self.W, strides=self.stride, padding=self.padding) + self.b
            return self.activation_fn(pre_act)
    
    def get_variables(self):
        return [self.W, self.b]



class MaxPool2D(Layer):
    def __init__(self, kernel=(2, 2), ns="MaxPool"):
        super(MaxPool2D, self).__init__(ns)
        if not isinstance(kernel, list) and len(kernel) is not 2:
            raise Exception("Invalid Pooling kernel size")
        self.k_size = [1] + list(kernel) + [1]
        self.strides = self.k_size

    def __call__(self, X):
        with tf.variable_scope(self.ns):
            return tf.nn.max_pool(X, self.k_size, self.strides, padding='SAME')

class LambdaLayer(Layer):
    def __init__(self, f, ns="lambda"):
        super(LambdaLayer, self).__init__(ns)
        self.f = f

    def __call__(self, x):
        act = self.f(x)
        with tf.name_scope(self.ns):
            tf.histogram_summary(self.ns + '/activations', act)
        return act

class ReLULayer(LambdaLayer):
    def __init__(self, ns="ReLU"):
        """ Syntaxic Sugar for ReLU Lambda Layer
        """
        super(ReLULayer, self).__init__(tf.nn.relu, ns)


class SeqLayer(Layer):
    def __init__(self, layers, ns='seq_layer'):
        """ 
        Initialise the Sequential Layer
        layers: Array containing all layers within the network
        """
        super(SeqLayer, self).__init__(ns)
        self.layers = layers

    def __call__(self, x):
        """
        X: tensor containing the input data that will be fed to the subsequent layers
        """
        for l in self.layers:
            x = l(x)
        return x
    
    def get_variables(self):
        """
        Return all variables present in all the network layers
        """
        variables = []
        for l in self.layers:
            variables.extend(l.get_variables())
        return variables