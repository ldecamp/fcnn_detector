""" Sets up all options related to the Training/Test Scripts
"""
import tensorflow as tf

flags = tf.app.flags

# Shared Options
flags.DEFINE_string('model_path', './models/snap_cnn', 'Directory where model weights are stored')

flags.DEFINE_integer('input_size', 180, 'size of input image')
flags.DEFINE_integer('min_area', 9, 'min area below which detection are ignored. i.e. box needs to be covering min 9 pixels') 
flags.DEFINE_integer('iou_thres', 0.25, 'overlapping area ratio for the detection to be considered successful')

# Train Options
flags.DEFINE_integer('max_epochs', 5000, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 8, 'Set the size of training mini-batch')
flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')

flags.DEFINE_string('data_dir', './datasets/full', 'Directory where data is stored')
flags.DEFINE_boolean('load_model', True, 'Whether to restore the previously saved model')
flags.DEFINE_string('summaries_dir', './tensorlogs', 'Directory where log outputs are stored')

# Test Options
flags.DEFINE_string('test_dir', './datasets/full', 'Directory where data is stored')
flags.DEFINE_string('pred_path', './output/prediction.csv', 'File where the final prediction will be stored')

FLAGS = flags.FLAGS