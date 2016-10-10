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

import os
import numpy as np
import tensorflow as tf

from time import time
from options import FLAGS
from train_dataset import TrainDataSet
from detector import LogoDetector
from model import SnapRapidCNNModelBuilder

def eval_predictions(detector, predictions, gtruth, labels):
    tp, fp, tn, fn = 0, 0, 0, 0
    # For each image in the validation/test set. Compare prediction with ground truth 
    # according to logic in detector and return a summary analysis
    for i in range(0, predictions.shape[0]):
        label = labels[i]
        pred = predictions[i, :, :]
        ltruth = np.argmax(gtruth[i, :, :], 2) #extract single mask from targets
        predicted = detector.detect(pred)

        if label == 0 and len(predicted) > 0:
            fp += len(predicted)
        elif label == 0 and len(predicted) == 0:
            tn += 1
        elif label == 1 and len(predicted) == 0:
            fn += 1
        else:
            truth = detector.detect(ltruth)
            # for each detection check if the detection found lies within boundaries of a target
            for d in predicted:
                found = False
                for g in truth:
                    found = detector.eval_detections(d, g)
                    if found:
                        tp += 1 
                        break
                if not found:
                    fp += 1
    return tp, fp, tn, fn

def eval_dataset(params, epoch, iter_func, name):
    session, x, y = params['session'], params['x'], params['y']
    f_eval, f_entropy = params['f_eval'], params['f_entropy']
    merged, swriter = params['holder'], params['writer']
    detector = params['detector']

    # Helper to either run assessment on validation or test set
    ds_err, ds_batches = 0, 0
    ds_tp, ds_fp, ds_tn, ds_fn = 0, 0, 0, 0

    # Call iterative function to evaluate data
    for (dval, targets, labels) in iter_func():
        preds, err, summary = session.run([f_eval, f_entropy, merged], feed_dict={x: dval, y: targets})
        ds_err += err
        ds_batches += 1
        tp, fp, tn, fn = eval_predictions(detector, preds, targets, labels)
        ds_fp += fp
        ds_tp += tp
        ds_tn += tn
        ds_fn += fn
        if swriter != None:
            swriter.add_summary(summary, epoch)
    
    print("  TP:{}, FP:{}, TN:{}, FN:{}".format(ds_tp, ds_fp, ds_tn, ds_fn))
    ds_prec = 0 if ds_tp == 0 else ds_tp / (ds_tp+ds_fp)
    ds_rec = 0 if ds_tp == 0 else ds_tp / (ds_tp+ds_fn)
    ds_acc = (ds_tp+ds_tn)/(ds_tp+ds_tn+ds_fn+ds_fp)
    ds_f1 = 0 if (ds_prec+ds_rec) == 0 else  2* (ds_prec*ds_rec) / (ds_prec+ds_rec)
    # Print stats
    print("  {} loss:\t\t{:.4f} %".format(name, ds_err / ds_batches))
    print("  {} accuracy:\t\t{:.4f} %".format(name, 100 * ds_acc))
    print("  {} f1:\t\t{:.4f} %".format(name, 100 * ds_f1))

    return (ds_err / ds_batches), ds_f1

def train():
    # Loading dataset
    print("Loading dataset")
    # random.seed(123) 
    model_builder = SnapRapidCNNModelBuilder((FLAGS.input_size, FLAGS.input_size, 1), n_classes=2)
    h, w, c = model_builder.get_output_shape()
    dataset = TrainDataSet(FLAGS.data_dir, (FLAGS.input_size, FLAGS.input_size, 1), (h, w, c), batch_size=FLAGS.batch_size)
    detector = LogoDetector(min_area=FLAGS.min_area, iou_threshold=FLAGS.iou_thres)

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, (None, FLAGS.input_size, FLAGS.input_size, 1), name='x-input')
        y = tf.placeholder(tf.float32, (None, h, w, c), name='y-truth')

    with tf.name_scope('model'):
        model = model_builder.get_model()
        all_vars = model.get_variables()
        model_saver = tf.train.Saver(all_vars)

    with tf.name_scope('output'):
        y_ = model(x)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_, y, name="cross_entropy"))
        tf.scalar_summary('cross entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.MomentumOptimizer(FLAGS.learning_rate, 0.9, use_nesterov=True)
        train_step = train_step.minimize(cross_entropy)
    
    with tf.name_scope('test'):
        y_eval = tf.arg_max(tf.nn.sigmoid(model(x)), 3)

    # Merge all the summaries and write them out to  output
    merged = tf.merge_all_summaries()

    best_val_loss = 100
    best_val_f1 = 0.0
    with tf.Session() as sess:
        # Initialise all variables
        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        if FLAGS.load_model:
            if os.path.exists(FLAGS.model_path):
                print("Restoring model from last snapshot")
                saver = tf.train.import_meta_graph(FLAGS.model_path + ".meta")
                saver.restore(sess, FLAGS.model_path)
            else:
                print("Could not find model at: {}. Training from scratch".format(FLAGS.model_path))

        train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
        valid_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/valid')
        test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
        
        # train until reach max epochs 
        for epoch in range(FLAGS.max_epochs):
            print("Training Epoch {} \n".format(epoch))
            
            train_err = 0.0
            train_batches = 0
            start_time = time()
            # Go Over all the training data in each epoch
            for (dtrain, targets, _) in dataset.iterate_train_minibatch():
                summary, err, _ = sess.run([merged, cross_entropy, train_step], 
                                feed_dict = {x: dtrain, y: targets})
                train_err += err
                train_batches += 1
                train_writer.add_summary(summary, epoch)
            
            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, FLAGS.max_epochs, time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

            params = {
                'session': sess,
                'f_eval': y_eval,
                'x': x,
                'y': y,
                'f_entropy': cross_entropy,
                'writer': valid_writer,
                'holder': merged,
                'detector': detector
            }
            val_loss, val_f1 = eval_dataset(params, epoch, dataset.iterate_validation_minibatch, 'validation')
            # If Loss lower then save model
            if val_loss < best_val_loss:
                print("New best validation loss {}".format(val_loss))
                best_val_loss = val_loss

            if val_f1 > best_val_f1:
                print("New best validation f1 {}. Saving Model".format(val_loss))
                best_val_f1 = val_f1
                model_saver.save(sess, FLAGS.model_path)
                
            # Every 10 iterations test
            if epoch % 10 == 0:
                params['writer'] = test_writer
                eval_dataset(params, epoch, dataset.iterate_test_minibatch, 'test')
        
        train_writer.close()
        test_writer.close()

def main(_):
    # Need to Find out how to store multiple runs in output
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()

if __name__ == "__main__":
    tf.app.run()