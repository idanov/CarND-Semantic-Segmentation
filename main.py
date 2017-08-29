#!/usr/bin/env python

import argparse
import os.path
import warnings
from distutils.version import LooseVersion

import tensorflow as tf

import helper
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def fcn_1x1(layer_out, num_classes):
    """
    A 1x1 convolution layer.
    :param layer_out: The output of the previous layer
    :param num_classes: Number of output classes
    :return: A 1x1 convolution layer
    """
    return tf.layers.conv2d(layer_out, num_classes, kernel_size=(1, 1), strides=(1, 1),
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))


def fcn_upsample(layer_out, num_classes, k_size, stride):
    """
    An upsampling convolution layer.
    :param layer_out: The output of the previous layer
    :param num_classes: Number of output classes
    :param k_size: Kernel size
    :param stride: Stride size
    :return: An upsample layer
    """
    return tf.layers.conv2d_transpose(layer_out, num_classes,
                                      kernel_size=(k_size, k_size), strides=(stride, stride), padding='same',
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # DONE: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # DONE: Implement function
    fcn3_1x1 = fcn_1x1(vgg_layer3_out, num_classes)
    fcn4_1x1 = fcn_1x1(vgg_layer4_out, num_classes)
    fcn7_1x1 = fcn_1x1(vgg_layer7_out, num_classes)

    fcn7_2x = fcn_upsample(fcn7_1x1, num_classes, 4, 2)
    fcn47_sum = tf.add(fcn4_1x1, fcn7_2x)
    fcn47_2x = fcn_upsample(fcn47_sum, num_classes, 4, 2)

    fcn347_sum = tf.add(fcn3_1x1, fcn47_2x)
    fcn347_2x = fcn_upsample(fcn347_sum, num_classes, 16, 8)
    return fcn347_2x
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # DONE: Implement function
    logits = tf.nn.sigmoid(tf.reshape(nn_last_layer, (-1, num_classes)), name='logits')
    labels = tf.nn.sigmoid(tf.reshape(correct_label, (-1, num_classes)), name='labels')
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess: tf.Session, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # DONE: Implement function
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        print("Starting epoch {}...".format(epoch))
        batch_idx = 0
        loss = 1.
        for batch_x, batch_y in get_batches_fn(batch_size):
            feed = {
                learning_rate: 1e-4,
                keep_prob: 1.,
                input_image: batch_x,
                correct_label: batch_y
            }
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed)
            batch_idx = batch_idx + 1
        print(" => Epoch: {} | loss: {}".format(epoch, loss))
tests.test_train_nn(train_nn)


def run(num_epochs=30, batch_size=4):
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    print("=====================")
    print("Start training with {} epochs and batch size of {}...".format(num_epochs, batch_size))
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)

        # DONE: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(last_layer, correct_label, learning_rate, num_classes)

        # DONE: Train NN using the train_nn function
        train_nn(sess, num_epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss,
                 input_image, correct_label, keep_prob, learning_rate)

        # DONE: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a semantic segmentation model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs.')
    parser.add_argument('--batch', type=int, default=4, help='Batch size.')
    args = parser.parse_args()
    run(args.epochs, args.batch)
