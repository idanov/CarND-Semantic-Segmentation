import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def custom_init(shape, dtype=tf.float32, partition_info=None, seed=0):
    """
    Custom tensor initializer for the conv2d and conv2d_transpose layers.
    :param shape: Shape of the tensor
    :param dtype: Data type of the tensor
    :param partition_info: Partition info
    :param seed: The seed of the random generator
    :return: A tensor of the given shape with random numbers
    """
    return tf.random_normal(shape, dtype=dtype, seed=seed)


def fcn_1x1(layer_out, num_classes):
    """
    A 1x1 convolution layer.
    :param layer_out: The output of the previous layer
    :param num_classes: Number of output classes
    :return: A 1x1 convolution layer
    """
    return tf.layers.conv2d(layer_out, num_classes, kernel_size=1, strides=1, kernel_initializer=custom_init)


def fcn_upsample(layer_out, num_classes, scale=2):
    """
    An upsampling convolution layer.
    :param layer_out: The output of the previous layer
    :param num_classes: Number of output classes
    :param scale: Sample factor (2x, 4x, etc)
    :return: An upsample layer
    """
    return tf.layers.conv2d_transpose(layer_out, num_classes, kernel_size=scale, strides=scale, kernel_initializer=custom_init)


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

    fcn7_2x = fcn_upsample(fcn7_1x1, num_classes, 2)
    fcn47_sum = tf.add(fcn4_1x1, fcn7_2x)
    fcn47_2x = fcn_upsample(fcn47_sum, num_classes, 2)

    return tf.add(fcn3_1x1, fcn47_2x, name='last_layer')
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
    logits = tf.nn.sigmoid(nn_last_layer, name='logits')
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=logits))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return tf.reshape(logits, [-1, num_classes]), train_op, cross_entropy_loss
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
    for i in range(epochs):
        print("Epoch {}...".format(i))
        for batch_x, batch_y in get_batches_fn(batch_size):
            feed = {
                learning_rate: 1e-4,
                keep_prob: 0.5,
                input_image: batch_x,
                correct_label: batch_y
            }
            result, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed)
            print("Batch loss: {}".format(loss))
tests.test_train_nn(train_nn)


def run():
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

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function

        # TODO: Train NN using the train_nn function

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
