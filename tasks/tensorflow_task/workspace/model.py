import tensorflow as tf


def inference(images):
    """Build the model

    Args:
       images: Images placeholder, from inputs()

    Returns:
       logits: Output tensor with the computed logits.
    """
    # conv1

    W_conv1 = __weight_variable([5, 5, 1, 32])
    b_conv1 = __bias_variable([32])

    x_image = tf.reshape(images, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(__conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = __max_pool_2x2(h_conv1)

    # conv2

    W_conv2 = __weight_variable([5, 5, 32, 64])
    b_conv2 = __bias_variable([64])

    h_conv2 = tf.nn.relu(__conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = __max_pool_2x2(h_conv2)

    # fc

    W_fc1 = __weight_variable([7 * 7 * 64, 1024])
    b_fc1 = __bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = __weight_variable([1024, 10])
    b_fc2 = __bias_variable([10])

    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return logits, keep_prob


def __weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def __bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def __conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def __max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.

    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size]

    Returns:
        loss: Loss tensor of type float.
    """
    y = tf.nn.softmax(logits) 
    return -tf.reduce_sum(labels*tf.log(y))


def training(loss, learning_rate):
    """Sets up the training Ops.

    Args:
        loss: Loss tensor, from loss()
        learning_rate: The learning rate to use for gradient descent.

    Returns:
        train_op: The Op for training.
    """
    optimizer = tf.train.AdamOptimizer(learning_rate)
    return optimizer.minimize(loss)
