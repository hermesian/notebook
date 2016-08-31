import tensorflow as tf

import model
from tensorflow.examples.tutorials.mnist import input_data

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 100, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 50, 'Batch size, '
                     'Must divide evenly into the dataset sizes.')


def run_training():
    # Get the mnist data set
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder("float")

    # Build a Graph that computes predictions from the inference model.
    logits = model.inference(x, keep_prob)

    # Add to the Graph the Ops for loss calculation.
    cross_entropy = model.loss(logits, y_)

    # Add to the Graph the Ops that calculate and apply gradients
    train_step = model.training(cross_entropy, FLAGS.learning_rate)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create a session for running Ops on the Graph
    sess = tf.InteractiveSession()

    # Run the Op to initialize the variables.
    init = tf.initialize_all_variables()
    sess.run(init)

    for i in range(FLAGS.max_steps):

        batch = mnist.train.next_batch(FLAGS.batch_size)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    # following line occur "Resource exhausted: OOM when allocating tensor
    # with shape[10000,28,28,32]"
    print("test accuracy %g" % accuracy.eval(
        feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}
    ))


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
