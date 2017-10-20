import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sigmoid = tf.nn.sigmoid
relu = tf.nn.relu
elu = tf.nn.elu

def bias(size, zero=False):
    if zero:
        return tf.Variable(tf.zeros([size], dtype=tf.float32))
    else:
        return tf.Variable(tf.random_normal([size], stddev=0.1))
    
def conv(tensor, n_out, ksize=3, stride=1):
    in_shape = tensor.shape.as_list()
    n_in = in_shape[-1]
    kernels = tf.Variable(tf.random_normal([ksize, ksize, n_in, n_out],  stddev=0.1))
    loss = tf.nn.l2_loss(kernels)
    tf.add_to_collection('l2_losses', loss)
    return tf.nn.conv2d(tensor,kernels, strides=[1, stride, stride, 1], padding='SAME') + bias(n_out)


def dense(tensor, n_out):
    in_shape = tensor.shape.as_list()
    n_in = in_shape[-1]
    weights = tf.Variable(tf.random_normal(shape=(n_in, n_out), mean=0, stddev=0.1))
    loss = tf.nn.l2_loss(weights)
    tf.add_to_collection('l2_losses', loss)
    return tf.matmul(tensor, weights) + bias(n_out)

def max_pool(tensor, ksize=3, stride=1):
    return tf.nn.max_pool(tensor, ksize=[1,ksize,ksize,1], strides=[1,stride,stride,1], padding='SAME')

def flatten(tensor):
    dim = tensor.shape.as_list()
    size = np.product(dim[1:])
    print(size)
    flat = tf.reshape(tensor, [-1, size])
    return flat