#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import sys

class Layers(object):

    def __init__(self):
        self.do_share = False

    def set_do_share(self, flag):
        self.do_share = flag

    def W( self, W_shape,  W_name='W', W_init=None):
        if W_init is None:
            W_initializer = tf.contrib.layers.xavier_initializer()
        else:
            W_initializer = tf.constant_initializer(W_init)

        return tf.get_variable(W_name, W_shape, initializer=W_initializer)

    def Wb( self, W_shape, b_shape, W_name='W', b_name='b', W_init=None, b_init=0.1):

        W = self.W(W_shape, W_name=W_name, W_init=None)
        b = tf.get_variable(b_name, b_shape, initializer=tf.constant_initializer(b_init))

        def _summaries(var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)
        _summaries(W)
        _summaries(b)

        return W, b


    def denseV2( self, scope, x, output_dim, activation=None):
        return tf.contrib.layers.fully_connected( x, output_dim, activation_fn=activation, reuse=self.do_share, scope=scope)

    def dense( self, scope, x, output_dim, activation=None):
        if len(x.get_shape()) == 2:   # 1d
            pass
        elif len(x.get_shape()) == 4: # cnn as NHWC
            #x = tf.reshape(x, [tf.shape(x)[0], -1]) # flatten
            x = tf.reshape(x, [x.get_shape().as_list()[0], -1]) # flatten
            #x = tf.reshape(x, [tf.cast(x.get_shape()[0], tf.int32), -1]) # flatten
        with tf.variable_scope(scope,reuse=self.do_share): W, b = self.Wb([x.get_shape()[1], output_dim], [output_dim])
        #with tf.variable_scope(scope,reuse=self.do_share): W, b = self.Wb([x.get_shape()[1], output_dim], [output_dim])
        o = tf.matmul(x, W) + b 
        return o if activation is None else activation(o)
    
    def lrelu(self, x, a=0.1):
        if a < 1e-16:
            return tf.nn.relu(x)
        else:
            return tf.maximum(x, a * x)

    def avg_pool(self, x, ksize=2, stride=2):
        return tf.nn.avg_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding='SAME')
    
    def max_pool(self, x, ksize=2, stride=2):
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding='SAME')

    def conv2d( self, scope, x, out_c, filter_size=(3,3), strides=(1,1,1,1), padding="SAME", activation=None):
        """
        x:       [BATCH_SIZE, in_height, in_width, in_channels]
        filter : [filter_height, filter_width, in_channels, out_channels]
        """
        filter = [filter_size[0], filter_size[1], int(x.get_shape()[3]), out_c]
        with tf.variable_scope(scope,reuse=self.do_share): W, b = self.Wb(filter, [out_c])
        o = tf.nn.conv2d(x, W, strides, padding) + b
        return o if activation is None else activation(o)

    ###########################################
    """             Softmax                 """
    ###########################################
    def softmax( self, scope, input, size):
        if input.get_shape()[1] != size:
            print("softmax w/ fc:", input.get_shape()[1], '->', size)
            return self.dense(scope, input, size, tf.nn.softmax)
        else:
            print("softmax w/o fc")
            return tf.nn.softmax(input)
    
    ###########################################
    """        Noise/Denose Function        """
    ###########################################
    def get_corrupted(self, x, noise_std=.10):
        return self.sampler( x, noise_std)

    def epsilon( self, _shape, _stddev=1.):
        return tf.truncated_normal(_shape, mean=0, stddev=_stddev)

    def sampler( self, mu, sigma):
        """
        mu,sigma : (BATCH_SIZE, z_size)
        """
        return mu + sigma*self.epsilon( tf.shape(mu) )
