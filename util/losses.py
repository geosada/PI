import tensorflow as tf
import numpy as np
import sys


eps = 1e-8

class LossFunctions(object):

    def __init__(self, layers, dataset, encoder):

        self.ls = layers
        self.d  = dataset
        self.encoder = encoder
        self.reconst_pixel_log_stdv = tf.get_variable("reconst_pixel_log_stdv", initializer=tf.constant(0.0))

    def get_loss_pyx(self, logit, y):

        loss = self._ce(logit, y)
        accur = self._accuracy(logit, y)
        return loss, accur

    def get_loss_pi(self, x, logit_real, is_train):
        logit_real = tf.stop_gradient(logit_real)
        logit_virtual = self.encoder(x, is_train=is_train, do_update_bn=False)
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(logit_real, logit_virtual))) + eps)
        return logit_real, logit_virtual, loss


    """ https://github.com/takerum/vat_tf/blob/master/layers.py """
    def _ce(self, logit, y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))

    def _accuracy(self, logit, y):
        pred = tf.argmax(logit, 1)
        true = tf.argmax(y, 1)
        return tf.reduce_mean(tf.to_float(tf.equal(pred, true)))
