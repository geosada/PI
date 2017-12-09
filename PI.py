import tensorflow as tf
import numpy as np
from layers import Layers
from losses import LossFunctions

class PI(object):

    def __init__(self, d, lr, lambda_pi_usl, use_pi):

        """ flags for each regularizor """
        self.use_pi   = use_pi

        """ data and external toolkits """
        self.d  = d  # dataset manager
        self.ls = Layers()
        self.lf = LossFunctions(self.ls, d, self.encoder)

        """ placeholders defined outside"""
        self.lr  = lr
        self.lambda_pi_usl = lambda_pi_usl	

    def encoder(self, x, is_train=True, do_update_bn=True):

        """ https://arxiv.org/pdf/1610.02242.pdf """

        if is_train:
            h = self.distort(x)
            h = self.ls.get_corrupted(x, 0.15)
        else:
            h = x

        scope = '1'
        h = self.ls.conv2d(scope+'_1', h, 128, activation=self.ls.lrelu)
        h = self.ls.conv2d(scope+'_2', h, 128, activation=self.ls.lrelu)
        h = self.ls.conv2d(scope+'_3', h, 128, activation=self.ls.lrelu)
        h = self.ls.max_pool(h)
        if is_train: h = tf.nn.dropout(h, 0.5)

        scope = '2'
        h = self.ls.conv2d(scope+'_1', h, 256, activation=self.ls.lrelu)
        h = self.ls.conv2d(scope+'_2', h, 256, activation=self.ls.lrelu)
        h = self.ls.conv2d(scope+'_3', h, 256, activation=self.ls.lrelu)
        h = self.ls.max_pool(h)
        if is_train: h = tf.nn.dropout(h, 0.5)

        scope = '3'
        h = self.ls.conv2d(scope+'_1', h, 512, activation=self.ls.lrelu)
        h = self.ls.conv2d(scope+'_2', h, 256, activation=self.ls.lrelu, filter_size=(1,1))
        h = self.ls.conv2d(scope+'_3', h, 128, activation=self.ls.lrelu, filter_size=(1,1))
        h = tf.reduce_mean(h, reduction_indices=[1, 2])  # Global average pooling
        h = self.ls.dense(scope, h, self.d.l)

        return h

    def build_graph_train(self, x_l, y_l, x, is_supervised=True):

        o = dict()  # output
        loss = 0

        logit = self.encoder(x)

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            logit_l = self.encoder(x_l, is_train=True, do_update_bn=False)  # for pyx and vat loss computation

        """ Classification Loss """
        o['Ly'], o['accur'] = self.lf.get_loss_pyx(logit_l, y_l)
        loss += o['Ly']

        """ PI Model Loss """
        if self.use_pi:
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                _,_,o['Lp'] = self.lf.get_loss_pi(x, logit, is_train=True)
                loss += self.lambda_pi_usl * o['Lp']
        else:
            o['Lp'] = tf.constant(0)

        """ set losses """
        o['loss'] = loss
        self.o_train = o

        """ set optimizer """
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5)
        #self.op = optimizer.minimize(loss)
        grads = optimizer.compute_gradients(loss)
        for i,(g,v) in enumerate(grads):
            if g is not None:
                #g = tf.Print(g, [g], "g %s = "%(v))
                grads[i] = (tf.clip_by_norm(g,5),v) # clip gradients
            else:
                print('g is None:', v)
                v = tf.Print(v, [v], "v = ", summarize=10000)
        self.op = optimizer.apply_gradients(grads) # return train_op


    def build_graph_test(self, x_l, y_l ):

        o = dict()  # output
        loss = 0

        logit_l = self.encoder(x_l, is_train=False, do_update_bn=False)  # for pyx and vat loss computation

        """ classification loss """
        o['Ly'], o['accur'] = self.lf.get_loss_pyx(logit_l, y_l)
        loss += o['Ly']

        """ set losses """
        o['loss'] = loss
        self.o_test = o

    def distort(self, x):
    
        _d = self.d

        def _distort(a_image):
            """
            bounding_boxes: A Tensor of type float32.
                3-D with shape [batch, N, 4] describing the N bounding boxes associated with the image. 
            Bounding boxes are supplied and returned as [y_min, x_min, y_max, x_max]
            """
            # shape: [1, 1, 4]
            bounding_boxes = tf.constant([[[1/10, 1/10, 9/10, 9/10]]], dtype=tf.float32)
                                                                                                         
            begin, size, _ = tf.image.sample_distorted_bounding_box(
                                (_d.h, _d.w, _d.c), bounding_boxes,
                                min_object_covered=(8.5/10.0),
                                aspect_ratio_range=[7.0/10.0, 10.0/7.0])

            a_image = tf.slice(a_image, begin, size)
            """ for the purpose of distorting not use tf.image.resize_image_with_crop_or_pad under """
            a_image = tf.image.resize_images(a_image, [_d.h, _d.w])
            """ due to the size of channel returned from tf.image.resize_images is not being given,
                specify it manually. """
            a_image = tf.reshape(a_image, [_d.h, _d.w, _d.c])
            return a_image

        """ process batch times in parallel """
        return tf.map_fn( _distort, x)
