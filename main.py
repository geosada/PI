import tensorflow as tf
import numpy as np
import sys, os, time

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/util')
from HandleIIDDataTFRecord import HandleIIDDataTFRecord
from PI  import PI

DO_TRAIN    = True
DO_TEST     = True
USE_PI      = True

tf.flags.DEFINE_string("dataset", "SVHN", "MNIST / CIFAR10 / SVHN / CharImages")
tf.flags.DEFINE_boolean("restore", False, "restore from the last check point")
tf.flags.DEFINE_string("dir_logs", "./out/", "")
FLAGS = tf.flags.FLAGS

if not DO_TRAIN and not FLAGS.restore:
    print('[WARN] FLAGS.restore is set to True compulsorily')
    FLAGS.restore = True

N_EPOCHS = 100

FILE_OF_CKPT  = os.path.join(FLAGS.dir_logs,"drawmodel.ckpt")

# learning rate decay
STARTER_LEARNING_RATE = 1e-3
DECAY_AFTER = 2
DECAY_INTERVAL = 2
DECAY_FACTOR = 0.97

def get_lambda_pi_usl(epoch):
    if USE_PI:
        import math
        def _rampup(epoch):
            """ https://github.com/smlaine2/tempens/blob/master/train.py """
            PI_RAMPUP_LENGTH = 80   # there seems to be no other option than 80, according to the paper.
            if epoch < PI_RAMPUP_LENGTH:
                p = 1.0 - (max(0.0, float(epoch)) / float(PI_RAMPUP_LENGTH))
                return math.exp(-p*p*5.0)
            else:
                return 1.0
            
        PI_W_MAX = 100
        _pi_m_n  = d.n_labeled / d.n_train
        return _rampup(epoch) * PI_W_MAX * _pi_m_n
    else:
        return 0.0


def test():
    accur = []
    for i in range(d.n_batches_test):
        r = sess.run(m.o_test)
        accur.append( r['accur'])
    return np.mean(accur, axis=0)

with tf.Graph().as_default() as g:

    ###########################################
    """             Load Data               """
    ###########################################
    BATCH_SIZE = 100
    d = HandleIIDDataTFRecord(FLAGS.dataset, BATCH_SIZE)
    (x_train, y_train), x, (x_test, y_test) = d.get_tfrecords()

    ###########################################
    """        Build Model Graphs           """
    ###########################################
    lr = tf.placeholder(tf.float32, shape=[], name="learning_rate")
    lambda_pi_usl = tf.placeholder(tf.float32, shape=(), name="lambda_pi_usl")

    with tf.variable_scope("watashinomodel") as scope:

        m = PI( d, lr, lambda_pi_usl, use_pi=USE_PI)

        print('... now building the graph for training.')
        m.build_graph_train(x_train,y_train,x) # the third one is a dummy for future
        scope.reuse_variables()
        if DO_TEST :
            print('... now building the graph for test.')
            m.build_graph_test(x_test,y_test)


    ###########################################
    """              Init                   """
    ###########################################
    init_op = tf.global_variables_initializer()
    for v in tf.all_variables(): print("[DEBUG] %s : %s" % (v.name,v.get_shape()))

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    sess  = tf.Session(config = config)

    _lr, ratio = STARTER_LEARNING_RATE, 1.0

    if FLAGS.restore:
        print("... restore from the last check point.")
        saver.restore(sess, FILE_OF_CKPT)
    else:
        sess.run(init_op)

    merged = tf.summary.merge_all()
    tf.get_default_graph().finalize()

    ###########################################
    """         Training Loop               """
    ###########################################
    if DO_TRAIN:
        print('... start training')
        tf.train.start_queue_runners(sess=sess)
        for epoch in range(1, N_EPOCHS+1):
    
            loss, accur = [],[]
            for i in range(d.n_batches_train):
        
                feed_dict = {lr:_lr, lambda_pi_usl:get_lambda_pi_usl(epoch)}
    
                """ do update """
                time_start = time.time()
                _, r, op, current_lr = sess.run([merged, m.o_train, m.op, m.lr], feed_dict=feed_dict)
                elapsed_time = time.time() - time_start

                loss.append(r['loss'])
                accur.append(r['accur'])
        
                if i % 100 == 0 and i != 0:

                    print(" iter:%2d, loss: %.5f, accr: %.5f, Ly: %s, Lp: %s, time:%.3f" % \
                             (i, np.mean(np.array(loss)), np.mean(np.array(accur)), r['Ly'], r['Lp'], elapsed_time ))
    
            """ test """
            if DO_TEST and epoch % 1 == 0:
                time_start = time.time()
                accur = test()
                elapsed_time = time.time() - time_start
                print("epoch:%d, accur: %s, time:%.3f" % (epoch, accur, elapsed_time ))
    
            """ save """
            if epoch % 1 == 0:
                print("Model saved in file: %s" % saver.save(sess,FILE_OF_CKPT))
    
    
            """ learning rate decay"""
            if (epoch % DECAY_INTERVAL == 0) and (epoch > DECAY_AFTER):
                ratio *= DECAY_FACTOR
                _lr = STARTER_LEARNING_RATE * ratio
                print('lr decaying is scheduled. epoch:%d, lr:%f <= %f' % ( epoch, _lr, current_lr))


sess.close()
