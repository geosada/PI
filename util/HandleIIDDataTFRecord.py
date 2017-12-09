import sys, os, time

class HandleIIDDataTFRecord(object):

    def __init__(self, dataset, batch_size, is_debug=False):

        self.dataset    = dataset
        self.batch_size = batch_size 
        self.is_debug   = is_debug

        if self.dataset == 'SVHN':
            from svhn import N_LABELED
            n_train, n_test, n_labeled = 73257, 26032, N_LABELED
            _h, _w, _c = 32,32,3
            _img_size = _h*_w*_c
            _l = 10
            _is_3d = True
        else:
            sys.exit('[ERROR] not implemented yet')

        self.h = _h
        self.w = _w
        self.c = _c
        self.l = _l
        self.is_3d     = _is_3d 
        self.img_size  = _img_size
        self.n_train   = n_train
        self.n_test    = n_test
        self.n_labeled = n_labeled
        self.n_batches_train = int(n_train/batch_size)
        self.n_batches_test  = int(n_test/batch_size)

    ########################################
    """             inputs              """
    ########################################
    def get_tfrecords(self):

        """
        xtrain: all records
        *_l   : partial records
        """
        if self.dataset =='SVHN':
            from svhn import inputs, unlabeled_inputs
            xtrain_l, ytrain_l = inputs(batch_size=self.batch_size, train=True,  validation=False, shuffle=True)
            xtrain             = unlabeled_inputs(batch_size=self.batch_size,    validation=False, shuffle=True)
            xtest , ytest      = inputs(batch_size=self.batch_size, train=False, validation=False, shuffle=True)
        else:
            sys.exit('[ERROR] not implemented yet')
        return (xtrain_l, ytrain_l), xtrain, (xtest , ytest)


if __name__ == '__main__':

    BATCH_SIZE = 20

    d = HandleIIDDataTFRecord( 'SVHN', BATCH_SIZE, is_debug=True)
    print(d.get_tfrecords())

    sys.exit('saigo')
