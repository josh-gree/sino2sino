#
import tensorflow as tf
import numpy as np
import h5py
import os
import time

from glob import glob
from ops import max_pool, conv2d, batch_norm, lrelu, deconv2d


class recon2recon(object):

    def __init__(self, sess, model_name, train_path, test_path, val_path):

        # initialization

        self.sess = sess
        self.batch_size = 1

        now = time.localtime()
        self.model_name = model_name + '_{}_{}_{}_{}'.format(now.tm_mday,now.tm_mon,now.tm_hour,now.tm_min)

        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path


        # Batch norm layers (why not put in model)
        self.g_bn_e1a = batch_norm(name='g_bn_e1a')
        self.g_bn_e2a = batch_norm(name='g_bn_e2a')
        self.g_bn_e3a = batch_norm(name='g_bn_e3a')
        self.g_bn_e4a = batch_norm(name='g_bn_e4a')
        self.g_bn_e5a = batch_norm(name='g_bn_e5a')

        self.g_bn_e1b = batch_norm(name='g_bn_e1b')
        self.g_bn_e2b = batch_norm(name='g_bn_e2b')
        self.g_bn_e3b = batch_norm(name='g_bn_e3b')
        self.g_bn_e4b = batch_norm(name='g_bn_e4b')
        self.g_bn_e5b = batch_norm(name='g_bn_e5b')

        self.g_bn_d1a = batch_norm(name='g_bn_d1a')
        self.g_bn_d2a = batch_norm(name='g_bn_d2a')
        self.g_bn_d3a = batch_norm(name='g_bn_d3a')
        self.g_bn_d4a = batch_norm(name='g_bn_d4a')

        self.g_bn_d1b = batch_norm(name='g_bn_d1b')
        self.g_bn_d2b = batch_norm(name='g_bn_d2b')
        self.g_bn_d3b = batch_norm(name='g_bn_d3b')
        self.g_bn_d4b = batch_norm(name='g_bn_d4b')

        self.g_bn_d1c = batch_norm(name='g_bn_d1c')
        self.g_bn_d2c = batch_norm(name='g_bn_d2c')
        self.g_bn_d3c = batch_norm(name='g_bn_d3c')
        self.g_bn_d4c = batch_norm(name='g_bn_d4c')


        # END build model
        self.build_model()
        self.dataset_sx_tr = self.data_size(data='train')
        self.dataset_sx_te = self.data_size(data='test')
        self.dataset_sx_v = self.data_size(data='val')

        print(self.model_name)

    def data_size(self,data='train'):

        if data == 'train':
            fnames = np.array(glob(self.train_path + '*'))
            return len(fnames)
        if data == 'test':
            fnames = np.array(glob(self.test_path + '*'))
            return len(fnames)
        if data == 'val':
            fnames = np.array(glob(self.val_path + '*'))
            return len(fnames)

    def build_gen(self,data='train'):

        if data == 'train':
            fnames = np.array(glob(self.train_path + '*'))
            sz = self.dataset_sx_tr
        if data == 'test':
            fnames = np.array(glob(self.test_path + '*'))
            sz = self.dataset_sx_te
        if data == 'val':
            fnames = np.array(glob(self.val_path + '*'))
            sz = self.dataset_sx_v

        if sz % self.batch_size == 0:
            batch_num = sz // self.batch_size
        else:
            batch_num = sz // self.batch_size + 1

        while True:
            perm = np.random.permutation(sz)
            fnames = fnames[perm]

            for k in range(batch_num):

                yield self.open_batch(fnames[k*self.batch_size:(k+1)*self.batch_size])

    def open_batch(self,fs):

        inp_array = np.zeros((self.batch_size,256,256,1))
        lab_array = np.zeros((self.batch_size,256,256,1))

        for k in range(self.batch_size):

            fname = fs[k]
            f = h5py.File(fname, "r")
            inp,lab = f['lim'][:],f['full'][:]

            inp = inp[...,np.newaxis]
            lab = lab[...,np.newaxis]

            inp_array[k,...] = inp
            lab_array[k,...] = lab

        return inp_array,lab_array

    def build_model(self):

        # Declare Inputs
        self.x = tf.placeholder(tf.float32,[self.batch_size,256,256,1])
        self.y = tf.placeholder(tf.float32,[self.batch_size,256,256,1])

        # Create model and params
        self.out = self.model(self.x)

        # Loss
        self.loss = tf.nn.l2_loss(self.out-self.y)
        # Saver object
        self.saver = tf.train.Saver()

    def model(self, x):

        # model should take training input and produce transformed as output

        e1_a = conv2d(x, 64, name='g_e1_conv_a',k_h = 3,k_w=3)
        e1_b = self.g_bn_e1a(conv2d(lrelu(e1_a), 64, name='g_e1_conv_b',k_h = 3,k_w=3))
        e1_c = self.g_bn_e1b(conv2d(lrelu(e1_b), 64, name='g_e1_conv_c',k_h = 3,k_w=3))

        m1 = max_pool(e1_c)

        e2_a = self.g_bn_e2a(conv2d(lrelu(m1), 128, name='g_e2_conv_a',k_h = 3,k_w=3))
        e2_b = self.g_bn_e2b(conv2d(lrelu(e2_a), 128, name='g_e2_conv_b',k_h = 3,k_w=3))

        m2 = max_pool(e2_b)

        e3_a = self.g_bn_e3a(conv2d(lrelu(m2), 256, name='g_e3_conv_a',k_h = 3,k_w=3))
        e3_b = self.g_bn_e3b(conv2d(lrelu(e3_a), 256, name='g_e3_conv_b',k_h = 3,k_w=3))

        m3 = max_pool(e3_b)

        e4_a = self.g_bn_e4a(conv2d(lrelu(m3), 512, name='g_e4_conv_a',k_h = 3,k_w=3))
        e4_b = self.g_bn_e4b(conv2d(lrelu(e4_a), 512, name='g_e4_conv_b',k_h = 3,k_w=3))

        m4 = max_pool(e4_b)

        e5_a = self.g_bn_e5a(conv2d(lrelu(m4), 1024, name='g_e5_conv_a',k_h = 3,k_w=3))
        e5_b = self.g_bn_e5b(conv2d(lrelu(e5_a), 1024, name='g_e5_conv_b',k_h = 3,k_w=3))

        d1, d1_w, d1_b = deconv2d(lrelu(e5_b),[self.batch_size, 32, 32, 1024], name='g_d1', with_w=True)
        d1 = self.g_bn_d1a(d1)
        d1 = tf.concat([d1, e4_b],3)
        #
        d1_a = self.g_bn_d1b(conv2d(lrelu(d1), 512, name='g_d1_conv_a',k_h = 3,k_w=3))
        d1_b = self.g_bn_d1c(conv2d(lrelu(d1_a), 512, name='g_d1_conv_b',k_h = 3,k_w=3))

        d2, d2_w, d2_b = deconv2d(lrelu(d1_b),[self.batch_size, 64, 64, 512], name='g_d2', with_w=True)
        d2 = self.g_bn_d2a(d2)
        d2 = tf.concat([d2, e3_b],3)

        d2_a = self.g_bn_d2b(conv2d(lrelu(d2), 256, name='g_d2_conv_a',k_h = 3,k_w=3))
        d2_b = self.g_bn_d2c(conv2d(lrelu(d2_a), 256, name='g_d2_conv_b',k_h = 3,k_w=3))

        d3, d3_w, d3_b = deconv2d(lrelu(d2_b),[self.batch_size, 128, 128, 256], name='g_d3', with_w=True)
        d3 = self.g_bn_d3a(d3)
        d3 = tf.concat([d3, e2_b],3)

        d3_a = self.g_bn_d3b(conv2d(lrelu(d3), 128, name='g_d3_conv_a',k_h = 3,k_w=3))
        d3_b = self.g_bn_d3c(conv2d(lrelu(d3_a), 128, name='g_d3_conv_b',k_h = 3,k_w=3))

        d4, d4_w, d4_b = deconv2d(lrelu(d3_b),[self.batch_size, 256, 256, 128], name='g_d4', with_w=True)
        d4 = self.g_bn_d4a(d4)
        d4 = tf.concat([d4, e1_b],3)

        d4_a = self.g_bn_d4b(conv2d(lrelu(d4), 64, name='g_d4_conv_a',k_h = 3,k_w=3))
        d4_b = self.g_bn_d4c(conv2d(lrelu(d4_a), 64, name='g_d4_conv_b',k_h = 3,k_w=3))

        resid = conv2d(d4_b, 1, k_h=1, k_w=1, name='residual')

        out = tf.add(resid,x,name='out')

        return out

    def init(self):
        # function to call to initialse variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def train(self, epochs):

        # training op created
        train_op = tf.train.AdamOptimizer(1e-2).minimize(self.loss)
        self.init()

        counter = 0
        for k in range(epochs):
            print('[epoch {}]\n'.format(k+1))
            self.gen = self.build_gen()
            for i in range(self.dataset_sx_tr):
                counter += 1
                inp,label = next(self.gen)
                curr_loss = self.sess.run(self.loss,
                                     feed_dict={self.x:inp,self.y:label})

                _ = self.sess.run(train_op,
                                  feed_dict={self.x:inp,self.y:label})

                if i % 50 == 0:
                    self.save(k,i,counter)
                    print('[iteration {}] -> val error: {}'.format(i,self.val_error()))
            print('\ntrain error: {}\ntest error: {}\nval error: {}\n\n'.format(self.train_error(),
                                                                            self.test_error(),
                                                                            self.val_error()))

    def val_error(self):

        gen = self.build_gen('val')

        error = 0
        for i in range(self.dataset_sx_v):
            x,y = next(gen)
            error += self.sess.run(self.loss,feed_dict={self.x:x,self.y:y})

        return error/self.dataset_sx_v

    def train_error(self):

        gen = self.build_gen()

        error = 0
        for i in range(self.dataset_sx_tr//4):
            x,y = next(gen)
            error += self.sess.run(self.loss,feed_dict={self.x:x,self.y:y})

        return error/(self.dataset_sx_tr//4)

    def test_error(self):

        gen = self.build_gen('test')

        error = 0
        for i in range(self.dataset_sx_te//4):
            x,y = next(gen)
            error += self.sess.run(self.loss,feed_dict={self.x:x,self.y:y})

        return error/(self.dataset_sx_te//4)

    def save(self, batch_num, iteration_num, counter):

        # to save model state

        if not os.path.exists(self.model_name):
            os.makedirs(self.model_name)

        self.saver.save(self.sess,
                        os.path.join(self.model_name, self.model_name),
                        global_step=counter)

    def load(self, checkpoint_dir):

        self.model_name = checkpoint_dir

        # to load model state
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False
