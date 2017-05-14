import tensorflow as tf
import numpy as np

from model import recon2recon

sess = tf.Session()
model = recon2recon(sess,'arch1_10','../../../data/processed/train_10_500/',
                               '../../../data/processed/test_10_500/',
                               '../../../data/processed/val_10_500/')

model.train(10)
