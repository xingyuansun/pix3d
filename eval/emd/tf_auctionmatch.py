import os
import tensorflow as tf
from tensorflow.python.framework import ops
auctionmatch_module = tf.load_op_library('./emd/tf_auctionmatch_so.so')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import emd.tf_sampling as tf_sampling


def auction_match(xyz1,xyz2):
    '''
    input:
        xyz1 : batch_size * #points * 3
        xyz2 : batch_size * #points * 3
    returns:
        matchl : batch_size * #npoints
        matchr : batch_size * #npoints
    '''
    return auctionmatch_module.auction_match(xyz1,xyz2)


ops.NoGradient('AuctionMatch')


@ops.RegisterShape('AuctionMatch')
def _auction_match_shape(op):
    shape1 = op.inputs[0].get_shape().with_rank(3)
    shape2 = op.inputs[1].get_shape().with_rank(3)
    return [
        tf.TensorShape([shape1.dims[0],shape1.dims[1]]),
        tf.TensorShape([shape2.dims[0],shape2.dims[1]])
    ]


class EMD:
    def __init__(self, npoint, batch_size):
        self.npoint = npoint
        self.batch_size = batch_size

        with tf.device('/gpu:0'):
            self.xyz1_in = tf.placeholder(tf.float32,shape=(self.batch_size,self.npoint,3))
            self.xyz2_in = tf.placeholder(tf.float32,shape=(self.batch_size,self.npoint,3))
            self.matchl_out,self.matchr_out = auction_match(self.xyz1_in,self.xyz2_in)
            self.matched_out = tf_sampling.gather_point(self.xyz2_in,self.matchl_out)

    def emd_dis(self, pts1, pts2):
        assert pts1.shape == pts2.shape
        assert pts1.shape[1] <= 4096
        nvox = pts1.shape[0]
        assert self.npoint == pts1.shape[1]

        rst = []
        for i in range(int((nvox + self.batch_size - 1) / self.batch_size)):
            xyz1 = np.zeros((self.batch_size, self.npoint, 3), dtype=np.float32)
            xyz2 = np.zeros((self.batch_size, self.npoint, 3), dtype=np.float32)
            length = min(self.batch_size, nvox - self.batch_size * i)
            xyz1[0:length, :, :] = pts1[self.batch_size * i:self.batch_size * i + length, :, :]
            xyz2[0:length, :, :] = pts2[self.batch_size * i:self.batch_size * i + length, :, :]
            with tf.Session('') as sess:
                ret = sess.run(self.matched_out,feed_dict={self.xyz1_in:xyz1,self.xyz2_in:xyz2})
            dis = np.sqrt(np.sum((xyz1[0:length, :] - ret[0:length, :]) ** 2, axis=2))
            rst.extend(np.mean(dis, axis=1))
        return np.array(rst)

