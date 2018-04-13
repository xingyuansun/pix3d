import tensorflow as tf
from tensorflow.python.framework import ops
nn_distance_module = tf.load_op_library('./cd/tf_nndistance_so.so')
import numpy as np
import math


def nn_distance(xyz1,xyz2):
    '''
    Computes the distance of nearest neighbors for a pair of point clouds
    input: xyz1: (batch_size,#points_1,3)  the first point cloud
    input: xyz2: (batch_size,#points_2,3)  the second point cloud
    output: dist1: (batch_size,#point_1)   distance from first to second
    output: idx1:  (batch_size,#point_1)   nearest neighbor from first to second
    output: dist2: (batch_size,#point_2)   distance from second to first
    output: idx2:  (batch_size,#point_2)   nearest neighbor from second to first
    '''
    return nn_distance_module.nn_distance(xyz1,xyz2)


@ops.RegisterGradient('NnDistance')
def _nn_distance_grad(op,grad_dist1,grad_idx1,grad_dist2,grad_idx2):
    xyz1 = op.inputs[0]
    xyz2 = op.inputs[1]
    idx1 = op.outputs[1]
    idx2 = op.outputs[3]
    return nn_distance_module.nn_distance_grad(xyz1,xyz2,grad_dist1,idx1,grad_dist2,idx2)


def cd_dis(xyz1, xyz2, batch_size):
    assert xyz1.shape[1] == xyz2.shape[1] and xyz1.shape[2] == 3 and xyz2.shape[2] == 3
    all_scores = list()
    for idx in range(int(math.ceil(xyz1.shape[0] / batch_size))):
        indl = idx * batch_size
        indh = min(xyz1.shape[0], (idx + 1) * batch_size)
        scores = _cd_dis(xyz1[indl:indh].copy(), xyz2[indl:indh].copy())
        all_scores.append(scores)
    all_scores = np.concatenate(all_scores)
    return all_scores


def _cd_dis(xyz1, xyz2):
    assert xyz1.shape[1] == xyz2.shape[1] and xyz1.shape[2] == 3 and xyz2.shape[2] == 3
    xyz1 = np.array(xyz1, dtype=np.float32)
    xyz2 = np.array(xyz2, dtype=np.float32)
    with tf.Session('') as sess:
        inp1 = tf.constant(xyz1)
        inp2 = tf.constant(xyz2)
        reta, retb, retc, retd = nn_distance(inp1, inp2)

        sess.run(tf.global_variables_initializer())
        dist1, idx1, dist2, idx2, inp1_, inp2_ = sess.run([reta, retb, retc, retd, inp1, inp2])
        # check that input values are properly assigned
        assert np.absolute(inp1_ - xyz1).max() < 1e-5
        assert np.absolute(inp2_ - xyz2).max() < 1e-5
        if inp1_.mean() == 0 or inp2_.mean() == 0 or dist1.max() == 0 or dist2.max() == 0:
            import pdb; pdb.set_trace()
        return np.mean(np.sqrt(dist1), axis=1) + np.mean(np.sqrt(dist2), axis=1)

