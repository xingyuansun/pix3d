import tensorflow as tf
from tensorflow.python.framework import ops
sampling_module=tf.load_op_library('./emd/tf_sampling_so.so')
def prob_sample(inp,inpr):
	'''
input:
	batch_size * ncategory float32
	batch_size * npoints   float32
returns:
	batch_size * npoints   int32
	'''
	return sampling_module.prob_sample(inp,inpr)
ops.NoGradient('ProbSample')
@ops.RegisterShape('ProbSample')
def _prob_sample_shape(op):
	shape1=op.inputs[0].get_shape().with_rank(2)
	shape2=op.inputs[1].get_shape().with_rank(2)
	return [tf.TensorShape([shape2.dims[0],shape2.dims[1]])]
def gather_point(inp,idx):
	'''
input:
	batch_size * ndataset * 3   float32
	batch_size * npoints        int32
returns:
	batch_size * npoints * 3    float32
	'''
	return sampling_module.gather_point(inp,idx)
@ops.RegisterShape('GatherPoint')
def _gather_point_shape(op):
	shape1=op.inputs[0].get_shape().with_rank(3)
	shape2=op.inputs[1].get_shape().with_rank(2)
	return [tf.TensorShape([shape1.dims[0],shape2.dims[1],shape1.dims[2]])]
@ops.RegisterGradient('GatherPoint')
def _gather_point_grad(op,out_g):
	inp=op.inputs[0]
	idx=op.inputs[1]
	return [sampling_module.gather_point_grad(inp,idx,out_g),None]
def farthest_point_sample(npoint,inp):
	'''
input:
	int32
	batch_size * ndataset * 3   float32
returns:
	batch_size * npoint         int32
	'''
	return sampling_module.farthest_point_sample(npoint,inp)
ops.NoGradient('FarthestPointSample')

