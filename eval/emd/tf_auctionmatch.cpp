#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <algorithm>
#include <vector>
#include <math.h>
using namespace tensorflow;
REGISTER_OP("AuctionMatch")
	.Input("xyz1: float32")
	.Input("xyz2: float32")
	.Output("matchl: int32")
	.Output("matchr: int32");
void AuctionMatchLauncher(int b,int n,const float * xyz1,const float * xyz2,int * matchl,int * matchr,float * cost);

class AuctionMatchGpuOp: public OpKernel{
	public:
		explicit AuctionMatchGpuOp(OpKernelConstruction* context):OpKernel(context){}
		void Compute(OpKernelContext * context)override{
			const Tensor& xyz1_tensor=context->input(0);
			OP_REQUIRES(context,xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3,errors::InvalidArgument("ApproxMatch expects (batch_size,num_points,3) xyz1 shape"));
			auto xyz1_flat=xyz1_tensor.flat<float>();
			const float * xyz1=&(xyz1_flat(0));
			int b=xyz1_tensor.shape().dim_size(0);
			int n=xyz1_tensor.shape().dim_size(1);
			OP_REQUIRES(context,n<=4096,errors::InvalidArgument("AuctionMatch handles at most 4096 dataset points"));

			const Tensor& xyz2_tensor=context->input(1);
			OP_REQUIRES(context,xyz2_tensor.dims()==3 && xyz2_tensor.shape().dim_size(2)==3 && xyz2_tensor.shape().dim_size(0)==b && xyz2_tensor.shape().dim_size(1)==n,errors::InvalidArgument("AuctionMatch expects (batch_size,num_points,3) xyz2 shape, and shape must match with xyz1"));
			auto xyz2_flat=xyz2_tensor.flat<float>();
			const float * xyz2=&(xyz2_flat(0));

			Tensor * matchl_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n},&matchl_tensor));
			auto matchl_flat=matchl_tensor->flat<int>();
			int * matchl=&(matchl_flat(0));
			Tensor * matchr_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,n},&matchr_tensor));
			auto matchr_flat=matchr_tensor->flat<int>();
			int * matchr=&(matchr_flat(0));

			Tensor temp_tensor;
			OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<float>::value,TensorShape{b,n,n},&temp_tensor));
			auto temp_flat=temp_tensor.flat<float>();
			float * temp=&(temp_flat(0));

			AuctionMatchLauncher(b,n,xyz1,xyz2,matchl,matchr,temp);
		}
};
REGISTER_KERNEL_BUILDER(Name("AuctionMatch").Device(DEVICE_GPU), AuctionMatchGpuOp);
