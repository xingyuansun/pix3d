#include <cstdio>
__global__ void AuctionMatchKernel(int b,int n,const float * __restrict__ xyz1,const float * __restrict__ xyz2,int * matchl,int * matchr,float * cost){
	//this kernel handles up to 4096 points
	const int NMax=4096;
	__shared__ short Queue[NMax];
	__shared__ short matchrbuf[NMax];
	__shared__ float pricer[NMax];
	__shared__ float bests[32][3];
	__shared__ int qhead,qlen;
	const int BufLen=2048;
	__shared__ float buf[BufLen];
	for (int bno=blockIdx.x;bno<b;bno+=gridDim.x){
		int cnt=0;
		float tolerance=1e-4;
		for (int j=threadIdx.x;j<n;j+=blockDim.x)
			matchl[bno*n+j]=-1;
		for (int j=threadIdx.x;j<n;j+=blockDim.x)
			matchrbuf[j]=-1;
		for (int j=threadIdx.x;j<n;j+=blockDim.x)
			Queue[j]=j;
		for (int j=threadIdx.x;j<n;j+=blockDim.x)
			pricer[j]=0;
		const int Block=512;
		for (int k0=0;k0<n;k0+=Block){
			int k1=min(n,k0+Block);
			for (int k=threadIdx.x;k<(k1-k0)*3;k+=blockDim.x)
				buf[k]=xyz1[bno*n*3+k0*3+k];
			__syncthreads();
			for (int j=threadIdx.x;j<n;j+=blockDim.x){
				float x2=xyz2[bno*n*3+j*3+0];
				float y2=xyz2[bno*n*3+j*3+1];
				float z2=xyz2[bno*n*3+j*3+2];
				for (int k=k0;k<k1;k++){
					float x1=buf[(k-k0)*3+0];
					float y1=buf[(k-k0)*3+1];
					float z1=buf[(k-k0)*3+2];
					float d=sqrtf((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2));
					cost[blockIdx.x*n*n+k*n+j]=d;
				}
			}
			__syncthreads();
		}
		if (threadIdx.x==0){
			qhead=0;
			qlen=n;
		}
		__syncthreads();
		int loaded=0;
		float value9,value10,value11,value12,value13,value14,value15,value16;
		while (qlen){
			int i=Queue[qhead];
			int i2;
			if (qhead+1<n)
				i2=Queue[qhead+1];
			else
				i2=Queue[0];
			float best=1e38f,best2=1e38f;
			int bestj=0;
			if (n==blockDim.x*8){
				int j=threadIdx.x;
				float value1,value2,value3,value4,value5,value6,value7,value8;
				if (loaded){
					value1=value9+pricer[j];
					value2=value10+pricer[j+blockDim.x];
					value3=value11+pricer[j+blockDim.x*2];
					value4=value12+pricer[j+blockDim.x*3];
					value5=value13+pricer[j+blockDim.x*4];
					value6=value14+pricer[j+blockDim.x*5];
					value7=value15+pricer[j+blockDim.x*6];
					value8=value16+pricer[j+blockDim.x*7];
					loaded=0;
				}else{
					value1=cost[blockIdx.x*n*n+i*n+j]+pricer[j];
					value2=cost[blockIdx.x*n*n+i*n+j+blockDim.x]+pricer[j+blockDim.x];
					value3=cost[blockIdx.x*n*n+i*n+j+blockDim.x*2]+pricer[j+blockDim.x*2];
					value4=cost[blockIdx.x*n*n+i*n+j+blockDim.x*3]+pricer[j+blockDim.x*3];
					value5=cost[blockIdx.x*n*n+i*n+j+blockDim.x*4]+pricer[j+blockDim.x*4];
					value6=cost[blockIdx.x*n*n+i*n+j+blockDim.x*5]+pricer[j+blockDim.x*5];
					value7=cost[blockIdx.x*n*n+i*n+j+blockDim.x*6]+pricer[j+blockDim.x*6];
					value8=cost[blockIdx.x*n*n+i*n+j+blockDim.x*7]+pricer[j+blockDim.x*7];
					value9=cost[blockIdx.x*n*n+i2*n+j];
					value10=cost[blockIdx.x*n*n+i2*n+j+blockDim.x];
					value11=cost[blockIdx.x*n*n+i2*n+j+blockDim.x*2];
					value12=cost[blockIdx.x*n*n+i2*n+j+blockDim.x*3];
					value13=cost[blockIdx.x*n*n+i2*n+j+blockDim.x*4];
					value14=cost[blockIdx.x*n*n+i2*n+j+blockDim.x*5];
					value15=cost[blockIdx.x*n*n+i2*n+j+blockDim.x*6];
					value16=cost[blockIdx.x*n*n+i2*n+j+blockDim.x*7];
					loaded=qlen>1;
				}
				int vj,vj2,vj3,vj4;
				if (value1<value2){
					vj=j;
				}else{
					vj=j+blockDim.x;
					float t=value1;
					value1=value2;
					value2=t;
				}
				if (value3<value4){
					vj2=j+blockDim.x*2;
				}else{
					vj2=j+blockDim.x*3;
					float t=value3;
					value3=value4;
					value4=t;
				}
				if (value5<value6){
					vj3=j+blockDim.x*4;
				}else{
					vj3=j+blockDim.x*5;
					float t=value5;
					value5=value6;
					value6=t;
				}
				if (value7<value8){
					vj4=j+blockDim.x*6;
				}else{
					vj4=j+blockDim.x*7;
					float t=value7;
					value7=value8;
					value8=t;
				}
				if (value1<value3){
					value2=fminf(value2,value3);
				}else{
					value2=fminf(value1,value4);
					value1=value3;
					vj=vj2;
				}
				if (value5<value7){
					value6=fminf(value6,value7);
				}else{
					value6=fminf(value5,value8);
					value5=value7;
					vj3=vj4;
				}
				if (value1<value5){
					best=value1;
					bestj=vj;
					best2=fminf(value2,value5);
				}else{
					best2=fminf(value1,value6);
					best=value5;
					bestj=vj3;
				}
			}else if (n>=blockDim.x*4){
				for (int j=threadIdx.x;j<n;j+=blockDim.x*4){
					float value1=cost[blockIdx.x*n*n+i*n+j]+pricer[j];
					float value2=cost[blockIdx.x*n*n+i*n+j+blockDim.x]+pricer[j+blockDim.x];
					float value3=cost[blockIdx.x*n*n+i*n+j+blockDim.x*2]+pricer[j+blockDim.x*2];
					float value4=cost[blockIdx.x*n*n+i*n+j+blockDim.x*3]+pricer[j+blockDim.x*3];
					int vj,vj2;
					if (value1<value2){
						vj=j;
					}else{
						vj=j+blockDim.x;
						float t=value1;
						value1=value2;
						value2=t;
					}
					if (value3<value4){
						vj2=j+blockDim.x*2;
					}else{
						vj2=j+blockDim.x*3;
						float t=value3;
						value3=value4;
						value4=t;
					}
					if (value1<value3){
						value2=fminf(value2,value3);
					}else{
						value2=fminf(value1,value4);
						value1=value3;
						vj=vj2;
					}
					if (best<value1){
						best2=fminf(best2,value1);
					}else{
						best2=fminf(best,value2);
						best=value1;
						bestj=vj;
					}
				}
			}else if (n>=blockDim.x*2){
				for (int j=threadIdx.x;j<n;j+=blockDim.x*2){
					float value1=cost[blockIdx.x*n*n+i*n+j]+pricer[j];
					float value2=cost[blockIdx.x*n*n+i*n+j+blockDim.x]+pricer[j+blockDim.x];
					int vj;
					if (value1<value2){
						vj=j;
					}else{
						vj=j+blockDim.x;
						float t=value1;
						value1=value2;
						value2=t;
					}
					if (best<value1){
						best2=fminf(best2,value1);
					}else{
						best2=fminf(best,value2);
						best=value1;
						bestj=vj;
					}
				}
			}else{
				for (int j=threadIdx.x;j<n;j+=blockDim.x){
					float value=cost[blockIdx.x*n*n+i*n+j]+pricer[j];
					if (best<value){
						best2=fminf(best2,value);
					}else{
						best2=best;
						bestj=j;
						best=value;
					}
				}
			}
			for (int i=16;i>0;i>>=1){
				float b1=__shfl_down(best,i,32);
				float b2=__shfl_down(best2,i,32);
				int bj=__shfl_down(bestj,i,32);
				if (best<b1){
					best2=fminf(b1,best2);
				}else{
					best=b1;
					best2=fminf(best,b2);
					bestj=bj;
				}
			}
			if ((threadIdx.x&31)==0){
				bests[threadIdx.x>>5][0]=best;
				bests[threadIdx.x>>5][1]=best2;
				*(int*)&bests[threadIdx.x>>5][2]=bestj;
			}
			__syncthreads();
			int nn=blockDim.x>>5;
			if (threadIdx.x<nn){
				best=bests[threadIdx.x][0];
				best2=bests[threadIdx.x][1];
				bestj=*(int*)&bests[threadIdx.x][2];
				for (int i=nn>>1;i>0;i>>=1){
					float b1=__shfl_down(best,i,32);
					float b2=__shfl_down(best2,i,32);
					int bj=__shfl_down(bestj,i,32);
					if (best<b1){
						best2=fminf(b1,best2);
					}else{
						best=b1;
						best2=fminf(best,b2);
						bestj=bj;
					}
				}
			}
			if (threadIdx.x==0){
				float delta=best2-best+tolerance;
				qhead++;
				qlen--;
				if (qhead>=n)
					qhead-=n;
				int old=matchrbuf[bestj];
				pricer[bestj]+=delta;
				cnt++;
				if (old!=-1){
					int ql=qlen;
					int tail=qhead+ql;
					qlen=ql+1;
					if (tail>=n)
						tail-=n;
					Queue[tail]=old;
				}
				if (cnt==(40*n)){
					if (tolerance==1.0)
						qlen=0;
					tolerance=fminf(1.0,tolerance*100);
					cnt=0;
				}
			}
			__syncthreads();
			if (threadIdx.x==0){
				matchrbuf[bestj]=i;
			}
		}
		__syncthreads();
		for (int j=threadIdx.x;j<n;j+=blockDim.x)
			matchr[bno*n+j]=matchrbuf[j];
		for (int j=threadIdx.x;j<n;j+=blockDim.x)
			matchl[bno*n+matchrbuf[j]]=j;
		__syncthreads();
	}
}
void AuctionMatchLauncher(int b,int n,const float * xyz1,const float * xyz2,int * matchl,int * matchr,float * cost){
	AuctionMatchKernel<<<32,512>>>(b,n,xyz1,xyz2,matchl,matchr,cost);
}

