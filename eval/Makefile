nvcc := ${CUDA_PATH}/bin/nvcc
cudalib := ${CUDA_PATH}/lib64
cudainclude := ${CUDA_PATH}/include
tensorflow := ${TENSORFLOW_PATH}/include

all: cd/tf_nndistance_so.so emd/tf_sampling_so.so emd/tf_auctionmatch_so.so
clean:
	rm cd/*.o cd/*.so emd/*.o emd/*.so
.PHONY : all clean

cd/tf_nndistance_so.so: cd/tf_nndistance_g.o cd/tf_nndistance.cpp
	g++ -std=c++11 cd/tf_nndistance.cpp cd/tf_nndistance_g.o -o cd/tf_nndistance_so.so -shared -fPIC -I $(tensorflow) -lcudart -L $(cudalib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0

cd/tf_nndistance_g.o: cd/tf_nndistance_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o cd/tf_nndistance_g.o cd/tf_nndistance_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

emd/tf_sampling_so.so: emd/tf_sampling.cpp emd/tf_sampling_g.o
	g++ -std=c++11 emd/tf_sampling.cpp emd/tf_sampling_g.o -o emd/tf_sampling_so.so -shared -fPIC -I $(tensorflow) -I $(cudainclude) -lcudart -L $(cudalib) -O2

emd/tf_sampling_g.o: emd/tf_sampling_g.cu
	$(nvcc) emd/tf_sampling_g.cu -o emd/tf_sampling_g.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

emd/tf_auctionmatch_so.so: emd/tf_auctionmatch.cpp emd/tf_auctionmatch_g.o
	g++ -std=c++11 emd/tf_auctionmatch.cpp emd/tf_auctionmatch_g.o -o emd/tf_auctionmatch_so.so -shared -fPIC -I $(tensorflow) -I $(cudainclude) -lcudart -L $(cudalib) -O2

emd/tf_auctionmatch_g.o: emd/tf_auctionmatch_g.cu
	$(nvcc) emd/tf_auctionmatch_g.cu -o emd/tf_auctionmatch_g.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_30
