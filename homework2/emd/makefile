# tensorflow = /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/include
# tensorflowlib = /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow

# all: tf_emddistance_so.so 
# .PHONY : all

# tf_emddistance_so.so:  tf_emddistance.cpp
# 	g++ -std=c++11 tf_emddistance.cpp  -o tf_emddistance_so.so -shared -fPIC -I $(tensorflow) -I $(tensorflow)/external/nsync/public -ltensorflow_framework -L$(tensorflowlib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0
nvcc = /Developer/NVIDIA/CUDA-9.1/bin
cudalib = /Developer/NVIDIA/CUDA-9.1/lib
tensorflow = /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/include
tensorflowlib = /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow

all: tf_emddistance_so.so 
.PHONY : all

tf_emddistance_so.so:  tf_emddistance.cpp
	g++ -std=c++11 tf_emddistance.cpp  -o tf_emddistance_so.so -shared -fPIC -I $(tensorflow) -I $(tensorflow)/external/nsync/public -ltensorflow_framework -L $(tensorflowlib) -O2 -lcudart -L $(cudalib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0

	# g++ -std=c++11 tf_emddistance.cpp  -o tf_emddistance_so.so -shared -fPIC -I $(tensorflow) -I $(tensorflow)/external/nsync/public -ltensorflow_framework -L$(tensorflowlib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0
