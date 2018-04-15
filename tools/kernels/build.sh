TF_CFLAGS=($(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'))
TF_LFLAGS=($(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'))
# g++ -std=c++11 -shared $1.cc -o $1.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
nvcc -std=c++11 -c -o $1.cu.o $1.cu.cc ${TF_CFLAGS[@]} -x cu -Xcompiler -fPIC
g++ -std=c++11 -shared -o $1.so $1.cc $1.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}