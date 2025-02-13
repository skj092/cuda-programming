mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$HOME/code/cuda-programming/libtorch ..
cmake --build . --config Release

