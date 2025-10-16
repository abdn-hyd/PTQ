# export cuda install path: export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc
export CUDACXX=/usr/local/cuda/bin/nvcc
# using which gcc $ g++ to determine their location, the path should be global shared in zsh or bash shell, otherwise rewrite the rc file
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# complile the cutlass class locally
cd submodules/cutlass
rm -rf build
mkdir -p build && cd build
# DCUTLASS_NVCC_ARCHS will be used to specify the category of Nvidia GPU, using RTX 40X0 as default.
# V100, TitanV: 70
# RTX 20x0, T4: 75
# A100: 80
# A10, RTX 30x0: 86
# RTX 40x0, L40: 89
# H100, H200: 90
# B200: 100
# B300: 103
# DRIVE Thor: 110
# RTX 50x0: 120
# DGX Spark: 121
# ------------------------------------------------------------------
# partial compilation as default from tools/profiler/CMakeLists.txt
# modules including:
# gemm, grouped gemm, block scaled gemm, blockwise gemm, sparse gemm
# trmm, symm, conv_2d, conv_3d
# ------------------------------------------------------------------
cmake .. -DCUTLASS_NVCC_ARCHS='80;86;89;90;120' -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_UNITY_BUILD_ENABLED=ON
make -j 16
