module purge
module add compiler/rocm/dtk/24.04.3 compiler/devtoolset/7.3.1 compiler/intel/2017.5.239 mpi/hpcx/2.11.0/gcc-7.3.1

DIST_DIR=$( cd -P -- "$(dirname -- "$0")" && pwd -P )
export DGDFT_DIR=${DIST_DIR}
export HIP_DIR=/public/software/compiler/dtk/dtk-24.04.3
export LIBXC_DIR=/public/home/acsa/weihu/yaoyf/software/pwdft-lib/lib/libxc-6.2.2
export FFTW_DIR=/public/home/acsa/weihu/yaoyf/software/pwdft-lib/lib/fftw-3.3.10
export MKL_ROOT=/opt/hpc/software/compiler/intel/intel-compiler-2017.5.239/mkl
export YAML_DIR=/public/home/acsa/weihu/yaoyf/software/pwdft-lib/lib/yaml-cpp-0.7.0
#export MFFT_DIR=/public/home/whu_ustc/DGDFT-2DFFT/MFFT-test
export LD_LIBRARY_PATH=/public/home/acsa/weihu/yaoyf/software/pwdft-lib/lib/fftw-3.3.10/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/hpc/software/compiler/intel/intel-compiler-2017.5.239/mkl/lib/intel64:$LD_LIBRARY_PATH
