// File: cuda_errors.c
// CUBLAS and CUFFT error checking.

// includes standard headers

#ifdef GPU  // only used for the GPU version of the PWDFT code. 
#ifndef _CUDA_ERRORS_HPP__
#define _CUDA_ERRORS_HPP__
#include <stdio.h>
#include <stdlib.h>
// #include <cublas_v2.h>
// #include <cufft.h>
#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>
#include <rocfft/rocfft.h>
/******************************************************/
// CUBLAS and CUFFT error checking, in library

// returns string for CUBLAS API error
char *cublasGetErrorString(hipblasStatus_t error);
char *cufftGetErrorString(rocfft_status error);
#endif
#endif
/******************************************************/
