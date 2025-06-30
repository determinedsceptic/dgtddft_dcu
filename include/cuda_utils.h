/** \file cuda_utils.h
 * \brief Definitions of persistent cuda pointers. 
 *
 * Each type internally the same, just typed differently.
 * Remember to also include cuda_utils.cu, which 
 * contains manipulation functions.
 */
#ifdef GPU
#ifndef __HIP_PLATFORM_HCC__
#define __HIP_PLATFORM_HCC__
#endif
#ifndef _CUDA_UTILS_
#define _CUDA_UTILS_
#include <assert.h>
#include <stdio.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
#include <hip/hip_runtime.h>
// #include <cufft.h>
#include <rocfft/rocfft.h>
#include <hipblas/hipblas.h>
#include <hip/hip_complex.h>
// #include "cuComplex.h"
#define NSTREAM 1
#if 0
#define CPU2GPU hipMemcpyHostToDevice 
#define GPU2CPU hipMemcpyDeviceToHost 
#define GPU2GPU hipMemcpyDeviceToDevice 
#endif 
typedef  double2 hipDoubleComplex;
typedef  float2 hipComplex;
//Macros for n-dimensional array access (column major format)
#define DIM2(x, y, xdim)                                       ((y * xdim) + x)
#define DIM3(x, y, z, xdim, ydim)                              ((((z * ydim) + y) * xdim) + x)
#define DIM4(x1, x2, x3, x4, x1dim, x2dim, x3dim)              ((((((x4 * x3dim) + x3) * x2dim) + x2) * x1dim) + x2)
#define DIM5(x1, x2, x3, x4, x5, x1dim, x2dim, x3dim, x4dim)   ((((((((x5 * x4dim) + x4) * x3dim) + x3) * x2dim) + x2) * x1dim) + x1)

#define CUDA_CALL(function) {\
hipError_t err = function; \
if (err != hipSuccess) \
  fprintf(stderr, "CURROR [%s,%d]: %s \n", \
  __FILE__,  __LINE__, hipGetErrorString(err)); \
}

#define CUDA_FFT_CALL(function) {\
hipfftResult err = function; \
if (err != CUFFT_SUCCESS) \
  fprintf(stderr, "CURROR [%s,%d]: %d \n", \
  __FILE__,  __LINE__, err); \
}

#if 0
/** N-dimensional double pointer */
typedef struct ndim_double {
  unsigned size;
  unsigned ndims;
  unsigned *dim;     //GPU
  unsigned lead_dim_size;
  double *ptr;      //current position, GPU
  double *start_pt;  //Starting point for access, GPU
} *ndim_double_ptr;

/** N dimensional cuDoubleComplex ptr */
typedef struct ndim_complex {
  unsigned size;
  unsigned ndims;
  unsigned *dim;
  unsigned lead_dim_size;
  hipDoubleComplex *ptr;        
  hipDoubleComplex *start_pt;
} *ndim_complex_ptr;

/**  int ptr */
typedef struct ndim_int {
  unsigned size;
  unsigned ndims;
  unsigned *dim;
  unsigned lead_dim_size;
  int *ptr;
  int *start_pt;
} *ndim_int_ptr;
#endif
/** void pointer, used mostly for casting */
//typedef struct void_p{
//  unsigned int size;
//  void* ptr;
//} void_p;
//
///** double pointer */
//typedef struct double_p{
//  unsigned int size;
//  double* ptr;
//} double_p;
//
///** float pointer */
//typedef struct float_p{
//  unsigned int size;
//  float* ptr;
//} float_p;
//
///** int pointer */
//typedef struct int_p{
//  unsigned int size;
//  int* ptr;
//} int_p;
//
///** Double Complex pointer (foo.x is real, foo.y is imag) */
//typedef struct cuDoubleComplex_p{
//  unsigned int size;
//  cuDoubleComplex* ptr;
//} cuDoubleComplex_p;
//
///** Complex pointer (foo.x is real, foo.y is imag) */
//typedef struct cuComplex_p{
//  unsigned int size;
//  cuComplex* ptr;
//} cuComplex_p;
//
//extern "C"{

extern double * dev_vtot;
extern double * dev_gkkR2C;
extern int    * dev_idxFineGridR2C;

extern int    * dev_NLindex;
extern int    * dev_NLpart;
extern double * dev_NLvecFine;
extern double * dev_atom_weight;
extern double * dev_temp_weight;
extern double * dev_TeterPrecond;
extern int totPart_gpu;

extern bool vtot_gpu_flag;
extern bool NL_gpu_flag;
extern bool teter_gpu_flag;

void cuda_setValue( hipComplex* dev, hipComplex val, int len );
void cuda_setValue( hipDoubleComplex* dev, hipDoubleComplex val, int len );
void cuda_setValue( double* dev, double val, int len );
void cuda_setValue( float* dev, float val, int len );
void cuda_memcpy_GPU2GPU( void * dest, void * src, size_t size);
void cuda_memcpy_GPU2CPU( void *cpu, void * gpu, size_t size );
void cuda_memcpy_CPU2GPU( void *gpu, void * cpu, size_t size );
void cuda_free( void *ptr);
void *cuda_malloc( size_t size);
void cuda_interpolate_wf_C2F( hipDoubleComplex * coarse_psi, hipDoubleComplex * fine_psi, int * index, int len, double factor);
void cuda_interpolate_wf_F2C( hipDoubleComplex * fine_psi, hipDoubleComplex * coarse_psi, int * index, int len, double factor);
void cuda_laplacian( hipDoubleComplex* psi, double * gkk, int len);
void cuda_vtot( double* psi, double * vtot, int len);
void cuda_memory(void);
void cuda_calculate_nonlocal( double * psiUpdate, double * psi, double * NL, int * index, int * parts,  double * atom_weight, double * weight, int blocks);
void cuda_teter( hipDoubleComplex* psi, double * vtot, int len);
void cuda_mapping_to_buf( double * buf, double * psi, int * index, int len );
void cuda_mapping_from_buf( double * psi, double * buf, int * index, int len );
void cuda_calculate_Energy( double * psi, double * energy, int nbands, int bandLen);
void cuda_batch_Scal( double * psi, double * vec, int nband, int bandLen);
void cu_X_Equal_AX_minus_X_eigVal( double * Xtemp, double * AX, double * X, double * eigen, int nbands, int bandLen);
void cuda_init_vtot();
void cuda_clean_vtot();
void cuda_set_vtot_flag();
void cuda_DMatrix_Add( double * A , double * B, int m, int n);
void cuda_Axpyz( double * X, double alpha, double * Y, double beta, double * Z, int length);
void cuda_cal_recvk( int * recvk, int * recvdisp, int width, int heightLocal, int mpisize);
void cuda_cal_sendk( int * sendk, int * senddispl, int widthLocal, int height, int heightBlockSize, int mpisize);
void cuda_hadamard_product( double * in1, double * in2, double * out, int length);
void cuda_set_vector( double * out, double *in, int length);
void cuda_XTX( hipDoubleComplex * X, double * Y, int length);
void cuda_reduce( double * density, double * sum, int nbands, int bandLen);
void cuda_sync();
//}
#endif
#endif
