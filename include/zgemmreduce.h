#ifndef ZGEMMREDUCE_H
#define ZGEMMREDUCE_H

#include <complex>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 执行复数矩阵乘法并计算每行的归约（元素平方模之和）
 *
 * @param transa 'N'表示不转置A，'T'表示转置A
 * @param transb 'N'表示不转置B，'T'表示转置B
 * @param m 矩阵A的行数
 * @param n 矩阵B的列数
 * @param k 矩阵A的列数/矩阵B的行数
 * @param A 列优先存储的复数矩阵A
 * @param lda 矩阵A的主维度
 * @param B 列优先存储的复数矩阵B
 * @param ldb 矩阵B的主维度
 * @param R 输出数组，存储计算结果（每行的归约值）
 * @return 0表示成功，非0表示错误
 */
int dcu_ColMajor_zgemm_reduce(char Transa, char Transb, 
                             int M, int N, int K,
                             const std::complex<double>* A, int lda,
                             const std::complex<double>* B, int ldb,
                             double* R);

int dcu_ColMajor_zgemm_reduce_baseline(char TransA, char TransB,
                                             int M, int N, int K,
                                             const std::complex<double>* A, int lda,
                                             const std::complex<double>* B, int ldb,
                                            double* R);

#ifdef __cplusplus
}
#endif

#endif // ZGEMMREDUCE_H