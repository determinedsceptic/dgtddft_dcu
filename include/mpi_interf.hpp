/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lin Lin

This file is part of DGDFT. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
(2) Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
(3) Neither the name of the University of California, Lawrence Berkeley
National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches, or
upgrades to the features, functionality or performance of the source code
("Enhancements") to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to Lawrence Berkeley National
Laboratory, without imposing a separate written license agreement for such
Enhancements, then you hereby grant the following license: a non-exclusive,
royalty-free perpetual license to install, use, modify, prepare derivative
works, incorporate into other computer software, distribute, and sublicense
such enhancements or derivative works thereof, in binary and source code form.
 */
/// @file mpi_interf.hpp
/// @brief Interface with MPI to facilitate communication.
/// @date 2012-11-03
#ifndef _MPI_INTERF_HPP_
#define _MPI_INTERF_HPP_

#include  "environment.hpp"
#ifdef GPU
// #include "cuda.h"
// #include "cuda_runtime.h"
#include <hip/hip_runtime.h>
#endif
namespace dgdft{

/// @namespace mpi
///
/// @brief Interface with MPI to facilitate communication.
namespace mpi{

// *********************************************************************
// Allgatherv
//
// NOTE: The interface is quite preliminary.
// *********************************************************************
void Allgatherv( 
    std::vector<Int>& localVec, 
    std::vector<Int>& allVec,
    MPI_Comm          comm );


// *********************************************************************
// Send / Recv for stringstream 
//
// Isend / Irecv is not here because the size and content has to be 
// communicated separately for non-blocking communication.
// *********************************************************************

void Send( std::stringstream& sstm, Int dest, Int tagSize, Int tagContent, 
    MPI_Comm comm );

void Recv ( std::stringstream& sstm, Int src, Int tagSize, Int tagContent, 
    MPI_Comm comm, MPI_Status& statSize, MPI_Status& statContent );

void Recv ( std::stringstream& sstm, Int src, Int tagSize, Int tagContent, 
    MPI_Comm comm );


// *********************************************************************
// Waitall
// *********************************************************************

void
  Wait    ( MPI_Request& req  );

void
  Waitall ( std::vector<MPI_Request>& reqs, std::vector<MPI_Status>& stats );

void
  Waitall ( std::vector<MPI_Request>& reqs );

// *********************************************************************
// Reduce
// *********************************************************************

void
  Reduce ( Real* sendbuf, Real* recvbuf, Int count, MPI_Op op, Int root, MPI_Comm comm );

void
  Reduce ( Complex* sendbuf, Complex* recvbuf, Int count, MPI_Op op, Int root, MPI_Comm comm );

void
  Allreduce ( Int* sendbuf, Int* recvbuf, Int count, MPI_Op op, MPI_Comm comm );

void
  Allreduce ( Real* sendbuf, Real* recvbuf, Int count, MPI_Op op, MPI_Comm comm );

void
  Allreduce ( float* sendbuf, float* recvbuf, Int count, MPI_Op op, MPI_Comm comm );

void
  Allreduce ( Complex* sendbuf, Complex* recvbuf, Int count, MPI_Op op, MPI_Comm comm );

// *********************************************************************
// Alltoall
// *********************************************************************

void
  Alltoallv ( Int *bufSend, Int *sizeSend, Int *displsSend, 
      Int *bufRecv, Int *sizeRecv, 
      Int *displsRecv, MPI_Comm comm );

void
  Alltoallv ( Real *bufSend, Int *sizeSend, Int *displsSend, 
      Real *bufRecv, Int *sizeRecv, 
      Int *displsRecv, MPI_Comm comm );

void
  Alltoallv ( Complex *bufSend, Int *sizeSend, Int *displsSend, 
      Complex *bufRecv, Int *sizeRecv, 
      Int *displsRecv, MPI_Comm comm );

#ifdef GPU
void 
  cuda_setDevice(MPI_Comm comm);
#endif
} // namespace mpi

} // namespace dgdft



#endif // _MPI_INTERF_HPP_

