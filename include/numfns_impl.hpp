/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Authors: Xinming Qin

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
/// @file numtns_impl.hpp
/// @brief Implementation of numerical tensor.
/// @date 2023-1-30

#include  "numfns_decl.hpp"

namespace  dgdft{

template <class F> 
  inline NumFns<F>::NumFns(Int m, Int n, Int k, Int l): m_(m), n_(n), k_(k), l_(l), owndata_(true) {
    if(m_>0 && n_>0 && k_>0 && l_>0) { data_ = new F[m_*n_*k_*l_]; if( data_ == NULL ) ErrorHandling("Cannot allocate memory."); } else data_=NULL;
  }

template <class F> 
  inline NumFns<F>::NumFns(Int m, Int n, Int k, Int l, bool owndata, F* data): m_(m), n_(n), k_(k),l_(l), owndata_(owndata) {
    if(owndata_) {
      if(m_>0 && n_>0 && k_>0 && l_>0) { data_ = new F[m_*n_*k_*l_]; if( data_ == NULL ) ErrorHandling("Cannot allocate memory."); } else data_=NULL;
      if(m_>0 && n_>0 && k_>0 && l_>0) { for(Int i=0; i<m_*n_*k_*l_; i++) data_[i] = data[i]; }
    } else {
      data_ = data;
    }
  }

template <class F> 
  inline NumFns<F>::NumFns(const NumFns& C): m_(C.m_), n_(C.n_), k_(C.k_), l_(C.l_), owndata_(C.owndata_) {
    if(owndata_) {
      if(m_>0 && n_>0 && k_>0 && l_>0) { data_ = new F[m_*n_*k_*l_]; if( data_ == NULL ) ErrorHandling("Cannot allocate memory."); } else data_=NULL;
      if(m_>0 && n_>0 && k_>0 && l_>0) { for(Int i=0; i<m_*n_*k_*l_; i++) data_[i] = C.data_[i]; }
    } else {
      data_ = C.data_;
    }
  }

template <class F> 
  inline NumFns<F>::~NumFns() { 
    if(owndata_) { 
      if(m_>0 && n_>0 && k_>0 && l_>0) { delete[] data_; data_ = NULL; } 
    }
  }

template <class F> 
  inline NumFns<F>& NumFns<F>::operator=(const NumFns& C) {
    // Do not copy if it is the same matrix.
    if(C.data_ == data_) return *this;

    if(owndata_) { 
      if(m_>0 && n_>0 && k_>0 && l_>0) { delete[] data_; data_ = NULL; } 
    }
    m_ = C.m_; n_=C.n_; k_=C.k_; l_==C.l_; owndata_=C.owndata_;
    if(owndata_) {
      if(m_>0 && n_>0 && k_>0 && l_>0) { data_ = new F[m_*n_*k_*l_]; if( data_ == NULL ) ErrorHandling("Cannot allocate memory."); } else data_=NULL;
      if(m_>0 && n_>0 && k_>0 && l_>0) { for(Int i=0; i<m_*n_*k_*l_; i++) data_[i] = C.data_[i]; }
    } else {
      data_ = C.data_;
    }
    return *this;
  }

template <class F> 
  inline void NumFns<F>::Resize(Int m, Int n, Int k, Int l)  {
    if( owndata_ == false ){
      ErrorHandling("Tensor being resized must own data.");
    }
    if(m_!=m || n_!=n || k_!=k || l_!=l) {
      if(m_>0 && n_>0 && k_>0 && l_>0) { delete[] data_; data_ = NULL; } 
      m_ = m; n_ = n; k_ = k; l_ = l;
      if(m_>0 && n_>0 && k_>0 && l_>0) { data_ = new F[m_*n_*k_*l_]; if( data_ == NULL ) ErrorHandling("Cannot allocate memory."); } else data_=NULL;
    }
  }

template <class F> 
  inline const F& NumFns<F>::operator()(Int a, Int b, Int c, Int d) const  {
#if ( _DEBUGlevel_ >= 1 )
    if( a < 0 || a >= m_ ||
        b < 0 || b >= n_ ||
        c < 0 || c >= k_ ||
        d < 0 || d >= l_ ) {
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << m_ << ", " << n_ << ", " << k_ << ", " << l_ << ")" << std::endl
        << "This index     ~ (" << a  << ", " << b  << ", " << c  << ", " << d  << ")" << std::endl;
      ErrorHandling( msg.str().c_str() );
    }
#endif
    return data_[a+b*m_*n_+c*m_*n_*k_];
  }

template <class F> 
  inline F& NumFns<F>:: operator()(Int a, Int b, Int c, Int d)  {
#if ( _DEBUGlevel_ >= 1 )
    if( a < 0 || a >= m_ ||
        b < 0 || b >= n_ ||
        c < 0 || c >= k_ ||
        d < 0 || d >= l_ ) {
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << m_ << ", " << n_ << ", " << k_ << ", " << l_ << ")" << std::endl
        << "This index     ~ (" << a  << ", " << b  << ", " << c  << ", " << d  << ")" << std::endl;
      ErrorHandling( msg.str().c_str() );
    }
#endif
    return data_[a+b*m_*n_+c*m_*n_*k_];
  }
/*
template <class F> 
  inline F* NumFns<F>::MatData (Int c, Int d) const {
#if ( _DEBUGlevel_ >= 1 )
    if( c < 0 || c >= k_  ||
        d < 0 || d >= l_ ) {
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << k_ << ", " << l_ << ")" << std::endl
        << "This index     ~ (" << c  << ", " << d  << ")" << std::endl;
      ErrorHandling( msg.str().c_str() );
    }
#endif
    return &(data_[d*m_*n_*k_ + c*m_*n_]);
  }

template <class F>
  inline F* NumFns<F>::VecData (Int b, Int c, Int d) const {
#if ( _DEBUGlevel_ >= 1 )
    if( j < 0 || j >= n_ ||
      std::ostringstream msg;
      msg
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << n_ << ", " << k_  << ", " << l_ << ")" << std::endl
        << "This index     ~ (" << b  << ", " << c   << ", " << d  << ")" << std::endl;
      ErrorHandling( msg.str().c_str() );
    }
#endif
    return &(data_[d*m_*n_*k_+c*m_*n_+b*m_]);
  }
*/
// *********************************************************************
// Utilities
// *********************************************************************

template <class F> inline void SetValue(NumFns<F>& T, F val)
{
  F *ptr = T.data_;
  for(Int i=0; i < T.m() * T.n() * T.k() * T.l(); i++) *(ptr++) = val; 

  return;
}

} // namespace dgdft
