
/* Authors: Xinming Qin

This file is part of DGHF. All rights reserved.

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
/// @file numtns_decl.hpp
/// @brief Numerical tensor
/// @date 2010-09-27

#include "environment.hpp"
#include "numtns_impl.hpp"

namespace  dgdft{

// Templated form of numerical tensor
//
// The main advantage of this portable NumVec structure is that it can
// either own (owndata == true) or view (owndata == false) a piece of
// data.

template <class F>
  class NumFns
  {
  public:
    Int m_, n_, k_, l_;
    bool owndata_;
    F* data_;
  public:
    NumFns(Int m=0, Int n=0, Int k=0, Int l=0);

    NumFns(Int m, Int n, Int k, Int l, bool owndata, F* data);

    NumFns(const NumFns& C);

    ~NumFns();

    NumFns& operator=(const NumFns& C);

    void Resize(Int m, Int n, Int k, Int l);

    const F& operator()(Int a, Int b, Int c, Int d) const;

    F& operator()(Int a, Int b, Int c, Int d);

    bool IsOwnData() const { return owndata_; }

    F* Data() const { return data_; }

//    F* MatData (Int c, Int d) const; 
//    F* VecData (Int b, Int c, Int d) const;


    Int m() const { return m_; }

    Int n() const { return n_; }

    Int k() const { return k_; }

    Int l() const { return l_; }

    Int Size() const { return m_ * n_ * k_ * l_; }
  };


// Commonly used
typedef NumFns<bool>       BolNumFns;
typedef NumFns<Int>        IntNumFns;
typedef NumFns<Real>       DblNumFns;
typedef NumFns<Complex>    CpxNumFns;

// Utilities
template <class F> inline void SetValue(NumFns<F>& T, F val);

//template <class F> inline Real Energy(const NumFns<F>& T);


} // namespace dgdft
