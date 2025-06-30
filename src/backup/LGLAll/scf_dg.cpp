/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Authors: Lin Lin, Wei Hu, Amartya Banerjee, and Xinming Qin, and Junwei Feng

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
/// @file scf_dg.cpp
/// @brief Self consistent iteration using the DG method.
/// @date 2013-02-05
/// @date 2014-08-06 Add intra-element parallelization.
//
/// Xinming Qin xmqin03@gmail.com
/// @date 2023-10-17 Add Hybrid DFT ALBs
/// @date 2023-11-05 Add HFX hamiltonian matrix
/// @date 2024-01-29 Add different mixing schemes 

/// Junwei Feng fengjw@mail.ustc.edu.cn
//  @date 2024-07-01 Add DG-RT-TDDFT




// 
#include  "scf_dg.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"
#include  "utility.hpp"
#include  "environment.hpp"
#ifdef ELSI
#include  "elsi.h"
#endif

namespace  dgdft{

using namespace dgdft::DensityComponent;
using namespace dgdft::esdf;
using namespace dgdft::scalapack;

// FIXME Leave the smoother function to somewhere more appropriate
Real Smoother ( Real x )
{
  Real t, z;
  if( x <= 0 )
    t = 1.0;
  else if( x >= 1 )
    t = 0.0;
  else{
    z = -1.0 / x + 1.0 / (1.0 - x );
    if( z < 0 )
      t = 1.0 / ( std::exp(z) + 1.0 );
    else
      t = std::exp(-z) / ( std::exp(-z) + 1.0 );
  }
  return t;
}        // -----  end of function Smoother  ----- 

SCFDG::SCFDG    (  )
{
  isPEXSIInitialized_ = false;
}        // -----  end of method SCFDG::SCFDG  ----- 

SCFDG::~SCFDG    (  )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );
#ifdef _USE_PEXSI_
  if( isPEXSIInitialized_ == true ){
    Int info;
    PPEXSIPlanFinalize(plan_, &info);
    if( info != 0 ){
      std::ostringstream msg;
      msg 
        << "PEXSI finalization returns info " << info << std::endl;
      ErrorHandling( msg.str().c_str() );
    }

    MPI_Comm_free( &pexsiComm_ );
  }
#endif
}         // -----  end of method SCFDG::~SCFDG  ----- 

void
SCFDG::Setup    ( 
    HamiltonianDG&              hamDG,
    DistVec<Index3, EigenSolver, ElemPrtn>&  distEigSol,
    DistFourier&                distfft,
    PeriodTable&                ptable,
    Int                         contxt    )
{
  Real timeSta, timeEnd;
 Begin_DG_TDDFT_=0;
  // *********************************************************************
  // Read parameters from ESDFParam
  // *********************************************************************
  // Control parameters
  {
    domain_             = esdfParam.domain;
    mixMaxDim_          = esdfParam.mixMaxDim;
    OutermixVariable_   = esdfParam.OutermixVariable;
    InnermixVariable_   = esdfParam.InnermixVariable;
    HybridInnermixVariable_   = esdfParam.HybridInnermixVariable;
    mixType_            = esdfParam.mixType;
    HybridmixType_      = esdfParam.DGHybridmixType;
    mixStepLength_      = esdfParam.mixStepLength;
    eigMinTolerance_    = esdfParam.eigMinTolerance;
    eigTolerance_       = esdfParam.eigTolerance;
    eigMinIter_         = esdfParam.eigMinIter;
    eigMaxIter_         = esdfParam.eigMaxIter;
    DFTscfInnerTolerance_  = esdfParam.DFTscfInnerTolerance;
    DFTscfInnerMinIter_    = esdfParam.DFTscfInnerMinIter;
    DFTscfInnerMaxIter_    = esdfParam.DFTscfInnerMaxIter;
    DFTscfOuterTolerance_  = esdfParam.DFTscfOuterTolerance;
    DFTscfOuterMinIter_    = esdfParam.DFTscfOuterMinIter;
    DFTscfOuterMaxIter_    = esdfParam.DFTscfOuterMaxIter;
    DFTscfOuterEnergyTolerance_    = esdfParam.DFTscfOuterEnergyTolerance;
    HybridscfInnerTolerance_       = esdfParam.HybridscfInnerTolerance;
    HybridscfInnerMinIter_         = esdfParam.HybridscfInnerMinIter;
    HybridscfInnerMaxIter_         = esdfParam.HybridscfInnerMaxIter;
    HybridscfOuterTolerance_       = esdfParam.HybridscfOuterTolerance;
    HybridscfOuterMinIter_         = esdfParam.HybridscfOuterMinIter;
    HybridscfOuterMaxIter_         = esdfParam.HybridscfOuterMaxIter;
    HybridscfOuterEnergyTolerance_ = esdfParam.HybridscfOuterEnergyTolerance;

    Ehfx_               = 0.0; 

    PhiMaxIter_         = esdfParam.HybridPhiMaxIter;
    PhiTolerance_       = esdfParam.HybridPhiTolerance;

    numUnusedState_     = esdfParam.numUnusedState;
    SVDBasisTolerance_  = esdfParam.SVDBasisTolerance;
    solutionMethod_     = esdfParam.solutionMethod;
    diagSolutionMethod_ = esdfParam.diagSolutionMethod;

    // Choice of smearing scheme : Fermi-Dirac (FD) or Gaussian_Broadening (GB) or Methfessel-Paxton (MP)
    // Currently PEXSI only supports FD smearing, so GB or MP have to be used with diag type methods
    SmearingScheme_ = esdfParam.smearing_scheme;
    if(solutionMethod_ == "pexsi")
      SmearingScheme_ = "FD";

    if(SmearingScheme_ == "GB")
      MP_smearing_order_ = 0;
    else if(SmearingScheme_ == "MP")
      MP_smearing_order_ = 2;
    else
      MP_smearing_order_ = -1; // For safety

    PWSolver_           = esdfParam.PWSolver;

    // Chebyshev Filtering related parameters for PWDFT on extended element
    if(PWSolver_ == "CheFSI")
      Diag_SCF_PWDFT_by_Cheby_ = 1;
    else
      Diag_SCF_PWDFT_by_Cheby_ = 0;

    First_SCF_PWDFT_ChebyFilterOrder_ = esdfParam.First_SCF_PWDFT_ChebyFilterOrder;
    First_SCF_PWDFT_ChebyCycleNum_ = esdfParam.First_SCF_PWDFT_ChebyCycleNum;
    General_SCF_PWDFT_ChebyFilterOrder_ = esdfParam.General_SCF_PWDFT_ChebyFilterOrder;
    PWDFT_Cheby_use_scala_ = esdfParam.PWDFT_Cheby_use_scala;
    PWDFT_Cheby_apply_wfn_ecut_filt_ =  esdfParam.PWDFT_Cheby_apply_wfn_ecut_filt;

    // Using PPCG for PWDFT on extended element
    if(PWSolver_ == "PPCG" || PWSolver_ == "PPCGScaLAPACK")
      Diag_SCF_PWDFT_by_PPCG_ = 1;
    else
      Diag_SCF_PWDFT_by_PPCG_ = 0;

    Tbeta_            = esdfParam.Tbeta;
    Tsigma_           = 1.0 / Tbeta_;
    scaBlockSize_     = esdfParam.scaBlockSize;
    numElem_          = esdfParam.numElem;
    ecutWavefunction_ = esdfParam.ecutWavefunction;
    densityGridFactor_= esdfParam.densityGridFactor;
    LGLGridFactor_    = esdfParam.LGLGridFactor;
    distancePeriodize_= esdfParam.distancePeriodize;

    potentialBarrierW_  = esdfParam.potentialBarrierW;
    potentialBarrierS_  = esdfParam.potentialBarrierS;
    potentialBarrierR_  = esdfParam.potentialBarrierR;

    XCType_             = esdfParam.XCType;
    VDWType_            = esdfParam.VDWType;
  }

  // Variables related to Chebyshev Filtered SCF iterations for DG  
  {
    Diag_SCFDG_by_Cheby_ = esdfParam.Diag_SCFDG_by_Cheby; // Default: 0
    SCFDG_Cheby_use_ScaLAPACK_ = esdfParam.SCFDG_Cheby_use_ScaLAPACK; // Default: 0

    First_SCFDG_ChebyFilterOrder_ = esdfParam.First_SCFDG_ChebyFilterOrder; // Default 60
    First_SCFDG_ChebyCycleNum_ = esdfParam.First_SCFDG_ChebyCycleNum; // Default 5

    Second_SCFDG_ChebyOuterIter_ = esdfParam.Second_SCFDG_ChebyOuterIter; // Default = 3
    Second_SCFDG_ChebyFilterOrder_ = esdfParam.Second_SCFDG_ChebyFilterOrder; // Default = 60
    Second_SCFDG_ChebyCycleNum_ = esdfParam.Second_SCFDG_ChebyCycleNum; // Default 3 

    General_SCFDG_ChebyFilterOrder_ = esdfParam.General_SCFDG_ChebyFilterOrder; // Default = 60
    General_SCFDG_ChebyCycleNum_ = esdfParam.General_SCFDG_ChebyCycleNum; // Default 1

    Cheby_iondynamics_schedule_flag_ = 0;
    scfdg_ion_dyn_iter_ = 0;
  }

  // Variables related to Chebyshev polynomial filtered 
  // complementary subspace iteration strategy in DGDFT
  // Only accessed if CheFSI is in use 

  if(Diag_SCFDG_by_Cheby_ == 1)
  {
    SCFDG_use_comp_subspace_ = esdfParam.scfdg_use_chefsi_complementary_subspace;  // Default: 0

    SCFDG_comp_subspace_parallel_ = SCFDG_Cheby_use_ScaLAPACK_; // Use serial or parallel routine depending on early CheFSI steps

    // Syrk and Syr2k based updates, available in parallel routine only
    SCFDG_comp_subspace_syrk_ = esdfParam.scfdg_chefsi_complementary_subspace_syrk; 
    SCFDG_comp_subspace_syr2k_ = esdfParam.scfdg_chefsi_complementary_subspace_syr2k;

    // Safeguard to ensure that CS strategy is called only after atleast one general CheFSI cycle has been called
    // This allows the initial guess vectors to be copied
    if(  SCFDG_use_comp_subspace_ == 1 && Second_SCFDG_ChebyOuterIter_ < 2)
      Second_SCFDG_ChebyOuterIter_ = 2;

    SCFDG_comp_subspace_nstates_ = esdfParam.scfdg_complementary_subspace_nstates; // Defaults to a fraction of extra states

    SCFDG_CS_ioniter_regular_cheby_freq_ = esdfParam.scfdg_cs_ioniter_regular_cheby_freq; // Defaults to 20

    SCFDG_CS_bigger_grid_dim_fac_ = esdfParam.scfdg_cs_bigger_grid_dim_fac; // Defaults to 1;

    // LOBPCG for top states option
    SCFDG_comp_subspace_LOBPCG_iter_ = esdfParam.scfdg_complementary_subspace_lobpcg_iter; // Default = 15
    SCFDG_comp_subspace_LOBPCG_tol_ = esdfParam.scfdg_complementary_subspace_lobpcg_tol; // Default = 1e-8

    // CheFSI for top states option
    Hmat_top_states_use_Cheby_ = esdfParam.Hmat_top_states_use_Cheby;
    Hmat_top_states_ChebyFilterOrder_ = esdfParam.Hmat_top_states_ChebyFilterOrder; 
    Hmat_top_states_ChebyCycleNum_ = esdfParam.Hmat_top_states_ChebyCycleNum; 
    Hmat_top_states_Cheby_delta_fudge_ = 0.0;

    // Extra precaution : Inner LOBPCG only available in serial mode and syrk type updates only available in paralle mode
    if(SCFDG_comp_subspace_parallel_ == 1){
      Hmat_top_states_use_Cheby_ = 1; 
    }
    else
    { 
      SCFDG_comp_subspace_syrk_ = 0;
      SCFDG_comp_subspace_syr2k_ = 0;
    }

    SCFDG_comp_subspace_N_solve_ = hamDG.NumExtraState() + SCFDG_comp_subspace_nstates_;     
    SCFDG_comp_subspace_engaged_ = false;
  }
  else
  {
    SCFDG_use_comp_subspace_ = false;
    SCFDG_comp_subspace_engaged_ = false;
  }

  // Ionic iteration related parameters
  scfdg_ion_dyn_iter_ = 0; // Ionic iteration number
  useEnergySCFconvergence_ = 0; // Whether to use energy based SCF convergence
  md_scf_etot_diff_tol_ = esdfParam.MDscfEtotdiff; // Tolerance for SCF total energy for energy based SCF convergence
  md_scf_eband_diff_tol_ = esdfParam.MDscfEbanddiff; // Tolerance for SCF band energy for energy based SCF convergence

  md_scf_etot_ = 0.0;
  md_scf_etot_old_ = 0.0;
  md_scf_etot_diff_ = 0.0;
  md_scf_eband_ = 0.0;
  md_scf_eband_old_ = 0.0; 
  md_scf_eband_diff_ = 0.0;

  Int mpirank; MPI_Comm_rank( domain_.comm, &mpirank );
  Int mpisize; MPI_Comm_size( domain_.comm, &mpisize );
  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  dmCol_ = numElem_[0] * numElem_[1] * numElem_[2];
  dmRow_ = mpisize / dmCol_;

  numProcScaLAPACK_ = esdfParam.numProcScaLAPACK;

  // Initialize PEXSI
#ifdef _USE_PEXSI_
  if( solutionMethod_ == "pexsi" )
  {
    Int info;
    // Initialize the PEXSI options
    PPEXSISetDefaultOptions( &pexsiOptions_ );

    pexsiOptions_.temperature      = 1.0 / Tbeta_;
    pexsiOptions_.gap              = esdfParam.energyGap;
    pexsiOptions_.deltaE           = esdfParam.spectralRadius;
    pexsiOptions_.numPole          = esdfParam.numPole;
    pexsiOptions_.isInertiaCount   = 1; 
    pexsiOptions_.maxPEXSIIter     = esdfParam.maxPEXSIIter;
    pexsiOptions_.muMin0           = esdfParam.muMin;
    pexsiOptions_.muMax0           = esdfParam.muMax;
    pexsiOptions_.muInertiaTolerance = 
      esdfParam.muInertiaTolerance;
    pexsiOptions_.muInertiaExpansion = 
      esdfParam.muInertiaExpansion;
    pexsiOptions_.muPEXSISafeGuard   = 
      esdfParam.muPEXSISafeGuard;
    pexsiOptions_.numElectronPEXSITolerance = 
      esdfParam.numElectronPEXSITolerance;

    muInertiaToleranceTarget_ = esdfParam.muInertiaTolerance;
    numElectronPEXSIToleranceTarget_ = esdfParam.numElectronPEXSITolerance;

    pexsiOptions_.ordering           = esdfParam.matrixOrdering;
    pexsiOptions_.npSymbFact         = esdfParam.npSymbFact;
    pexsiOptions_.verbosity          = 1; // FIXME

    numProcRowPEXSI_     = esdfParam.numProcRowPEXSI;
    numProcColPEXSI_     = esdfParam.numProcColPEXSI;
    inertiaCountSteps_   = esdfParam.inertiaCountSteps;

    // Provide a communicator for PEXSI
    numProcPEXSICommCol_ = numProcRowPEXSI_ * numProcColPEXSI_;

    if( numProcPEXSICommCol_ > dmCol_ ){
      std::ostringstream msg;
      msg 
        << "In the current implementation, "
        << "the number of processors per pole = " << numProcPEXSICommCol_ 
        << ", and cannot exceed the number of elements = " << dmCol_ 
        << std::endl;
      ErrorHandling( msg.str().c_str() );
    }

    Int numProcPEXSICommRow_ = mpisize / numProcPEXSICommCol_;
    Int mpisizePEXSI = numProcPEXSICommRow_ * numProcPEXSICommCol_;
    numProcTotalPEXSI_ = numProcPEXSICommRow_ * numProcPEXSICommCol_;
    Int mpirank_tanspose = mpirankCol + mpirankRow * mpisizeCol;
    Int outputFileIndex;

    if(mpirank_tanspose == 0 ){
      outputFileIndex = 0;		  
    }
    else{
      outputFileIndex = -1;
    }

    Int isProcPEXSI = 1;
    Int mpirankPEXSI = mpirank_tanspose;

#if ( _DEBUGlevel_ >= 2 )
    statusOFS 
      << "mpirank = " << mpirank << std::endl
      << "mpisize = " << mpisize << std::endl
      << "mpirankRow = " << mpirankRow << std::endl
      << "mpirankCol = " << mpirankCol << std::endl
      << "mpisizeRow = " << mpisizeRow << std::endl
      << "mpisizeCol = " << mpisizeCol << std::endl
      << "outputFileIndex = " << outputFileIndex << std::endl
      << "mpirankPEXSI = " << mpirankPEXSI << std::endl
      << "numProcPEXSICommRow_ = " << numProcPEXSICommRow_ << std::endl
      << "numProcPEXSICommCol_ = " << numProcPEXSICommCol_ << std::endl
      << "isProcPEXSI = " << isProcPEXSI << std::endl
      << "mpisizePEXSI = " << numProcTotalPEXSI_ << std::endl
      << std::endl;
#endif
    MPI_Comm_split( domain_.comm, isProcPEXSI, mpirankPEXSI, &planComm_);

    plan_ = 
       PPEXSIPlanInitialize(
       planComm_,
       numProcRowPEXSI_,
       numProcColPEXSI_,
       outputFileIndex,
       &info );

    if( info != 0 ){
        std::ostringstream msg;
        msg 
          << "PEXSI initialization returns info " << info << std::endl;
        ErrorHandling( msg.str().c_str() );
    }
  }
#endif // _USE_PEXSI_

  // other SCFDG parameters
  {
    hamDGPtr_      = &hamDG;
    distEigSolPtr_ = &distEigSol;
    distfftPtr_    = &distfft;
    ptablePtr_     = &ptable;
    elemPrtn_      = distEigSol.Prtn();
    contxt_        = contxt;

    vtotLGLSave_.SetComm(domain_.colComm);
    vtotLGLSave_.Prtn()   = elemPrtn_;

    if( hamDG.IsHybrid() ){
      distDMMat_.SetComm(domain_.colComm);  
      distEDMMat_.SetComm(domain_.colComm);
      distFDMMat_.SetComm(domain_.colComm); 
      distDMMatSave_.SetComm(domain_.colComm);
      distDMMat_.Prtn()     = hamDG.HMat().Prtn();
      distDMMatSave_.Prtn()     = hamDG.HMat().Prtn();
    }

    if( InnermixVariable_ == "potential" || InnermixVariable_ == "density" ){
      mixOuterSave_.SetComm(domain_.colComm);
      mixInnerSave_.SetComm(domain_.colComm);
      dfOuterMat_.SetComm(domain_.colComm);
      dvOuterMat_.SetComm(domain_.colComm);
      dfInnerMat_.SetComm(domain_.colComm);
      dvInnerMat_.SetComm(domain_.colComm);
      cdfInnerMat_.SetComm(domain_.colComm);

      mixOuterSave_.Prtn()  = elemPrtn_;
      mixInnerSave_.Prtn()  = elemPrtn_;
      dfOuterMat_.Prtn()    = elemPrtn_;
      dvOuterMat_.Prtn()    = elemPrtn_;
      dfInnerMat_.Prtn()    = elemPrtn_;
      dvInnerMat_.Prtn()    = elemPrtn_;
      cdfInnerMat_.Prtn()   = elemPrtn_;
    }

#ifdef _USE_PEXSI_
    distDMMat_.Prtn()      = hamDG.HMat().Prtn();
    distEDMMat_.Prtn()     = hamDG.HMat().Prtn();
    distFDMMat_.Prtn()     = hamDG.HMat().Prtn(); 
#endif
    if(SCFDG_use_comp_subspace_ == 1)
      distDMMat_.Prtn() = hamDG.HMat().Prtn();

    // The number of processors in the column communicator must be the
    // number of elements, and mpisize should be a multiple of the
    // number of elements.
    if( (mpisize % dmCol_) != 0 ){
      statusOFS << "mpisize = " << mpisize << " mpirank = " << mpirank << std::endl;
      statusOFS << "dmCol_ = " << dmCol_ << " dmRow_ = " << dmRow_ << std::endl;
      std::ostringstream msg;
      msg << "Total number of processors do not fit to the number processors per element." << std::endl;
      ErrorHandling( msg.str().c_str() );
    }

    // FIXME fixed ratio between the size of the extended element and
    // the element
    for( Int d = 0; d < DIM; d++ ){
      extElemRatio_[d] = ( numElem_[d]>1 ) ? 3 : 1;
    }

    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumVec emptyLGLVec( hamDG.NumLGLGridElem().prod() );
            SetValue( emptyLGLVec, 0.0 );
            vtotLGLSave_.LocalMap()[key] = emptyLGLVec;
          }
    } // for (i)

    if( InnermixVariable_ == "potential" ||  InnermixVariable_ == "density"){
      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              DblNumVec  emptyVec( hamDG.NumUniformGridElemFine().prod() );
              SetValue( emptyVec, 0.0 );
              mixOuterSave_.LocalMap()[key] = emptyVec;
              mixInnerSave_.LocalMap()[key] = emptyVec;
              DblNumMat  emptyMat( hamDG.NumUniformGridElemFine().prod(), mixMaxDim_ );
              SetValue( emptyMat, 0.0 );
              DblNumMat  emptyMat2( hamDG.NumUniformGridElemFine().prod(), 2 );
              SetValue( emptyMat2, 0.0 );
              dfOuterMat_.LocalMap()[key]   = emptyMat;
              dvOuterMat_.LocalMap()[key]   = emptyMat;
              dfInnerMat_.LocalMap()[key]   = emptyMat;
              dvInnerMat_.LocalMap()[key]   = emptyMat;
              if( mixType_ == "broyden" ){
                cdfInnerMat_.LocalMap()[key] = emptyMat2;
              }
            } // own this element
      }  // for (i)
    } // mixVariable

    // Restart the density in the global domain
    restartDensityFileName_ = "DEN";
    // Restart the wavefunctions in the extended element
    restartWfnFileName_     = "WFNEXT";
  }

  // *********************************************************************
  // Initialization
  // *********************************************************************

  // Density
  DistDblNumVec&  density = hamDGPtr_->Density();

  density.SetComm(domain_.colComm);

  if( esdfParam.isRestartDensity ) {
    // Only the first processor column reads the matrix

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Restarting density from DEN_ files." << std::endl;
#endif

    if( mpirankRow == 0 ){
      std::istringstream rhoStream;      
      SeparateRead( restartDensityFileName_, rhoStream, mpirankCol );

      Real sumDensityLocal = 0.0, sumDensity = 0.0;

      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              std::vector<DblNumVec> gridpos(DIM);

              // Dummy variables and not used
              for( Int d = 0; d < DIM; d++ ){
                deserialize( gridpos[d], rhoStream, NO_MASK );
              }

              Index3 keyRead;
              deserialize( keyRead, rhoStream, NO_MASK );
              if( keyRead[0] != key[0] ||
                  keyRead[1] != key[1] ||
                  keyRead[2] != key[2] ){
                std::ostringstream msg;
                msg 
                  << "Mpirank " << mpirank << " is reading the wrong file."
                  << std::endl
                  << "key     ~ " << key << std::endl
                  << "keyRead ~ " << keyRead << std::endl;
                ErrorHandling( msg.str().c_str() );
              }

              DblNumVec   denVecRead;
              DblNumVec&  denVec = density.LocalMap()[key];
              deserialize( denVecRead, rhoStream, NO_MASK );
              if( denVecRead.Size() != denVec.Size() ){
                std::ostringstream msg;
                msg 
                  << "The size of restarting density does not match with the current setup."  
                  << std::endl
                  << "input density size   ~ " << denVecRead.Size() << std::endl
                  << "current density size ~ " << denVec.Size()     << std::endl;
                ErrorHandling( msg.str().c_str() );
              }
              denVec = denVecRead;
              for( Int p = 0; p < denVec.Size(); p++ ){
                sumDensityLocal += denVec(p);
              }
            }
      } // for (i)

      // Rescale the density
      mpi::Allreduce( &sumDensityLocal, &sumDensity, 1, MPI_SUM,
          domain_.colComm );

      Print( statusOFS, "Restart density. Sum of density      = ", 
          sumDensity * domain_.Volume() / domain_.NumGridTotalFine() );
    } // mpirank == 0

    // Broadcast the density to the column
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumVec&  denVec = density.LocalMap()[key];
            MPI_Bcast( denVec.Data(), denVec.Size(), MPI_DOUBLE, 0, domain_.rowComm );
          }
    }

  } // else using the zero initial guess
  else {
    if( esdfParam.isUseAtomDensity ){
      statusOFS << "Use superposition of atomic density as initial "
        << "guess for electron density." << std::endl;

      GetTime( timeSta );
      hamDGPtr_->CalculateAtomDensity( *ptablePtr_, *distfftPtr_ );
      GetTime( timeEnd );
      statusOFS << "Time for calculating the atomic density = " 
        << timeEnd - timeSta << " [s]" << std::endl;

      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              DblNumVec&  denVec = density.LocalMap()[key];
              DblNumVec&  atomdenVec  = hamDGPtr_->AtomDensity().LocalMap()[key];
              blas::Copy( denVec.Size(), atomdenVec.Data(), 1, denVec.Data(), 1 );
            }
          } // for (i)
    }
    else{
      statusOFS << "Generating initial density through linear combination of pseudocharges." 
        << std::endl;
      // Initialize the electron density using the pseudocharge
      // make sure the pseudocharge is initialized
      DistDblNumVec& pseudoCharge = hamDGPtr_->PseudoCharge();

      pseudoCharge.SetComm(domain_.colComm);

      Real sumDensityLocal = 0.0, sumPseudoChargeLocal = 0.0;
      Real sumDensity, sumPseudoCharge;
      Real EPS = 1e-14;

      // make sure that the electron density is positive
      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              DblNumVec&  denVec = density.LocalMap()[key];
              DblNumVec&  ppVec  = pseudoCharge.LocalMap()[key];
              for( Int p = 0; p < denVec.Size(); p++ ){
                //                            denVec(p) = ppVec(p);
                denVec(p) = ( ppVec(p) > EPS ) ? ppVec(p) : EPS;
                sumDensityLocal += denVec(p);
                sumPseudoChargeLocal += ppVec(p);
              }
            }
      } // for (i)

      // Rescale the density
      mpi::Allreduce( &sumDensityLocal, &sumDensity, 1, MPI_SUM,
          domain_.colComm );
      mpi::Allreduce( &sumPseudoChargeLocal, &sumPseudoCharge, 
          1, MPI_SUM, domain_.colComm );

      Print( statusOFS, "Initial density. Sum of density      = ", 
          sumDensity * domain_.Volume() / domain_.NumGridTotalFine() );
#if ( _DEBUGlevel_ >= 1 )
      Print( statusOFS, "Sum of pseudo charge        = ", 
          sumPseudoCharge * domain_.Volume() / domain_.NumGridTotalFine() );
#endif
      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              DblNumVec&  denVec = density.LocalMap()[key];
              blas::Scal( denVec.Size(), sumPseudoCharge / sumDensity, 
                  denVec.Data(), 1 );
            }
      } // for (i)
    }  // esdfParam.isUseAtomDensity
  } // Restart the density

  // Wavefunctions in the extended element
  if( esdfParam.isRestartWfn ){
    statusOFS << "Restarting basis functions from WFNEXT_ files"
      << std::endl;
    std::istringstream wfnStream;      
    SeparateRead( restartWfnFileName_, wfnStream, mpirank );

    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
            Spinor& psi = eigSol.Psi();
            DblNumTns& wavefun = psi.Wavefun();
            DblNumTns  wavefunRead;
            std::vector<DblNumVec> gridpos(DIM);
            for( Int d = 0; d < DIM; d++ ){
              deserialize( gridpos[d], wfnStream, NO_MASK );
            }

            Index3 keyRead;
            deserialize( keyRead, wfnStream, NO_MASK );
            if( keyRead[0] != key[0] ||
                keyRead[1] != key[1] ||
                keyRead[2] != key[2] ){
              std::ostringstream msg;
              msg 
                << "Mpirank " << mpirank << " is reading the wrong file."
                << std::endl
                << "key     ~ " << key << std::endl
                << "keyRead ~ " << keyRead << std::endl;
              ErrorHandling( msg.str().c_str() );
            }
            deserialize( wavefunRead, wfnStream, NO_MASK );

            if( wavefunRead.Size() != wavefun.Size() ){
              std::ostringstream msg;
              msg 
                << "The size of restarting basis function does not match with the current setup."  
                << std::endl
                << "input basis size   ~ " << wavefunRead.Size() << std::endl
                << "current basis size ~ " << wavefun.Size()     << std::endl;
              ErrorHandling( msg.str().c_str() );
            }

            wavefun = wavefunRead;
          }
    } // for (i)

  }  //esdfParam.isRestartWfn 
  else{ 
    statusOFS << "Initial random basis functions in the extended element."
      << std::endl;

    // Use random initial guess for basis functions in the extended element.
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
            Spinor& psi = eigSol.Psi();

         // Wavefuns have been set in dgdft init, why reInit them in extended element here ?  
         // FIXME   xmqin
         //   UniformRandom( psi.Wavefun() );
            // For debugging purpose
            // Make sure that the initial wavefunctions in each element
            // are the same, when different number of processors are
            // used for intra-element parallelization.
            { 
              Spinor  psiTemp;
              psiTemp.Setup( eigSol.FFT().domain, 1, psi.NumStateTotal(), psi.NumStateTotal(), 0.0 );

              Int mpirankp, mpisizep;
              MPI_Comm_rank( domain_.rowComm, &mpirankp );
              MPI_Comm_size( domain_.rowComm, &mpisizep );

              if (mpirankp == 0){
                SetRandomSeed(1);
                UniformRandom( psiTemp.Wavefun() );
              }
              MPI_Bcast(psiTemp.Wavefun().Data(), psiTemp.Wavefun().m()*psiTemp.Wavefun().n()*psiTemp.Wavefun().p(), MPI_DOUBLE, 0, domain_.rowComm);

              Int size = psi.Wavefun().m() * psi.Wavefun().n();
              Int nocc = psi.Wavefun().p();

              IntNumVec& wavefunIdx = psi.WavefunIdx();
              NumTns<Real>& wavefun = psi.Wavefun();
             
              for (Int k=0; k<nocc; k++) {
                Real *ptr = psi.Wavefun().MatData(k);
                Real *ptr1 = psiTemp.Wavefun().MatData(wavefunIdx(k));
                for (Int i=0; i<size; i++) {
                  *ptr = *ptr1;
                  ptr = ptr + 1;
                  ptr1 = ptr1 + 1;
                }
              }
            } // random

           // xmqin add for occ HSE  //Check
            DblNumVec& occ = eigSol.Ham().OccupationRate();
            Int npsi = psi.NumStateTotal();
            Int nocc = eigSol.Ham().NumOccupiedState();

            if(nocc <= npsi) {
              occ.Resize( npsi );
              SetValue( occ, 0.0 );
              for( Int k = 0; k < nocc; k++ ){
                occ[k] = 1.0;
              }
            }
            else {
              std::ostringstream msg;
              msg
              << "number of ALBs is " << npsi << " is less than number of occupied orbitals " << nocc << " in this extended element "  
              << std::endl;
              ErrorHandling( msg.str().c_str() );
            }
          }
        } // for (i)
    Print( statusOFS, "Initial basis functions with random guess." );
  } // if (isRestartWfn_)

  // Generate the transfer matrix from the periodic uniform grid on each
  // extended element to LGL grid.  
  // 05/06/2015:
  // Based on the new understanding of the dual grid treatment, the
  // interpolation must be performed through a fine Fourier grid
  // (uniform grid) and then interpolate to the LGL grid.
  {
    PeriodicUniformToLGLMat_.resize(DIM);
    PeriodicUniformFineToLGLMat_.resize(DIM);
    PeriodicGridExtElemToGridElemMat_.resize(DIM);

    EigenSolver& eigSol = (*distEigSol.LocalMap().begin()).second;
    Domain dmExtElem = eigSol.FFT().domain;
    Domain dmElem;
    for( Int d = 0; d < DIM; d++ ){
      dmElem.length[d]   = domain_.length[d] / numElem_[d];
      dmElem.numGrid[d]  = domain_.numGrid[d] / numElem_[d];
      dmElem.numGridFine[d]  = domain_.numGridFine[d] / numElem_[d];
      // PosStart relative to the extended element 
      dmExtElem.posStart[d] = 0.0;
      dmElem.posStart[d] = ( numElem_[d] > 1 ) ? dmElem.length[d] : 0;
    }

    Index3 numLGL        = hamDG.NumLGLGridElem();
    Index3 numUniform    = dmExtElem.numGrid;
    Index3 numUniformFine    = dmExtElem.numGridFine;
    Index3 numUniformFineElem    = dmElem.numGridFine;
    Point3 lengthUniform = dmExtElem.length;

    std::vector<DblNumVec>  LGLGrid(DIM);
    LGLMesh( dmElem, numLGL, LGLGrid ); 
    std::vector<DblNumVec>  UniformGrid(DIM);
    UniformMesh( dmExtElem, UniformGrid );
    std::vector<DblNumVec>  UniformGridFine(DIM);
    UniformMeshFine( dmExtElem, UniformGridFine );
    std::vector<DblNumVec>  UniformGridFineElem(DIM);
    UniformMeshFine( dmElem, UniformGridFineElem );

    for( Int d = 0; d < DIM; d++ ){
      DblNumMat&  localMat = PeriodicUniformToLGLMat_[d];
      DblNumMat&  localMatFineElem = PeriodicGridExtElemToGridElemMat_[d];
      localMat.Resize( numLGL[d], numUniform[d] );
      localMatFineElem.Resize( numUniformFineElem[d], numUniform[d] );
      SetValue( localMat, 0.0 );
      SetValue( localMatFineElem, 0.0 );
      DblNumVec KGrid( numUniform[d] );
      for( Int i = 0; i <= numUniform[d] / 2; i++ ){
        KGrid(i) = i * 2.0 * PI / lengthUniform[d];
      }
      for( Int i = numUniform[d] / 2 + 1; i < numUniform[d]; i++ ){
        KGrid(i) = ( i - numUniform[d] ) * 2.0 * PI / lengthUniform[d];
      }

      for( Int j = 0; j < numUniform[d]; j++ ){

        for( Int i = 0; i < numLGL[d]; i++ ){
          localMat(i, j) = 0.0;
          for( Int k = 0; k < numUniform[d]; k++ ){
            localMat(i,j) += std::cos( KGrid(k) * ( LGLGrid[d](i) -
                  UniformGrid[d](j) ) ) / numUniform[d];
          } // for (k)
        } // for (i)

        for( Int i = 0; i < numUniformFineElem[d]; i++ ){
          localMatFineElem(i, j) = 0.0;
          for( Int k = 0; k < numUniform[d]; k++ ){
            localMatFineElem(i,j) += std::cos( KGrid(k) * ( UniformGridFineElem[d](i) -
                  UniformGrid[d](j) ) ) / numUniform[d];
          } // for (k)
        } // for (i)
      } // for (j)
    } // for (d)


    for( Int d = 0; d < DIM; d++ ){
      DblNumMat&  localMatFine = PeriodicUniformFineToLGLMat_[d];
      localMatFine.Resize( numLGL[d], numUniformFine[d] );
      SetValue( localMatFine, 0.0 );
      DblNumVec KGridFine( numUniformFine[d] );
      for( Int i = 0; i <= numUniformFine[d] / 2; i++ ){
        KGridFine(i) = i * 2.0 * PI / lengthUniform[d];
      }
      for( Int i = numUniformFine[d] / 2 + 1; i < numUniformFine[d]; i++ ){
        KGridFine(i) = ( i - numUniformFine[d] ) * 2.0 * PI / lengthUniform[d];
      }

      for( Int j = 0; j < numUniformFine[d]; j++ ){

        for( Int i = 0; i < numLGL[d]; i++ ){
          localMatFine(i, j) = 0.0;
          for( Int k = 0; k < numUniformFine[d]; k++ ){
            localMatFine(i,j) += std::cos( KGridFine(k) * ( LGLGrid[d](i) -
                  UniformGridFine[d](j) ) ) / numUniformFine[d];
          } // for (k)
        } // for (i)

      } // for (j)
    } // for (d)

    // Assume the initial error is O(1)
    scfOuterNorm_ = 1.0;
    scfInnerNorm_ = 1.0;
    scfInnerMaxDif_ = 1.0;

#if ( _DEBUGlevel_ >= 2 )
    statusOFS << "PeriodicUniformToLGLMat[0] = "
      << PeriodicUniformToLGLMat_[0] << std::endl;
    statusOFS << "PeriodicUniformToLGLMat[1] = " 
      << PeriodicUniformToLGLMat_[1] << std::endl;
    statusOFS << "PeriodicUniformToLGLMat[2] = "
      << PeriodicUniformToLGLMat_[2] << std::endl;
    statusOFS << "PeriodicUniformFineToLGLMat[0] = "
      << PeriodicUniformFineToLGLMat_[0] << std::endl;
    statusOFS << "PeriodicUniformFineToLGLMat[1] = " 
      << PeriodicUniformFineToLGLMat_[1] << std::endl;
    statusOFS << "PeriodicUniformFineToLGLMat[2] = "
      << PeriodicUniformFineToLGLMat_[2] << std::endl;
    statusOFS << "PeriodicGridExtElemToGridElemMat[0] = "
      << PeriodicGridExtElemToGridElemMat_[0] << std::endl;
    statusOFS << "PeriodicGridExtElemToGridElemMat[1] = "
      << PeriodicGridExtElemToGridElemMat_[1] << std::endl;
    statusOFS << "PeriodicGridExtElemToGridElemMat[2] = "
      << PeriodicGridExtElemToGridElemMat_[2] << std::endl;
#endif
  }

  // Whether to apply potential barrier in the extended element. CANNOT
  // be used together with periodization option
  if( esdfParam.isPotentialBarrier ) {
    vBarrier_.resize(DIM);
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            Domain& dmExtElem = distEigSolPtr_->LocalMap()[key].FFT().domain;
            std::vector<DblNumVec> gridpos(DIM);
            UniformMeshFine ( dmExtElem, gridpos );

            for( Int d = 0; d < DIM; d++ ){
              Real length   = dmExtElem.length[d];
              Int numGridFine   = dmExtElem.numGridFine[d];
              Real posStart = dmExtElem.posStart[d]; 
              Real center   = posStart + length / 2.0;

              // FIXME
              Real EPS      = 1.0;           // For stability reason
              Real dist;

              vBarrier_[d].Resize( numGridFine );
              SetValue( vBarrier_[d], 0.0 );
              for( Int p = 0; p < numGridFine; p++ ){
                dist = std::abs( gridpos[d][p] - center );
                // Only apply the barrier for region outside barrierR
                if( dist > potentialBarrierR_){
                  vBarrier_[d][p] = potentialBarrierS_* std::exp( - potentialBarrierW_ / 
                      ( dist - potentialBarrierR_ ) ) / std::pow( dist - length / 2.0 - EPS, 2.0 );
                }
              }
            } // for (d)

#if ( _DEBUGlevel_ >= 2  )
            statusOFS << "gridpos[0] = " << std::endl << gridpos[0] << std::endl;
            statusOFS << "gridpos[1] = " << std::endl << gridpos[1] << std::endl;
            statusOFS << "gridpos[2] = " << std::endl << gridpos[2] << std::endl;
            statusOFS << "vBarrier[0] = " << std::endl << vBarrier_[0] << std::endl;
            statusOFS << "vBarrier[1] = " << std::endl << vBarrier_[1] << std::endl;
            statusOFS << "vBarrier[2] = " << std::endl << vBarrier_[2] << std::endl;
#endif
          } // own this element
    } // for (k)
  }

  // Whether to periodize the potential in the extended element. CANNOT
  // be used together with barrier option.
  if( esdfParam.isPeriodizePotential ){
    vBubble_.resize(DIM);
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            Domain& dmExtElem = distEigSolPtr_->LocalMap()[key].FFT().domain;
            std::vector<DblNumVec> gridpos(DIM);
            UniformMeshFine ( dmExtElem, gridpos );

            for( Int d = 0; d < DIM; d++ ){
              Real length   = dmExtElem.length[d];
              Int numGridFine   = dmExtElem.numGridFine[d];
              Real posStart = dmExtElem.posStart[d]; 
              // FIXME
              Real EPS = 0.2; // Criterion for distancePeriodize_
              vBubble_[d].Resize( numGridFine );
              SetValue( vBubble_[d], 1.0 );

              if( distancePeriodize_[d] > EPS ){
                Real lb = posStart + distancePeriodize_[d];
                Real rb = posStart + length - distancePeriodize_[d];
                for( Int p = 0; p < numGridFine; p++ ){
                  if( gridpos[d][p] > rb ){
                    vBubble_[d][p] = Smoother( (gridpos[d][p] - rb ) / 
                        (distancePeriodize_[d] - EPS) );
                  }

                  if( gridpos[d][p] < lb ){
                    vBubble_[d][p] = Smoother( (lb - gridpos[d][p] ) / 
                        (distancePeriodize_[d] - EPS) );
                  }
                }
              }
            } // for (d)

#if ( _DEBUGlevel_ >= 2  )
            statusOFS << "gridpos[0] = " << std::endl << gridpos[0] << std::endl;
            statusOFS << "gridpos[1] = " << std::endl << gridpos[1] << std::endl;
            statusOFS << "gridpos[2] = " << std::endl << gridpos[2] << std::endl;
            statusOFS << "vBubble[0] = " << std::endl << vBubble_[0] << std::endl;
            statusOFS << "vBubble[1] = " << std::endl << vBubble_[1] << std::endl;
            statusOFS << "vBubble[2] = " << std::endl << vBubble_[2] << std::endl;
#endif
          } // own this element
    } // for (k)
  }

  // Initial value
  efreeDifPerAtom_ = 100.0;

#ifdef ELSI
  // ELSI interface initilization for ELPA
  if((diagSolutionMethod_ == "elpa") && ( solutionMethod_ == "diag" ))
  {
    // Step 1. init the ELSI interface 
    Int Solver = 1;      // 1: ELPA, 2: LibSOMM 3: PEXSI for dense matrix, default to use ELPA
    Int parallelism = 1; // 1 for multi-MPIs 
    Int storage = 0;     // ELSI only support DENSE(0) 
    Int sizeH = hamDG.NumBasisTotal(); 
    Int n_states = hamDG.NumOccupiedState();

    Int n_electrons = 2.0* n_states;
    statusOFS << std::endl<<" Done Setting up ELSI iterface " 
              << Solver << " " << sizeH << " " << n_states
              << std::endl<<std::endl;

    c_elsi_init(Solver, parallelism, storage, sizeH, n_electrons, n_states);

    // Step 2.  setup MPI Domain
    MPI_Comm newComm;
    MPI_Comm_split(domain_.comm, contxt, mpirank, &newComm);
    Int comm = MPI_Comm_c2f(newComm);
    c_elsi_set_mpi(comm); 

    // step 3: setup blacs for elsi. 

    if(contxt >= 0)
      c_elsi_set_blacs(contxt, scaBlockSize_);   

    //  customize the ELSI interface to use identity matrix S
    c_elsi_customize(0, 1, 1.0E-8, 1, 0, 0); 

    // use ELPA 2 stage solver
    c_elsi_customize_elpa(2); 
  }

  if( solutionMethod_ == "pexsi" ){
    Int Solver = 3;      // 1: ELPA, 2: LibSOMM 3: PEXSI for dense matrix, default to use ELPA
    Int parallelism = 1; // 1 for multi-MPIs 
    Int storage = 1;     // PEXSI only support sparse(1) 
    Int sizeH = hamDG.NumBasisTotal(); 
    Int n_states = hamDG.NumOccupiedState();
    Int n_electrons = 2.0* n_states;

    statusOFS << std::endl<<" Done Setting up ELSI iterface " 
              << std::endl << " sizeH " << sizeH 
              << std::endl << " n_electron " << n_electrons
              << std::endl << " n_states "  << n_states
              << std::endl<<std::endl;

    c_elsi_init(Solver, parallelism, storage, sizeH, n_electrons, n_states);

    Int comm = MPI_Comm_c2f(pexsiComm_);
    c_elsi_set_mpi(comm); 

    c_elsi_customize(1, 1, 1.0E-8, 1, 0, 0); 
  }
#endif

  // Need density gradient for semilocal XC functionals,  xmqin
  {
    isCalculateGradRho_ = false;
    if( esdfParam.XCType == "XC_GGA_XC_PBE" ||
      esdfParam.XCType == "XC_HYB_GGA_XC_HSE06" ||
      esdfParam.XCType == "XC_HYB_GGA_XC_PBEH" ) {
      isCalculateGradRho_ = true;
    }
  }

  return ;
}         // -----  end of method SCFDG::Setup  ----- 

void
SCFDG::Update    ( )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  HamiltonianDG& hamDG = *hamDGPtr_;

  {
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumVec  emptyVec( hamDG.NumUniformGridElemFine().prod() );
            SetValue( emptyVec, 0.0 );
            mixOuterSave_.LocalMap()[key] = emptyVec;
            mixInnerSave_.LocalMap()[key] = emptyVec;
            DblNumMat  emptyMat( hamDG.NumUniformGridElemFine().prod(), mixMaxDim_ );
            SetValue( emptyMat, 0.0 );
            dfOuterMat_.LocalMap()[key]   = emptyMat;
            dvOuterMat_.LocalMap()[key]   = emptyMat;
            dfInnerMat_.LocalMap()[key]   = emptyMat;
            dvInnerMat_.LocalMap()[key]   = emptyMat;

            DblNumVec  emptyLGLVec( hamDG.NumLGLGridElem().prod() );
            SetValue( emptyLGLVec, 0.0 );
            vtotLGLSave_.LocalMap()[key] = emptyLGLVec;
          } // own this element
    }  // for (i)
  }

  return ;
}         // -----  end of method SCFDG::Update  ----- 

void
SCFDG::Iterate    (  )
{
  MPI_Barrier(domain_.comm);
  MPI_Barrier(domain_.colComm);
  MPI_Barrier(domain_.rowComm);

  Int mpirank; MPI_Comm_rank( domain_.comm, &mpirank );
  Int mpisize; MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);

  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  Real timeSta, timeEnd;

  Domain dmElem;
  for( Int d = 0; d < DIM; d++ ){
    dmElem.length[d]   = domain_.length[d] / numElem_[d];
    dmElem.numGrid[d]  = domain_.numGrid[d] / numElem_[d];
    dmElem.numGridFine[d]  = domain_.numGridFine[d] / numElem_[d];
    dmElem.posStart[d] = ( numElem_[d] > 1 ) ? dmElem.length[d] : 0;
  }

  HamiltonianDG&  hamDG = *hamDGPtr_;
  DistFourier&    fftDG = *distfftPtr_;

  if( !hamDG.IsHybrid() ) {
    statusOFS << "Regular DFT potential Calculations " << std::endl;
    if( isCalculateGradRho_ ) {
      GetTime( timeSta );
      hamDG.CalculateGradDensity( fftDG );
      GetTime( timeEnd );
      statusOFS << "Time for calculating gradient of density is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
    }

    GetTime( timeSta );
    hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), fftDG );
    GetTime( timeEnd );
    statusOFS << "Time for calculating XC is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
    // Compute the Hartree potential
    GetTime( timeSta );
    hamDG.CalculateHartree( hamDG.Vhart(), fftDG );
    GetTime( timeEnd );
    statusOFS << "Time for calculating Hartree is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
    // No external potential
    // Compute the total potential
    GetTime( timeSta );
    hamDG.CalculateVtot( hamDG.Vtot() );
    GetTime( timeEnd );
    statusOFS << "Time for calculating Vtot is " <<
      timeEnd - timeSta << " [s]" << std::endl;
  } 

  // xmqin add for HSE06
  //  FixMe:  First outer SCF for hybrids using PBE
  if( !hamDG.IsHybrid() || !hamDG.IsEXXActive() ){
    std::ostringstream msg;
    msg << "Starting regular DGDFT SCF iteration.";
    PrintBlock( statusOFS, msg.str() );
    bool isSCFConverged = false;

    if( !hamDG.IsEXXActive() && hamDG.IsHybrid() ) {
      hamDG.Setup_XC( "XC_GGA_XC_PBE");

      statusOFS << "Re-calculate PBE-XC for Hybrid DFT " << std::endl;
      if( isCalculateGradRho_ ) {
        GetTime( timeSta );
        hamDG.CalculateGradDensity( fftDG );
        GetTime( timeEnd );
        statusOFS << "Time for calculating gradient of density is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
      }

      GetTime( timeSta );
      hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), fftDG );
      GetTime( timeEnd );
      statusOFS << "Time for calculating XC is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
      // Compute the Hartree potential
      GetTime( timeSta );
      hamDG.CalculateHartree( hamDG.Vhart(), fftDG );
      GetTime( timeEnd );
      statusOFS << "Time for calculating Hartree is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
      // No external potential
      // Compute the total potential
      GetTime( timeSta );
      hamDG.CalculateVtot( hamDG.Vtot() );
      GetTime( timeEnd );
      statusOFS << "Time for calculating Vtot is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
    } // !ham.IsEXXActive() && ham.IsHybrid()

    Real timeIterStart(0), timeIterEnd(0);
    Real timeTotalStart(0), timeTotalEnd(0);

    scfTotalInnerIter_  = 0;

    GetTime( timeTotalStart );

    Int iter;

    // Total number of SVD basis functions. Determined at the first
    // outer SCF and is not changed later. This facilitates the reuse of
    // symbolic factorization

    for (iter=1; iter <= DFTscfOuterMaxIter_; iter++) {


if(iter==2)
{
isSCFConverged=true;
iter=10000;
}
      if ( isSCFConverged && (iter >= DFTscfOuterMinIter_ ) ) 

{
      
if(mpirank==0)
std::cout<<"RTTDDFT"<<std::endl;
RTTDDFT_RK4(5000,0.001);
break;

}

// *********************************************************************
      // Performing each iteartion
      // *********************************************************************
      {
        std::ostringstream msg;
        msg << "Outer SCF iteration # " << iter;
        PrintBlock( statusOFS, msg.str() );
      }

      GetTime( timeIterStart );

      // *********************************************************************
      // Update the local potential in the extended element and the element.
      //
      // NOTE: The modification of the potential on the extended element
      // to reduce the Gibbs phenomena is now in UpdateElemLocalPotential
      // *********************************************************************
      {
        GetTime(timeSta);

        UpdateElemLocalPotential();

        GetTime( timeEnd );
        statusOFS << "OuterSCF:: Time for updating the local potential in the extended element and the element is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
      }

      // *********************************************************************
      // Solve the basis functions in the extended element
      // *********************************************************************

      if(iter  > 1 && InnermixVariable_ == "densitymatrix"){
        for( Int k = 0; k < numElem_[2]; k++ )
          for( Int j = 0; j < numElem_[1]; j++ )
            for( Int i = 0; i < numElem_[0]; i++ ){
              Index3 key( i, j, k );
              if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){

                DblNumMat& basisOld = hamDG.BasisLGL().LocalMap()[key];
                DblNumMat& basisSave = hamDG.BasisLGLSave().LocalMap()[key];

                basisSave.Resize( basisOld.m(), basisOld.n() );

                SetValue( basisSave, 0.0 );

                blas::Copy( basisOld.m()*basisOld.n(), basisOld.Data(), 1, basisSave.Data(), 1 );
              }
        }
      }

      Real timeBasisSta, timeBasisEnd;
      GetTime(timeBasisSta);
      // FIXME  magic numbers to fixe the basis
      //        if( (iter <= 5) || (efreeDifPerAtom_ >= 1e-3) ){
//          if(1) {
      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){

              EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
              DblNumVec&    vtotExtElem = eigSol.Ham().Vtot();
              Index3 numGridExtElem = eigSol.FFT().domain.numGrid;
              Index3 numGridExtElemFine = eigSol.FFT().domain.numGridFine;
              Index3 numLGLGrid = hamDG.NumLGLGridElem();

              Index3 numGridElemFine = dmElem.numGridFine;

              // Skip the interpoation if there is no adaptive local
              // basis function.  
              if( eigSol.Psi().NumState() == 0 ){
                hamDG.BasisLGL().LocalMap()[key].Resize( numLGLGrid.prod(), 0 );  
                hamDG.BasisUniformFine().LocalMap()[key].Resize( numGridElemFine.prod(), 0 );  
                continue;
              }

              // Solve the basis functions in the extended element

              Real eigTolNow;
              if( esdfParam.isEigToleranceDynamic ){
                // Dynamic strategy to control the tolerance
                if( iter == 1 )
                  eigTolNow = 1e-2;
                else
                  eigTolNow = eigTolerance_;
              }
              else{
                // Static strategy to control the tolerance
                eigTolNow = eigTolerance_;
              }

              Int numEig = (eigSol.Psi().NumStateTotal())-numUnusedState_;
#if ( _DEBUGlevel_ >= 0 ) 
              statusOFS << " The current tolerance used by the eigensolver is " 
                << eigTolNow << std::endl;
              statusOFS << " The target number of converged eigenvectors is " 
                << numEig << std::endl << std::endl;
#endif

              GetTime( timeSta );
              // FIXME multiple choices of solvers for the extended
              // element should be given in the input file
              if(Diag_SCF_PWDFT_by_Cheby_ == 1)
              {
                // Use CheFSI or LOBPCG on first step 
                if(iter <= 1)
                {
                  if(First_SCF_PWDFT_ChebyCycleNum_ <= 0)
                  { 
                    statusOFS << " >>>> Calling LOBPCG for ALB generation on extended element ..." << std::endl;
#ifndef _COMPLEX_
                    eigSol.LOBPCGSolveReal(numEig, iter, eigMaxIter_, eigMinTolerance_, eigTolNow );    
#endif
                  }
                  else
                  {
                    statusOFS << " >>>> Calling CheFSI with random guess for ALB generation on extended element ..." << std::endl;
                    eigSol.FirstChebyStep(numEig, First_SCF_PWDFT_ChebyCycleNum_, First_SCF_PWDFT_ChebyFilterOrder_);
                  }
                  statusOFS << std::endl;

                }
                else
                {
                  statusOFS << " >>>> Calling CheFSI with previous ALBs for generation of new ALBs ..." << std::endl;
                  statusOFS << " >>>> Will carry out " << eigMaxIter_ << " CheFSI cycles." << std::endl;
                  for (int cheby_iter = 1; cheby_iter <= eigMaxIter_; cheby_iter ++)
                  {
                    statusOFS << std::endl << " >>>> CheFSI for ALBs : Cycle " << cheby_iter << " of " << eigMaxIter_ << " ..." << std::endl;
                    eigSol.GeneralChebyStep(numEig, General_SCF_PWDFT_ChebyFilterOrder_);
                  }
                  statusOFS << std::endl;
                }
              }
              else if(Diag_SCF_PWDFT_by_PPCG_ == 1)
              {
                // Use LOBPCG on very first step, i.e., while starting from random guess
                if(iter <= 1)
                {
                  statusOFS << " >>>> Calling LOBPCG for ALB generation on extended element ..." << std::endl;
                  eigSol.LOBPCGSolveReal(numEig, iter, eigMaxIter_, eigMinTolerance_, eigTolNow );    
                }
                else
                {
                  statusOFS << " >>>> Calling PPCG with previous ALBs for generation of new ALBs ..." << std::endl;
                  eigSol.PPCGSolveReal(numEig, iter, eigMaxIter_, eigMinTolerance_, eigTolNow );
                }
              }             
              else 
              {
                Int eigDynMaxIter = eigMaxIter_;
                eigSol.LOBPCGSolveReal(numEig, iter, eigDynMaxIter, eigMinTolerance_, eigTolNow );
              }

              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << std::endl << " Eigensolver time = "     << timeEnd - timeSta
                << " [s]" << std::endl;
#endif


              // Print out the information
              statusOFS << std::endl 
                << "ALB calculation in extended element " << key << std::endl;
              Real maxRes = 0.0, avgRes = 0.0;
              for(Int ii = 0; ii < eigSol.EigVal().m(); ii++){
                if( maxRes < eigSol.ResVal()(ii) ){
                  maxRes = eigSol.ResVal()(ii);
                }
              avgRes = avgRes + eigSol.ResVal()(ii);
#if ( _DEBUGlevel_ >= 1 )
              Print( statusOFS, 
                     "basis#   = ", ii, 
                     "eigval   = ", eigSol.EigVal()(ii),
                     "resval   = ", eigSol.ResVal()(ii));
#endif
              }
              avgRes = avgRes / eigSol.EigVal().m();
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << std::endl;
              Print(statusOFS, " Max residual of basis = ", maxRes );
              Print(statusOFS, " Avg residual of basis = ", avgRes );
              statusOFS << std::endl;
#endif

              GetTime( timeSta );
              Spinor& psi = eigSol.Psi();

//              statusOFS << "Call Spinor::TransformCoarseToFine." << std::endl;
//              psi.TransformCoarseToFine( eigSol.FFT() );

              DblNumMat& basis = hamDG.BasisLGL().LocalMap()[key];

              SVDLocalizeBasis ( iter, numGridExtElem,
                    numGridElemFine, numLGLGrid, psi, basis );

//              SVDLocalizeBasis ( iter, numGridExtElemFine,
//                    numGridElemFine, numLGLGrid, psi, basis );

            } // own this element
      } // for (i)
//  } // if( perform ALB calculation )


      GetTime( timeBasisEnd );

      statusOFS << std::endl << "Time for generating ALB function is " <<
        timeBasisEnd - timeBasisSta << " [s]" << std::endl << std::endl;


// =================================================================================
// New DM  xmqin
//
      if(iter > 1 && InnermixVariable_ == "densitymatrix"){

        statusOFS << " Iter # " << iter << std::endl;
        statusOFS << " Initialize new density matrix for the current ALBs "  << std::endl;

        GetTime( timeSta );

        ProjectDM ( hamDG.BasisLGLSave(), hamDG.BasisLGL(), distDMMat_);

        GetTime (timeEnd);
        statusOFS << "Time for computing density in the global domain is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
        
        statusOFS << " Generate New DM for this Outer SCF " << std::endl;  
        GetTime( timeSta );
        hamDG.CalculateDensityDM2(
          hamDG.Density(), hamDG.DensityLGL(), distDMMat_ );
        MPI_Barrier( domain_.comm );
        GetTime( timeEnd );
        statusOFS << "Time for computing density in the global domain is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
      } // if(iter > && densitymxtrix )

      // Routine for re-orienting eigenvectors based on current basis set
      if(Diag_SCFDG_by_Cheby_ == 1)
      {
        Real timeSta, timeEnd;
        Real extra_timeSta, extra_timeEnd;

        if(  ALB_LGL_deque_.size() > 0)
        {  
          statusOFS << std::endl << " Rotating the eigenvectors from the previous step ... ";
          GetTime(timeSta);
        }
        // Figure out the element that we own using the standard loop
        for( Int k = 0; k < numElem_[2]; k++ )
          for( Int j = 0; j < numElem_[1]; j++ )
            for( Int i = 0; i < numElem_[0]; i++ )
            {
              Index3 key( i, j, k );
              // If we own this element
              if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) )
              {
                EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
                Index3 numLGLGrid    = hamDG.NumLGLGridElem();
                Spinor& psi = eigSol.Psi();

                // Assuming that wavefun has only 1 component, i.e., spin-unpolarized
                // These are element specific quantities

                // This is the band distributed local basis
                DblNumMat& ref_band_distrib_local_basis = hamDG.BasisLGL().LocalMap()[key];
                DblNumMat band_distrib_local_basis(ref_band_distrib_local_basis.m(),ref_band_distrib_local_basis.n());

                blas::Copy(ref_band_distrib_local_basis.m() * ref_band_distrib_local_basis.n(), 
                           ref_band_distrib_local_basis.Data(), 1,
                           band_distrib_local_basis.Data(), 1);   

                // LGL weights and sqrt weights
                DblNumTns&   LGLWeight3D = hamDG.LGLWeight3D();
                DblNumTns    sqrtLGLWeight3D( numLGLGrid[0], numLGLGrid[1], numLGLGrid[2] );

                Real *ptr1 = LGLWeight3D.Data(), *ptr2 = sqrtLGLWeight3D.Data();
                for( Int i = 0; i < numLGLGrid.prod(); i++ ){
                  *(ptr2++) = std::sqrt( *(ptr1++) );
                }

                // Scale band_distrib_local_basis using sqrt(weights)
                for( Int g = 0; g < band_distrib_local_basis.n(); g++ ){
                  Real *ptr1 = band_distrib_local_basis.VecData(g);
                  Real *ptr2 = sqrtLGLWeight3D.Data();
                  for( Int l = 0; l < band_distrib_local_basis.m(); l++ ){
                    *(ptr1++)  *= *(ptr2++);
                  }
                }

                // Figure out a few dimensions for the row-distribution
                Int heightLGL = numLGLGrid.prod();
                // FIXME! This assumes that SVD does not get rid of basis
                // In the future there should be a parameter to return the
                // number of basis functions on the local DG element
                Int width = psi.NumStateTotal() - numUnusedState_;

                Int widthBlocksize = width / mpisizeRow;
                Int widthLocal = widthBlocksize;

                Int heightLGLBlocksize = heightLGL / mpisizeRow;
                Int heightLGLLocal = heightLGLBlocksize;

                if(mpirankRow < (width % mpisizeRow)){
                  widthLocal = widthBlocksize + 1;
                }

                if(mpirankRow < (heightLGL % mpisizeRow)){
                  heightLGLLocal = heightLGLBlocksize + 1;
                }

                // Convert from band distribution to row distribution
                DblNumMat row_distrib_local_basis(heightLGLLocal, width);
                SetValue(row_distrib_local_basis, 0.0);  

                statusOFS << std::endl << " AlltoallForward: Changing distribution of local basis functions ... ";
                GetTime(extra_timeSta);
                AlltoallForward(band_distrib_local_basis, row_distrib_local_basis, domain_.rowComm);
                GetTime(extra_timeEnd);
                statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta) << " s)";

                // Push the row-distributed matrix into the deque
                ALB_LGL_deque_.push_back( row_distrib_local_basis );

                // If the deque has 2 elements, compute the overlap and perform a rotation of the eigenvectors
                if( ALB_LGL_deque_.size() == 2)
                {
                  GetTime(extra_timeSta);
                  statusOFS << std::endl << " Computing the overlap matrix using basis sets on LGL grid ... ";

                  // Compute the local overlap matrix V2^T * V1            
                  DblNumMat Overlap_Mat( width, width );
                  DblNumMat Overlap_Mat_Temp( width, width );
                  SetValue( Overlap_Mat, 0.0 );
                  SetValue( Overlap_Mat_Temp, 0.0 );

                  double *ptr_0 = ALB_LGL_deque_[0].Data();
                  double *ptr_1 = ALB_LGL_deque_[1].Data();

                  blas::Gemm( 'T', 'N', width, width, heightLGLLocal,
                      1.0, ptr_1, heightLGLLocal, 
                      ptr_0, heightLGLLocal, 
                      0.0, Overlap_Mat_Temp.Data(), width );

                  // Reduce along rowComm (i.e., along the intra-element direction)
                  // to compute the actual overlap matrix
                  MPI_Allreduce( Overlap_Mat_Temp.Data(), 
                      Overlap_Mat.Data(), 
                      width * width, 
                      MPI_DOUBLE, 
                      MPI_SUM, 
                      domain_.rowComm );

                  GetTime(extra_timeEnd);
                  statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta) << " s)";

                  // Rotate the current eigenvectors : This can also be done in parallel
                  // at the expense of an AllReduce along rowComm

                  statusOFS << std::endl << " Rotating the eigenvectors using overlap matrix ... ";
                  GetTime(extra_timeSta);

                  DblNumMat &eigvecs_local = (hamDG.EigvecCoef().LocalMap().begin())->second;               

                  DblNumMat temp_loc_eigvecs_buffer;
                  temp_loc_eigvecs_buffer.Resize(eigvecs_local.m(), eigvecs_local.n());

                  blas::Copy( eigvecs_local.m() * eigvecs_local.n(), 
                      eigvecs_local.Data(), 1, 
                      temp_loc_eigvecs_buffer.Data(), 1 ); 

                  blas::Gemm( 'N', 'N', Overlap_Mat.m(), eigvecs_local.n(), Overlap_Mat.n(), 
                      1.0, Overlap_Mat.Data(), Overlap_Mat.m(), 
                      temp_loc_eigvecs_buffer.Data(), temp_loc_eigvecs_buffer.m(), 
                      0.0, eigvecs_local.Data(), eigvecs_local.m());

                  GetTime(extra_timeEnd);
                  statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta) << " s)" << std::endl;

                  ALB_LGL_deque_.pop_front();
                }        

              } // End of if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) )

        } // End of loop over key indices i.e., for( Int i = 0; i < numElem_[0]; i++ )

        if( iter > 1)
        {
          GetTime(timeEnd);
          statusOFS << std::endl << " All steps of basis rotation completed. ( " << (timeEnd - timeSta) << " s )"<< std::endl;
        }
      } // End of if(Diag_SCFDG_by_Cheby_ == 1)

      // *********************************************************************
      // Inner SCF iteration 
      //
      // Assemble and diagonalize the DG matrix until convergence is
      // reached for updating the basis functions in the next step.
      // *********************************************************************

      GetTime(timeSta);

      // Save the mixing variable in the outer SCF iteration 
      if( OutermixVariable_ == "density" || OutermixVariable_ == "potential" ){
        for( Int k = 0; k < numElem_[2]; k++ )
          for( Int j = 0; j < numElem_[1]; j++ )
            for( Int i = 0; i < numElem_[0]; i++ ){
              Index3 key( i, j, k );
              if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                if( OutermixVariable_ == "density" ){
                  DblNumVec& oldVec = hamDG.Density().LocalMap()[key];
                  mixOuterSave_.LocalMap()[key] = oldVec;
                }
                else if( OutermixVariable_ == "potential" ){
                  DblNumVec& oldVec = hamDG.Vtot().LocalMap()[key];
                  mixOuterSave_.LocalMap()[key] = oldVec;
                }
              } // own this element
        } // for (i)
      }

      // Main function here
      InnerIterate( iter );

//      if( mpirank == 0 ){
//        PrintState( );
//      }

      MPI_Barrier( domain_.comm );
      GetTime( timeEnd );
      statusOFS << "Time for all inner SCF iterations is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;

      // *********************************************************************
      // Post processing 
      // *********************************************************************

      Int numAtom = hamDG.AtomList().size();
      efreeDifPerAtom_ = std::abs(Efree_ - EfreeHarris_) / numAtom;

      // Energy based convergence parameters
      if(iter > 1)
      {        
        md_scf_eband_old_ = md_scf_eband_;
        md_scf_etot_old_ = md_scf_etot_;      
      }
      else
      {
        md_scf_eband_old_ = 0.0;                
        md_scf_etot_old_ = 0.0;
      } 

      md_scf_eband_ = Ekin_;
      md_scf_eband_diff_ = std::abs(md_scf_eband_old_ - md_scf_eband_) / double(numAtom);
      md_scf_etot_ = Etot_;
      md_scf_etot_diff_ = std::abs(md_scf_etot_old_ - md_scf_etot_) / double(numAtom);

      //--------------------------------------------------------------------------------------
      // Compute the error of the mixing variable 
      {
        Real normMixDifLocal = 0.0, normMixOldLocal = 0.0;
        Real normMixDif, normMixOld;
        for( Int k = 0; k < numElem_[2]; k++ )
          for( Int j = 0; j < numElem_[1]; j++ )
            for( Int i = 0; i < numElem_[0]; i++ ){
              Index3 key( i, j, k );
              if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                if( OutermixVariable_ == "density" ){
                  DblNumVec& oldVec = mixOuterSave_.LocalMap()[key];
                  DblNumVec& newVec = hamDG.Density().LocalMap()[key];

                  for( Int p = 0; p < oldVec.m(); p++ ){
                    normMixDifLocal += pow( oldVec(p) - newVec(p), 2.0 );
                    normMixOldLocal += pow( oldVec(p), 2.0 );
                  }
                }
                else if( OutermixVariable_ == "potential" ){
                  DblNumVec& oldVec = mixOuterSave_.LocalMap()[key];
                  DblNumVec& newVec = hamDG.Vtot().LocalMap()[key];

                  for( Int p = 0; p < oldVec.m(); p++ ){
                    normMixDifLocal += pow( oldVec(p) - newVec(p), 2.0 );
                    normMixOldLocal += pow( oldVec(p), 2.0 );
                  }
                }
              } // own this element
        } // for (i)

        mpi::Allreduce( &normMixDifLocal, &normMixDif, 1, MPI_SUM, 
              domain_.colComm );
        mpi::Allreduce( &normMixOldLocal, &normMixOld, 1, MPI_SUM,
              domain_.colComm );

        normMixDif = std::sqrt( normMixDif );
        normMixOld = std::sqrt( normMixOld );

        scfOuterNorm_    = normMixDif / normMixOld;

        Print(statusOFS, "OUTERSCF: EfreeHarris                 = ", EfreeHarris_ ); 
          //            Print(statusOFS, "OUTERSCF: EfreeSecondOrder            = ", EfreeSecondOrder_ ); 
        Print(statusOFS, "OUTERSCF: Efree                       = ", Efree_ ); 
        Print(statusOFS, "OUTERSCF: norm(out-in)/norm(in) = ", scfOuterNorm_ ); 

        Print(statusOFS, "OUTERSCF: Efree diff per atom   = ", efreeDifPerAtom_ ); 

        if(useEnergySCFconvergence_ == 1)
        {
          Print(statusOFS, "OUTERSCF: MD SCF Etot diff (per atom)           = ", md_scf_etot_diff_); 
          Print(statusOFS, "OUTERSCF: MD SCF Eband diff (per atom)          = ", md_scf_eband_diff_); 
        }
        statusOFS << std::endl;
      } // Compute the error of the mixing variable

//-------------------------------------------------------------------------
      // Print out the state variables of the current iteration
      //    PrintState( );
      // Check for convergence
      if(useEnergySCFconvergence_ == 0)
      { 
        if( (iter >= 2) && 
          ( (scfOuterNorm_ < DFTscfOuterTolerance_) && 
            (efreeDifPerAtom_ < DFTscfOuterEnergyTolerance_) ) ){
            /* converged */
          statusOFS << " Outer SCF is converged in " << iter << " steps !" << std::endl;
          isSCFConverged = true;
        }
      }  
      else
      {
        if( (iter >= 2) && 
            (md_scf_etot_diff_ < md_scf_etot_diff_tol_) &&
            (md_scf_eband_diff_ < md_scf_eband_diff_tol_) )
        {
          // converged via energy criterion
          statusOFS << " Outer SCF is converged via energy condition in " << iter << " steps !" << std::endl;
          isSCFConverged = true;
        }
      } // if(useEnergySCFconvergence_ == 0)
      // Potential mixing for the outer SCF iteration. or no mixing at all anymore?
      // It seems that no mixing is the best.

      GetTime( timeIterEnd );
      statusOFS << "Time for this SCF iteration = " << timeIterEnd - timeIterStart
        << " [s]" << std::endl;
    } // for( iter )

    GetTime( timeTotalEnd );

    statusOFS << std::endl;
    statusOFS << "Total time for all SCF iterations = " << 
      timeTotalEnd - timeTotalStart << " [s]" << std::endl;

    if(scfdg_ion_dyn_iter_ >= 1)
    {
      statusOFS << " Ion dynamics iteration " << scfdg_ion_dyn_iter_ << " : ";
    }

    if( isSCFConverged == true ){
      statusOFS << "Total number of outer SCF steps for SCF convergence = " <<
        iter - 1 << std::endl;
    }
    else{
      statusOFS << "Total number of outer SCF steps (SCF not converged) = " <<
        DFTscfOuterMaxIter_ << std::endl;
    } // if(scfdg_ion_dyn_iter_ >= 1)

  } // if( !hamDG.IsHybrid() || !hamDG.IsEXXActive())

  // xmqin add for HSE06
  //  FixMe:  Next outer SCF for hybrids

  if( hamDG.IsHybrid() ){

    std::ostringstream msg;
    msg << "Starting Hybrid DFT SCF iteration.";
    PrintBlock( statusOFS, msg.str() );
    bool isSCFConverged = false;

    if( hamDG.IsEXXActive() == false )
      hamDG.SetEXXActive(true);

    if(esdfParam.XCType == "XC_HYB_GGA_XC_HSE06")
    {
      hamDG.Setup_XC( "XC_HYB_GGA_XC_HSE06");
    }
    else if (esdfParam.XCType == "XC_HYB_GGA_XC_PBEH")
    {
      hamDG.Setup_XC( "XC_HYB_GGA_XC_PBEH");
    }

    statusOFS << "Re-calculate XC for Hybrid DFT " << std::endl;

    if( isCalculateGradRho_ ) {
      GetTime( timeSta );
      hamDG.CalculateGradDensity( fftDG );
      GetTime( timeEnd );

      statusOFS << "Time for calculating gradient of density is " <<
         timeEnd - timeSta << " [s]" << std::endl << std::endl;
    }

    GetTime( timeSta );
    hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), fftDG );
    GetTime( timeEnd );
    statusOFS << "Time for calculating XC is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
    // Compute the Hartree potential
    GetTime( timeSta );
    hamDG.CalculateHartree( hamDG.Vhart(), fftDG );
    GetTime( timeEnd );
    statusOFS << "Time for calculating Hartree is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
    // No external potential
    // Compute the total potential
    GetTime( timeSta );
    hamDG.CalculateVtot( hamDG.Vtot() );
    GetTime( timeEnd );
    statusOFS << "Time for calculating Vtot is " <<
      timeEnd - timeSta << " [s]" << std::endl;

    Real timeIterStart(0), timeIterEnd(0);
    Real timeTotalStart(0), timeTotalEnd(0);

    scfTotalInnerIter_  = 0;

    GetTime( timeTotalStart );

    Int iter;

    // Total number of SVD basis functions. Determined at the first
    // outer SCF and is not changed later. This facilitates the reuse of
    // symbolic factorization
    SetupDMMix (); 

    for (iter=1; iter <= HybridscfOuterMaxIter_; iter++) {
      if ( isSCFConverged && (iter >= HybridscfOuterMinIter_ ) ) break;

      // Performing each iteartion
      {
        std::ostringstream msg;
        msg << "Hybrid DFT Outer SCF iteration # " << iter;
        PrintBlock( statusOFS, msg.str() );
      }

      GetTime( timeIterStart );
      
      // *********************************************************************
      // Update the local potential in the extended element and the element.
      //
      // NOTE: The modification of the potential on the extended element
      // to reduce the Gibbs phenomena is now in UpdateElemLocalPotential
      // *********************************************************************
      {
        GetTime(timeSta);

        UpdateElemLocalPotential();

        GetTime( timeEnd );

        statusOFS << "OuterSCF:: Time for updating the local potential in the extended element and the element is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
      }

      if( iter == 1 || solutionMethod_ == "diag" ) {
        GetTime(timeSta);
        scfdg_compute_fullDM();
        GetTime( timeEnd );
        statusOFS << "InnerSCF: Recalculate density matrix for diag method " << std::endl;
      }

      { 
        // Generate new basis for HSE 
        for( Int k = 0; k < numElem_[2]; k++ )
          for( Int j = 0; j < numElem_[1]; j++ )
            for( Int i = 0; i < numElem_[0]; i++ ){
              Index3 key( i, j, k );
              if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){

                DblNumMat& basisOld = hamDG.BasisLGL().LocalMap()[key];
                DblNumMat& basisSave = hamDG.BasisLGLSave().LocalMap()[key];

                basisSave.Resize( basisOld.m(), basisOld.n() );

                SetValue( basisSave, 0.0 );

                blas::Copy( basisOld.m()*basisOld.n(), basisOld.Data(), 1, basisSave.Data(), 1 );
              }
        }

        // *********************************************************************
        // Solve the basis functions in the extended element
        // *********************************************************************

        Real timeBasisSta, timeBasisEnd;
        GetTime(timeBasisSta);
        // FIXME  magic numbers to fixe the basis
//      if( (iter <= 5) || (efreeDifPerAtom_ >= 1e-3) ){
        for( Int k = 0; k < numElem_[2]; k++ )
          for( Int j = 0; j < numElem_[1]; j++ )
            for( Int i = 0; i < numElem_[0]; i++ ){
              Index3 key( i, j, k );
              if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
        
               // Need ham for extended elements  xmqin
               // hamDG don't include hamKS for extended element
                Hamiltonian&  ham = eigSol.Ham();
                Spinor&       psi = eigSol.Psi();
                Fourier& fft =  eigSol.FFT();
               
                DblNumVec&    vtotExtElem = eigSol.Ham().Vtot();
                Index3 numGridExtElem = eigSol.FFT().domain.numGrid;
                Index3 numGridExtElemFine = eigSol.FFT().domain.numGridFine;
                Index3 numLGLGrid     = hamDG.NumLGLGridElem();

                Index3 numGridElemFine    = dmElem.numGridFine;

                // Skip the interpoation if there is no adaptive local
                // basis function.  
                if( eigSol.Psi().NumState() == 0 ){
                  hamDG.BasisLGL().LocalMap()[key].Resize( numLGLGrid.prod(), 0 );  
                  hamDG.BasisUniformFine().LocalMap()[key].Resize( numGridElemFine.prod(), 0 );  
                  continue;
                }

                // Solve the basis functions in the extended element
                Real eigTolNow;
                if( esdfParam.isEigToleranceDynamic ){
                  // Dynamic strategy to control the tolerance
                  if( iter == 1 )
                    eigTolNow = 1e-2;
                  else
                    eigTolNow = eigTolerance_;
                }
                else{
                  // Static strategy to control the tolerance
                  eigTolNow = eigTolerance_;
                }

                Int numEig = (eigSol.Psi().NumStateTotal())-numUnusedState_;
                statusOFS << "The current tolerance used by the eigensolver is " 
                  << eigTolNow << std::endl;
                statusOFS << "The target number of converged eigenvectors is " 
                  << numEig << std::endl << std::endl;
                //------------------------------------------------------------------------------------
                //Hybrid calculation in PWDFT to obtain hybrid-functional basis
                bool isFixColumnDF = false;
                if( ham.IsEXXActive() == false )
                   ham.SetEXXActive(true);
                // psi for EXX
                // Each psi includes a factor fac=sqrt(Volume/ntot) for DFT calculations
                // That is \int |psi(r)|^2 dr =  sum_g |psi(r_g)|^2 * dv = 1
                // dv = Volume/ntot , fac = \sqrt (dv)
                // For simply, psi(r_g) is multiplied by fac
                //
                // However, the Denisty \rho(r) and Potential V(r) do not include this fac
                // Because the density has been be scaled in hamiltonian.cpp
                // blas::Scal( ntotFine, (numSpin_ * Real(numOccupiedState_) * Real(ntotFine)) / ( vol * val ),
                //        density_.VecData(RHO), 1 );

                // Then,  H * psi_i = \epsilon psi_i aslo has a fac (not in H(r) but in psi(r) )
                //
                // For EXX calculation: phi is update in inner iteration step
                //
                // Vx(r,r')*phi_i(r') = \sum_j psi_j(r) *v(r,r')* psi_j(r') * phi_i(r')
                //
                // In order to make Vx * phi shares the same fac as H * psi
                // we have to use SetPhiEXX transforms psi(r) from psi(r_g) * fac to psi(r_g), that is remove the fac
                // for Vx(r, r')

                // Fock energies
                Real fock0 = 0.0, fock1 = 0.0, fock2 = 0.0;

                // EXX: Run SCF::Iterate here
                bool isPhiIterConverged = false;

                Real dExx;

                GetTime( timeSta );

                ham.SetPhiEXX( psi, fft );

#if ( _DEBUGlevel_ >= 2 )
                statusOFS << " psi.NumStateTotal " << psi.NumStateTotal() << std::endl;
                statusOFS << " ham.NumOccupiedState " << ham.NumOccupiedState() << std::endl;
                statusOFS << " ham.OccupationRate " << ham.OccupationRate() << std::endl;
#endif

                if( esdfParam.isHybridACE ){
                 if( esdfParam.isHybridDF ){
                    ham.CalculateVexxACEDF( psi, fft, isFixColumnDF );
                    // Fix the column after the first iteraiton
                    isFixColumnDF = true;
                 }
                  else
                  {
                    ham.CalculateVexxACE ( psi, fft );
                  }
                }

                GetTime( timeEnd );
                statusOFS << "Time for updating Phi related variable is " <<
                timeEnd - timeSta << " [s]" << std::endl;

                GetTime( timeSta );
                fock2 = ham.CalculateEXXEnergy( psi, fft );
                GetTime( timeEnd );

                Print(statusOFS, "ExtElem Fock energy 1    = ",  fock2, "[au]");

                statusOFS << "Time for computing the EXX energy is " <<
                  timeEnd - timeSta << " [s]" << std::endl;

                Efock_ = fock2;
                fock1  = fock2;

                GetTime( timeSta );

                for( Int phiIter = 1; phiIter <= PhiMaxIter_; phiIter++ ){
                  std::ostringstream msg;
                  msg << "Phi iteration # " << phiIter;
                  PrintBlock( statusOFS, msg.str() );

                  if(Diag_SCF_PWDFT_by_Cheby_ == 1)
                  {
                  // Use CheFSI or LOBPCG on first step 
                    if(iter <= 1)
                    {
                      if(First_SCF_PWDFT_ChebyCycleNum_ <= 0)
                      { 
                        statusOFS << " >>>> Calling LOBPCG for ALB generation on extended element ..." << std::endl;
                        eigSol.LOBPCGSolveReal(numEig, iter, eigMaxIter_, eigMinTolerance_, eigTolNow );    
                      }
                      else
                      {
                        statusOFS << " >>>> Calling CheFSI with random guess for ALB generation on extended element ..." << std::endl;
                        eigSol.FirstChebyStep(numEig, First_SCF_PWDFT_ChebyCycleNum_, First_SCF_PWDFT_ChebyFilterOrder_);
                      }
                      statusOFS << std::endl;
                    }
                    else
                    {
                      statusOFS << " >>>> Calling CheFSI with previous ALBs for generation of new ALBs ..." << std::endl;
                      statusOFS << " >>>> Will carry out " << eigMaxIter_ << " CheFSI cycles." << std::endl;

                      for (int cheby_iter = 1; cheby_iter <= eigMaxIter_; cheby_iter ++)
                      {
                        statusOFS << std::endl << " >>>> CheFSI for ALBs : Cycle " << cheby_iter << " of " << eigMaxIter_ << " ..." << std::endl;
                        eigSol.GeneralChebyStep(numEig, General_SCF_PWDFT_ChebyFilterOrder_);
                      }
                      statusOFS << std::endl;
                    }
                  }
                  else if(Diag_SCF_PWDFT_by_PPCG_ == 1)
                  {
                    // Use LOBPCG on very first step, i.e., while starting from random guess
                    if(iter <= 1)
                    {
                      statusOFS << " >>>> Calling LOBPCG for ALB generation on extended element ..." << std::endl;
                      eigSol.LOBPCGSolveReal(numEig, iter, eigMaxIter_, eigMinTolerance_, eigTolNow );    
                    }
                    else
                    {
                      statusOFS << " >>>> Calling PPCG with previous ALBs for generation of new ALBs ..." << std::endl;
                      eigSol.PPCGSolveReal(numEig, iter, eigMaxIter_, eigMinTolerance_, eigTolNow );
                    }
                  }             
                  else 
                  {
                    Int eigDynMaxIter = eigMaxIter_;
                    eigSol.LOBPCGSolveReal(numEig, iter, eigDynMaxIter, eigMinTolerance_, eigTolNow );
                  }

                  GetTime( timeEnd );
                  statusOFS << std::endl << "Eigensolver time = "     << timeEnd - timeSta
                    << " [s]" << std::endl;

                  GetTime( timeSta );
#if ( _DEBUGlevel_ >= 2 )
                  statusOFS << " psi.NumStateTotal " << psi.NumStateTotal() << std::endl;
                  statusOFS << " ham.NumOccupiedState " << ham.NumOccupiedState() << std::endl;
                  statusOFS << " ham.OccupationRate " << ham.OccupationRate() << std::endl;
#endif

                  CalculateOccupationRateExtElem( eigSol.EigVal(), ham.OccupationRate(),
                      psi.NumStateTotal(), ham.NumOccupiedState() );
                  GetTime( timeEnd );

                  statusOFS << "Time for computing occupation rate in PWDFT is " <<
                    timeEnd - timeSta << " [s]" << std::endl << std::endl;

                  // Update Phi <- Psi
                  GetTime( timeSta );
                  ham.SetPhiEXX( psi, fft );
  
                  if( esdfParam.isHybridACE ){
                    if( esdfParam.isHybridDF ){
                      ham.CalculateVexxACEDF( psi, fft, isFixColumnDF );
                      // Fix the column after the first iteraiton
                      isFixColumnDF = true;
                    }
                    else
                    {
                      ham.CalculateVexxACE ( psi, fft );
                    }
                  }
  
                  GetTime( timeEnd );
                  statusOFS << "Time for updating Phi related variable is " <<
                  timeEnd - timeSta << " [s]" << std::endl;
  
                  GetTime( timeSta );
                  fock2 = ham.CalculateEXXEnergy( psi, fft );
                  GetTime( timeEnd );
  
                  // Note: initially fock1 = 0.0. So it should at least run for 1 iteration.
                  dExx = std::abs(fock2 - fock1) / std::abs(fock2);
                  fock1 = fock2;
                  Efock_ = fock2;
            
                  statusOFS << std::endl;
                  Print(statusOFS, "ExtElem Fock energy       = ",  Efock_, "[au]");
                  Print(statusOFS, "dExx for PWDFT            = ",  dExx, "[au]");
                  if( dExx < PhiTolerance_ ){
                    statusOFS << "SCF for hybrid functional is converged in "
                      << phiIter << " steps !" << std::endl;
                    isPhiIterConverged = true;
                  }
  
                  if ( isPhiIterConverged ) break;
                } // for(phiIter)
  
                // Print out the information
                statusOFS << std::endl 
                  << "ALB calculation in extended element " << key << std::endl;
                Real maxRes = 0.0, avgRes = 0.0;
                for(Int ii = 0; ii < eigSol.EigVal().m(); ii++){
                  if( maxRes < eigSol.ResVal()(ii) ){
                    maxRes = eigSol.ResVal()(ii);
                  }
                  avgRes = avgRes + eigSol.ResVal()(ii);
#if ( _DEBUGlevel_ >= 2 )
                  Print(statusOFS, 
                    "basis#   = ", ii, 
                    "eigval   = ", eigSol.EigVal()(ii),
                    "resval   = ", eigSol.ResVal()(ii));
#endif
                }
                avgRes = avgRes / eigSol.EigVal().m();
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << std::endl;
                Print(statusOFS, "Max residual of basis = ", maxRes );
                Print(statusOFS, "Avg residual of basis = ", avgRes );
                statusOFS << std::endl;
#endif

//                statusOFS << "Call Spinor::TransformCoarseToFine." << std::endl;
//                psi.TransformCoarseToFine( eigSol.FFT() );

                DblNumMat& basis = hamDG.BasisLGL().LocalMap()[key];


                SVDLocalizeBasis ( iter, numGridExtElem,
                  numGridElemFine, numLGLGrid, psi, basis );

//                SVDLocalizeBasis ( iter, numGridExtElemFine,
//                  numGridElemFine, numLGLGrid, psi, basis );
              } // own this element
        } // for (i)

        GetTime( timeBasisEnd );

        statusOFS << std::endl << "Time for generating ALB function is " <<
          timeBasisEnd - timeBasisSta << " [s]" << std::endl;

#if ( _DEBUGlevel_ >= 1 )  
        statusOFS << std::endl << " Hybrid ALB and DFT DM " << std::endl;

        for( Int k = 0; k < numElem_[2]; k++ )
           for( Int j = 0; j < numElem_[1]; j++ )
             for( Int i = 0; i < numElem_[0]; i++ ){
               Index3 key( i, j, k );
               if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){

                 DblNumMat& basisNew = hamDG.BasisLGL().LocalMap()[key];
                 DblNumMat& basisOld = hamDG.BasisLGLSave().LocalMap()[key];
                 Int m = basisNew.m();
                 Int n = basisNew.n();
                 Int size = m * n;

                  statusOFS <<" Difference  " <<std::endl;
                 for (Int g =0; g < size; g++ ) {
                  statusOFS <<  basisOld.data_[g]  <<  " -- " << basisNew.data_[g] << "   " ; 
                 }
                  statusOFS << std::endl;
 
               }
        
        }

#endif

        MPI_Barrier( domain_.comm );

        // New DM  xmqin
        GetTime( timeSta );

        ProjectDM ( hamDG.BasisLGLSave(), hamDG.BasisLGL(), distDMMat_);

        GetTime (timeEnd);
        statusOFS << "Time for computing density in the global domain is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
            
        statusOFS << "Generate New DM for this Outer SCF " << std::endl;  
        GetTime( timeSta );
        hamDG.CalculateDensityDM2( hamDG.Density(), hamDG.DensityLGL(), distDMMat_ );
        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << " Time for computing density in the global domain is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
      }

      if(esdfParam.isDGHFISDF){
        hamDG.DGHFX_ISDF( );
      }
      else{
        hamDG.CollectNeighborBasis( );
      }

      // Routine for re-orienting eigenvectors based on current basis set
      if(Diag_SCFDG_by_Cheby_ == 1)
      {
        Real timeSta, timeEnd;
        Real extra_timeSta, extra_timeEnd;

        if(  ALB_LGL_deque_.size() > 0)
        {  
          statusOFS << std::endl << " Rotating the eigenvectors from the previous step ... ";
          GetTime(timeSta);
        }
        // Figure out the element that we own using the standard loop
        for( Int k = 0; k < numElem_[2]; k++ )
          for( Int j = 0; j < numElem_[1]; j++ )
            for( Int i = 0; i < numElem_[0]; i++ )
            {
              Index3 key( i, j, k );

              // If we own this element
              if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) )
              {
                EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
                Index3 numLGLGrid    = hamDG.NumLGLGridElem();
                Spinor& psi = eigSol.Psi();

                // Assuming that wavefun has only 1 component, i.e., spin-unpolarized
                // These are element specific quantities

                // This is the band distributed local basis
                DblNumMat& ref_band_distrib_local_basis = hamDG.BasisLGL().LocalMap()[key];
                DblNumMat band_distrib_local_basis(ref_band_distrib_local_basis.m(),ref_band_distrib_local_basis.n());

                blas::Copy(ref_band_distrib_local_basis.m() * ref_band_distrib_local_basis.n(), 
                    ref_band_distrib_local_basis.Data(), 1,
                    band_distrib_local_basis.Data(), 1);   

                // LGL weights and sqrt weights
                DblNumTns&   LGLWeight3D = hamDG.LGLWeight3D();
                DblNumTns    sqrtLGLWeight3D( numLGLGrid[0], numLGLGrid[1], numLGLGrid[2] );

                Real *ptr1 = LGLWeight3D.Data(), *ptr2 = sqrtLGLWeight3D.Data();
                for( Int i = 0; i < numLGLGrid.prod(); i++ ){
                  *(ptr2++) = std::sqrt( *(ptr1++) );
                }

                // Scale band_distrib_local_basis using sqrt(weights)
                for( Int g = 0; g < band_distrib_local_basis.n(); g++ ){
                  Real *ptr1 = band_distrib_local_basis.VecData(g);
                  Real *ptr2 = sqrtLGLWeight3D.Data();
                  for( Int l = 0; l < band_distrib_local_basis.m(); l++ ){
                    *(ptr1++)  *= *(ptr2++);
                  }
                }
                // Figure out a few dimensions for the row-distribution
                Int heightLGL = numLGLGrid.prod();
                // FIXME! This assumes that SVD does not get rid of basis
                // In the future there should be a parameter to return the
                // number of basis functions on the local DG element
                Int width = psi.NumStateTotal() - numUnusedState_;

                Int widthBlocksize = width / mpisizeRow;
                Int widthLocal = widthBlocksize;

                Int heightLGLBlocksize = heightLGL / mpisizeRow;
                Int heightLGLLocal = heightLGLBlocksize;

                if(mpirankRow < (width % mpisizeRow)){
                  widthLocal = widthBlocksize + 1;
                }

                if(mpirankRow < (heightLGL % mpisizeRow)){
                  heightLGLLocal = heightLGLBlocksize + 1;
                }

                // Convert from band distribution to row distribution
                DblNumMat row_distrib_local_basis(heightLGLLocal, width);
                SetValue(row_distrib_local_basis, 0.0);  

                statusOFS << std::endl << " AlltoallForward: Changing distribution of local basis functions ... ";
                GetTime(extra_timeSta);
                AlltoallForward(band_distrib_local_basis, row_distrib_local_basis, domain_.rowComm);
                GetTime(extra_timeEnd);
                statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta) << " s)";

                // Push the row-distributed matrix into the deque
                ALB_LGL_deque_.push_back( row_distrib_local_basis );

                // If the deque has 2 elements, compute the overlap and perform a rotation of the eigenvectors
                if( ALB_LGL_deque_.size() == 2)
                {
                  GetTime(extra_timeSta);
                  statusOFS << std::endl << " Computing the overlap matrix using basis sets on LGL grid ... ";

                  // Compute the local overlap matrix V2^T * V1            
                  DblNumMat Overlap_Mat( width, width );
                  DblNumMat Overlap_Mat_Temp( width, width );
                  SetValue( Overlap_Mat, 0.0 );
                  SetValue( Overlap_Mat_Temp, 0.0 );

                  double *ptr_0 = ALB_LGL_deque_[0].Data();
                  double *ptr_1 = ALB_LGL_deque_[1].Data();

                  blas::Gemm( 'T', 'N', width, width, heightLGLLocal,
                      1.0, ptr_1, heightLGLLocal, 
                      ptr_0, heightLGLLocal, 
                      0.0, Overlap_Mat_Temp.Data(), width );

                  // Reduce along rowComm (i.e., along the intra-element direction)
                  // to compute the actual overlap matrix
                  MPI_Allreduce( Overlap_Mat_Temp.Data(), 
                      Overlap_Mat.Data(), 
                      width * width, 
                      MPI_DOUBLE, 
                      MPI_SUM, 
                      domain_.rowComm );

                  GetTime(extra_timeEnd);
                  statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta) << " s)";

                  // Rotate the current eigenvectors : This can also be done in parallel
                  // at the expense of an AllReduce along rowComm

                  statusOFS << std::endl << " Rotating the eigenvectors using overlap matrix ... ";
                  GetTime(extra_timeSta);

                  DblNumMat &eigvecs_local = (hamDG.EigvecCoef().LocalMap().begin())->second;               

                  DblNumMat temp_loc_eigvecs_buffer;
                  temp_loc_eigvecs_buffer.Resize(eigvecs_local.m(), eigvecs_local.n());

                  blas::Copy( eigvecs_local.m() * eigvecs_local.n(), 
                      eigvecs_local.Data(), 1, 
                      temp_loc_eigvecs_buffer.Data(), 1 ); 

                  blas::Gemm( 'N', 'N', Overlap_Mat.m(), eigvecs_local.n(), Overlap_Mat.n(), 
                      1.0, Overlap_Mat.Data(), Overlap_Mat.m(), 
                      temp_loc_eigvecs_buffer.Data(), temp_loc_eigvecs_buffer.m(), 
                      0.0, eigvecs_local.Data(), eigvecs_local.m());

                  GetTime(extra_timeEnd);
                  statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta) << " s)" << std::endl;

                  ALB_LGL_deque_.pop_front();
                }        
              } // End of if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) )
          } // End of loop over key indices i.e., for( Int i = 0; i < numElem_[0]; i++ )
  
          if( iter > 1)
          {
            GetTime(timeEnd);
            statusOFS << std::endl << " All steps of basis rotation completed. ( " << (timeEnd - timeSta) << " s )"<< std::endl;
          }
      } // End of if(Diag_SCFDG_by_Cheby_ == 1)
  
      // *********************************************************************
      // Inner SCF iteration 
      //
      // Assemble and diagonalize the DG matrix until convergence is
      // reached for updating the basis functions in the next step.
      // *********************************************************************
  
      GetTime(timeSta);
  
      // Save the mixing variable in the outer SCF iteration 
      if( OutermixVariable_ == "density" || OutermixVariable_ == "potential" ){
        for( Int k = 0; k < numElem_[2]; k++ )
          for( Int j = 0; j < numElem_[1]; j++ )
            for( Int i = 0; i < numElem_[0]; i++ ){
              Index3 key( i, j, k );
              if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                if( OutermixVariable_ == "density" ){
                  DblNumVec& oldVec = hamDG.Density().LocalMap()[key];
                  mixOuterSave_.LocalMap()[key] = oldVec;
                }
                else if( OutermixVariable_ == "potential" ){
                  DblNumVec& oldVec = hamDG.Vtot().LocalMap()[key];
                  mixOuterSave_.LocalMap()[key] = oldVec;
                }
              } // own this element
            } // for (i)
      }
      // Main function here
      InnerIterate( iter );
  
      MPI_Barrier( domain_.comm );
      GetTime( timeEnd );
      statusOFS << "Time for all inner SCF iterations is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
      // *********************************************************************
      // Post processing 
      // *********************************************************************
  
      Int numAtom = hamDG.AtomList().size();
      efreeDifPerAtom_ = std::abs(Efree_ - EfreeHarris_) / numAtom;
  
      // Energy based convergence parameters
      if(iter > 1)
      {        
        md_scf_eband_old_ = md_scf_eband_;
        md_scf_etot_old_ = md_scf_etot_;      
      }
      else
      {
        md_scf_eband_old_ = 0.0;                
        md_scf_etot_old_ = 0.0;
      } 
  
      md_scf_eband_ = Ekin_;
      md_scf_eband_diff_ = std::abs(md_scf_eband_old_ - md_scf_eband_) / double(numAtom);
      md_scf_etot_ = Etot_;
      //md_scf_etot_ = EfreeHarris_;
      md_scf_etot_diff_ = std::abs(md_scf_etot_old_ - md_scf_etot_) / double(numAtom);
      //Int numAtom = hamDG.AtomList().size();;
  
  
      // Compute the error of the mixing variable 
      {
        Real normMixDifLocal = 0.0, normMixOldLocal = 0.0;
        Real normMixDif, normMixOld;
        for( Int k = 0; k < numElem_[2]; k++ )
          for( Int j = 0; j < numElem_[1]; j++ )
            for( Int i = 0; i < numElem_[0]; i++ ){
              Index3 key( i, j, k );
              if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                if( OutermixVariable_ == "density" ){
                  DblNumVec& oldVec = mixOuterSave_.LocalMap()[key];
                  DblNumVec& newVec = hamDG.Density().LocalMap()[key];

                  for( Int p = 0; p < oldVec.m(); p++ ){
                    normMixDifLocal += pow( oldVec(p) - newVec(p), 2.0 );
                    normMixOldLocal += pow( oldVec(p), 2.0 );
                  }
                }
                else if( OutermixVariable_ == "potential" ){
                  DblNumVec& oldVec = mixOuterSave_.LocalMap()[key];
                  DblNumVec& newVec = hamDG.Vtot().LocalMap()[key];

                  for( Int p = 0; p < oldVec.m(); p++ ){
                    normMixDifLocal += pow( oldVec(p) - newVec(p), 2.0 );
                    normMixOldLocal += pow( oldVec(p), 2.0 );
                  }
                }
              } // own this element
            } // for (i)


        mpi::Allreduce( &normMixDifLocal, &normMixDif, 1, MPI_SUM, 
            domain_.colComm );
        mpi::Allreduce( &normMixOldLocal, &normMixOld, 1, MPI_SUM,
            domain_.colComm );

        normMixDif = std::sqrt( normMixDif );
        normMixOld = std::sqrt( normMixOld );

        scfOuterNorm_    = normMixDif / normMixOld;

        Print(statusOFS, "OUTERSCF: EfreeHarris                 = ", EfreeHarris_ ); 
        Print(statusOFS, "OUTERSCF: Efree                       = ", Efree_ ); 
        Print(statusOFS, "OUTERSCF: norm(out-in)/norm(in) = ", scfOuterNorm_ ); 

        Print(statusOFS, "OUTERSCF: Efree diff per atom   = ", efreeDifPerAtom_ ); 

        if(useEnergySCFconvergence_ == 1)
        {
          Print(statusOFS, "OUTERSCF: MD SCF Etot diff (per atom)           = ", md_scf_etot_diff_); 
          Print(statusOFS, "OUTERSCF: MD SCF Eband diff (per atom)          = ", md_scf_eband_diff_); 
        }
        statusOFS << std::endl;
      }
      //    PrintState( );

    // Check for convergence
      if(useEnergySCFconvergence_ == 0)
      {  
        if( (iter >= 2) && 
            ( (scfOuterNorm_ < HybridscfOuterTolerance_) && 
              (efreeDifPerAtom_ < HybridscfOuterEnergyTolerance_) ) ){
        /* converged */
          statusOFS << " Outer SCF is converged in " << iter << " steps !" << std::endl;
          isSCFConverged = true;
        }
      }
      else
      {
        if( (iter >= 2) && 
          (md_scf_etot_diff_ < md_scf_etot_diff_tol_) &&
          (md_scf_eband_diff_ < md_scf_eband_diff_tol_) )
        {
        // converged via energy criterion
          statusOFS << " Outer SCF is converged via energy condition in " << iter << " steps !" << std::endl;
          isSCFConverged = true;

        }
      } // if(useEnergySCFconvergence_ == 0)

      // Potential mixing for the outer SCF iteration. or no mixing at all anymore?
      // It seems that no mixing is the best.

      GetTime( timeIterEnd );
      statusOFS << "Time for this SCF iteration = " << timeIterEnd - timeIterStart
        << " [s]" << std::endl;

    } // for( iter )

    GetTime( timeTotalEnd );

    statusOFS << std::endl;
    statusOFS << "Total time for all SCF iterations = " << 
      timeTotalEnd - timeTotalStart << " [s]" << std::endl;

    if(scfdg_ion_dyn_iter_ >= 1)
    {
      statusOFS << " Ion dynamics iteration " << scfdg_ion_dyn_iter_ << " : ";
    }

    if( isSCFConverged == true ){
      statusOFS << " Total number of outer SCF steps for SCF convergence = " <<
        iter - 1 << std::endl;
    }
    else{
      statusOFS << " Total number of outer SCF steps (SCF not converged) = " <<
        HybridscfOuterMaxIter_ << std::endl;
    } // if(scfdg_ion_dyn_iter_ >= 1)

  } //       if( hamDG.IsHybrid()  )


  //    if(0)
  //    {
  //      // Output the electron density on the LGL grid in each element
  //    if(0)
  //    {
  //      // Output the electron density on the LGL grid in each element
  //      std::ostringstream rhoStream;      
  //
  //      NumTns<std::vector<DblNumVec> >& LGLGridElem =
  //        hamDG.LGLGridElem();
  //
  //      for( Int k = 0; k < numElem_[2]; k++ )
  //        for( Int j = 0; j < numElem_[1]; j++ )
  //          for( Int i = 0; i < numElem_[0]; i++ ){
  //            Index3 key( i, j, k );
  //            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
  //              DblNumVec&  denVec = hamDG.DensityLGL().LocalMap()[key];
  //              std::vector<DblNumVec>& grid = LGLGridElem(i, j, k);
  //              for( Int d = 0; d < DIM; d++ ){
  //                serialize( grid[d], rhoStream, NO_MASK );
  //              }
  //              serialize( denVec, rhoStream, NO_MASK );
  //            }
  //          } // for (i)
  //      SeparateWrite( "DENLGL", rhoStream );
  //    }



  // *********************************************************************
  // Calculate the VDW contribution and the force
  // *********************************************************************
  Real timeForceSta, timeForceEnd;
  GetTime( timeForceSta );
  if( solutionMethod_ == "diag" ){

    if(SCFDG_comp_subspace_engaged_ == false)
    {
      statusOFS << std::endl << " Computing forces using eigenvectors ..." << std::endl;
      hamDG.CalculateForce( fftDG );
    }
    else
    {
      double extra_timeSta, extra_timeEnd;

      statusOFS << std::endl << " Computing forces using Density Matrix ...";
      statusOFS << std::endl << " Computing full Density Matrix for Complementary Subspace method ...";
      GetTime(extra_timeSta);

      // Compute the full DM in the complementary subspace method
      scfdg_complementary_subspace_compute_fullDM();

      GetTime(extra_timeEnd);

      statusOFS << std::endl << " DM Computation took " << (extra_timeEnd - extra_timeSta) << " s." << std::endl;

      // Call the PEXSI force evaluator
      hamDG.CalculateForceDM( fftDG, distDMMat_ );        
    }
  }
  else if( solutionMethod_ == "pexsi" ){
    hamDG.CalculateForceDM( fftDG, distDMMat_ );
  }
  GetTime( timeForceEnd );
  statusOFS << "Time for computing the force is " <<
    timeForceEnd - timeForceSta << " [s]" << std::endl << std::endl;

  // Calculate the VDW energy
  if( VDWType_ == "DFT-D2"){
    CalculateVDW ( Evdw_, forceVdw_ );
    // Update energy
    Etot_  += Evdw_;
    Efree_ += Evdw_;
    EfreeHarris_ += Evdw_;
    Ecor_  += Evdw_;

    // Update force
    std::vector<Atom>& atomList = hamDG.AtomList();
    for( Int a = 0; a < atomList.size(); a++ ){
      atomList[a].force += Point3( forceVdw_(a,0), forceVdw_(a,1), forceVdw_(a,2) );
    }
  } 

  // Output the information after SCF
  {
    Real HOMO, LUMO, EG;

    HOMO = hamDG.EigVal()( hamDG.NumOccupiedState()-1 );
    if( hamDG.NumExtraState() > 0 ) {
      LUMO = hamDG.EigVal()( hamDG.NumOccupiedState());
      EG = LUMO - HOMO;
    }
#if 1
  if(SCFDG_comp_subspace_engaged_ == false)
  {
    statusOFS << std::endl << "Eigenvalues in the global domain." << std::endl;
    for(Int i = 0; i < hamDG.EigVal().m(); i++){
      Print(statusOFS,
          "band#    = ", i,
          "eigval   = ", hamDG.EigVal()(i),
          "occrate  = ", hamDG.OccupationRate()(i));
    }
  }
#endif

   
    // Print out the energy
    PrintBlock( statusOFS, "Energy" );
    //statusOFS 
    //  << "NOTE:  Ecor  = Exc + Exx - EVxc - Ehart - Eself + Evdw" << std::endl
    //  << "       Etot  = Ekin + Ecor" << std::endl
    //  << "       Efree = Etot + Entropy" << std::endl << std::endl;
    Print(statusOFS, "! EfreeHarris     = ",  EfreeHarris_, "[au]");
    Print(statusOFS, "! Etot            = ",  Etot_, "[au]");
    Print(statusOFS, "! Exc             = ",  Exc_, "[au]");
    Print(statusOFS, "! Exx             = ",  Ehfx_, "[au]");
    Print(statusOFS, "! Efree           = ",  Efree_, "[au]");
    Print(statusOFS, "! Evdw            = ",  Evdw_, "[au]"); 
    Print(statusOFS, "! Fermi           = ",  fermi_, "[au]");
    Print(statusOFS, "! HOMO            = ",  HOMO*au2ev, "[eV]");
    if( hamDG.NumExtraState() > 0 ){
      Print(statusOFS, "! LUMO            = ",  LUMO*au2ev, "[eV]");
      Print(statusOFS, "! Band Gap        = ",  EG*au2ev, "[eV]");
    }

    statusOFS << std::endl << "  Convergence information : " << std::endl;
    Print(statusOFS, "! norm(out-in)/norm(in) = ",  scfOuterNorm_ ); 
    Print(statusOFS, "! Efree diff per atom   = ",  efreeDifPerAtom_, "[au]"); 

    if(useEnergySCFconvergence_ == 1)
    {
      Print(statusOFS, "! MD SCF Etot diff (per atom)  = ",  md_scf_etot_diff_, "[au]"); 
      Print(statusOFS, "! MD SCF Eband diff (per atom) = ",  md_scf_eband_diff_, "[au]"); 
    }
  }

  // Print out the force
  PrintBlock( statusOFS, "Atomic Force" );

  Point3 forceCM(0.0, 0.0, 0.0);
  std::vector<Atom>& atomList = hamDG.AtomList();
  Int numAtom = atomList.size();

  for( Int a = 0; a < numAtom; a++ ){
    Print( statusOFS, "atom", a, "force", atomList[a].force );
    forceCM += atomList[a].force;
  }
  statusOFS << std::endl;
  Print( statusOFS, "force for centroid  : ", forceCM );
  Print( statusOFS, "Max force magnitude : ", MaxForce(atomList) );
  statusOFS << std::endl;

  // *********************************************************************
  // Output information
  // *********************************************************************

  // Output the atomic structure, and other information for describing
  // density, basis functions etc.
  // 
  // Only mpirank == 0 works on this

  Real timeOutputSta, timeOutputEnd;
  GetTime( timeOutputSta );

  if( mpirank == 0 ){
    std::ostringstream structStream;
    statusOFS << std::endl 
      << "Output the structure information" 
      << std::endl;
    // Domain
    serialize( domain_.length, structStream, NO_MASK );
    serialize( domain_.numGrid, structStream, NO_MASK );
    serialize( domain_.numGridFine, structStream, NO_MASK );
    serialize( domain_.posStart, structStream, NO_MASK );
    serialize( numElem_, structStream, NO_MASK );

    // Atomic information
    serialize( hamDG.AtomList(), structStream, NO_MASK );
    std::string structFileName = "STRUCTURE";

    std::ofstream fout(structFileName.c_str());
    if( !fout.good() ){
      std::ostringstream msg;
      msg 
        << "File " << structFileName.c_str() << " cannot be opened." 
        << std::endl;
      ErrorHandling( msg.str().c_str() );
    }
    fout << structStream.str();
    fout.close();
  }

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          if( esdfParam.isOutputDensity ){
            if( mpirankRow == 0 ){
              statusOFS << std::endl 
                << "Output the electron density on the global grid" 
                << std::endl;
              // Output the wavefunctions on the uniform grid
              {
                std::ostringstream rhoStream;      

                NumTns<std::vector<DblNumVec> >& uniformGridElem =
                  hamDG.UniformGridElemFine();
                std::vector<DblNumVec>& grid = hamDG.UniformGridElemFine()(i, j, k);
                for( Int d = 0; d < DIM; d++ ){
                  serialize( grid[d], rhoStream, NO_MASK );
                }

                serialize( key, rhoStream, NO_MASK );
                serialize( hamDG.Density().LocalMap()[key], rhoStream, NO_MASK );

                SeparateWrite( restartDensityFileName_, rhoStream, mpirankCol );
              }

              // Output the wavefunctions on the LGL grid
              if(0)
              {
                std::ostringstream rhoStream;      

                // Generate the uniform mesh on the extended element.
                std::vector<DblNumVec>& gridpos = hamDG.LGLGridElem()(i,j,k);
                for( Int d = 0; d < DIM; d++ ){
                  serialize( gridpos[d], rhoStream, NO_MASK );
                }
                serialize( key, rhoStream, NO_MASK );
                serialize( hamDG.DensityLGL().LocalMap()[key], rhoStream, NO_MASK );
                SeparateWrite( "DENLGL", rhoStream, mpirankCol );
              }

            } // if( mpirankRow == 0 )
          }

          // Output potential in extended element, and only mpirankRow
          // == 0 does the job of for each element.
          if( esdfParam.isOutputPotExtElem ) {
            if( mpirankRow == 0 ){
              statusOFS 
                << std::endl 
                << "Output the total potential and external potential in the extended element."
                << std::endl;
              std::ostringstream potStream;      
              EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];

              // Generate the uniform mesh on the extended element.
              //              std::vector<DblNumVec> gridpos;
              //              UniformMeshFine ( eigSol.FFT().domain, gridpos );
              //              for( Int d = 0; d < DIM; d++ ){
              //                serialize( gridpos[d], potStream, NO_MASK );
              //              }


              serialize( key, potStream, NO_MASK );
              serialize( eigSol.Ham().Vtot(), potStream, NO_MASK );
              serialize( eigSol.Ham().Vext(), potStream, NO_MASK );
              SeparateWrite( "POTEXT", potStream, mpirankCol );
            } // if( mpirankRow == 0 )
          }

          // Output wavefunction in the extended element.  All processors participate
          if( esdfParam.isOutputWfnExtElem )
          {
            statusOFS 
              << std::endl 
              << "Output the wavefunctions in the extended element."
              << std::endl;

            EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
            std::ostringstream wavefunStream;      

            // Generate the uniform mesh on the extended element.
            // NOTE 05/06/2015: THIS IS NOT COMPATIBLE WITH THAT OF THE ALB2DEN!!
            std::vector<DblNumVec> gridpos;
            UniformMesh ( eigSol.FFT().domain, gridpos );
            for( Int d = 0; d < DIM; d++ ){
              serialize( gridpos[d], wavefunStream, NO_MASK );
            }

            serialize( key, wavefunStream, NO_MASK );
            serialize( eigSol.Psi().Wavefun(), wavefunStream, NO_MASK );
            SeparateWrite( restartWfnFileName_, wavefunStream, mpirank);
          }

          // Output wavefunction in the element on LGL grid. All processors participate.
          if( esdfParam.isOutputALBElemLGL )
          {
            statusOFS 
              << std::endl 
              << "Output the wavefunctions in the element on a LGL grid."
              << std::endl;
            // Output the wavefunctions in the extended element.
            std::ostringstream wavefunStream;      
            // Generate the uniform mesh on the extended element.
            std::vector<DblNumVec>& gridpos = hamDG.LGLGridElem()(i,j,k);
            for( Int d = 0; d < DIM; d++ ){
              serialize( gridpos[d], wavefunStream, NO_MASK );
            }
            serialize( key, wavefunStream, NO_MASK );
            serialize( hamDG.BasisLGL().LocalMap()[key], wavefunStream, NO_MASK );
            serialize( hamDG.LGLWeight3D(), wavefunStream, NO_MASK );
            SeparateWrite( "ALBLGL", wavefunStream, mpirank );
          }

          // Output wavefunction in the element on uniform fine grid.
          // All processors participate
          // NOTE: 
          // Since interpolation needs to be performed, this functionality can be slow.
          if( esdfParam.isOutputALBElemUniform )
          {
            statusOFS 
              << std::endl 
              << "Output the wavefunctions in the element on a fine LGL grid."
              << std::endl;
            // Output the wavefunctions in the extended element.
            std::ostringstream wavefunStream;      

            // Generate the uniform mesh on the extended element.
            serialize( key, wavefunStream, NO_MASK );
            DblNumMat& basisLGL = hamDG.BasisLGL().LocalMap()[key];
            DblNumMat basisUniformFine( 
                hamDG.NumUniformGridElemFine().prod(), 
                basisLGL.n() );
            SetValue( basisUniformFine, 0.0 );

            DblNumMat basisUniform(
                hamDG.NumUniformGridElem().prod(),
                basisLGL.n() );
            SetValue( basisUniform, 0.0 );

            for( Int g = 0; g < basisLGL.n(); g++ ){
              hamDG.InterpLGLToUniform(
                  hamDG.NumLGLGridElem(),
                  hamDG.NumUniformGridElemFine(),
                  basisLGL.VecData(g),
                  basisUniformFine.VecData(g) );

              hamDG.InterpLGLToUniform2(
                  hamDG.NumLGLGridElem(),
                  hamDG.NumUniformGridElem(),
                  basisLGL.VecData(g),
                  basisUniform.VecData(g) );
            }

            DblNumTns&   LGLWeight3D = hamDG.LGLWeight3D();
            DblNumMat basisTemp (basisLGL.m(), basisLGL.n());
            SetValue( basisTemp, 0.0 );
            Real factor = domain_.Volume() / domain_.NumGridTotalFine();
            Real factor2 = domain_.Volume() / domain_.NumGridTotal();

            // This is the same as the FourDotProduct process.
            for( Int g = 0; g < basisLGL.n(); g++ ){
              Real *ptr1 = LGLWeight3D.Data();
              Real *ptr2 = basisLGL.VecData(g);
              Real *ptr3 = basisTemp.VecData(g);
              for( Int l = 0; l < basisLGL.m(); l++ ){
                *(ptr3++) = (*(ptr1++)) * (*(ptr2++)) ;
              }
            }

            DblNumMat Smat(basisLGL.n(), basisLGL.n() );
            SetValue( Smat, 0.0 );
            DblNumMat Smat2(basisLGL.n(), basisLGL.n() );
            SetValue( Smat2, 0.0 );
            DblNumMat Smat3(basisLGL.n(), basisLGL.n() );
            SetValue( Smat3, 0.0 );

            blas::Gemm( 'T', 'N',basisLGL.n() , basisLGL.n(), basisLGL.m(),
                    1.0, basisLGL.Data(), basisLGL.m(),
                    basisTemp.Data(), basisLGL.m(), 0.0,
                    Smat.Data(), basisLGL.n() );

            blas::Gemm( 'T', 'N',basisUniformFine.n() , basisUniformFine.n(), basisUniformFine.m(),
                    factor, basisUniformFine.Data(), basisUniformFine.m(),
                    basisUniformFine.Data(), basisUniformFine.m(), 0.0,
                    Smat2.Data(), basisUniformFine.n() );

            blas::Gemm( 'T', 'N',basisUniform.n() , basisUniform.n(), basisUniform.m(),
                    factor2, basisUniform.Data(), basisUniform.m(),
                    basisUniform.Data(), basisUniform.m(), 0.0,
                    Smat3.Data(), basisUniform.n() );

            for( Int p = 0; p < basisLGL.n(); p++ ){
                statusOFS << " p " << p << " q " << p << std::endl;
                statusOFS << " SmatLGL " << Smat(p,p) << " SmatUniformFine " <<  Smat2(p,p) << " SmatUniform " <<  Smat3(p,p)  << std::endl;
            }

            // Generate the uniform mesh on the extended element.
            // NOTE 05/06/2015: THIS IS NOT COMPATIBLE WITH THAT OF THE ALB2DEN!!
            std::vector<DblNumVec> gridpos;
            EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
            // UniformMeshFine ( eigSol.FFT().domain, gridpos );
            // for( Int d = 0; d < DIM; d++ ){
            //   serialize( gridpos[d], wavefunStream, NO_MASK );
            // }

            serialize( key, wavefunStream, NO_MASK );
            serialize( basisUniformFine, wavefunStream, NO_MASK );
            SeparateWrite( "ALBUNIFORM", wavefunStream, mpirank );
          }
          // Output the eigenvector coefficients and only
          // mpirankRow == 0 does the job of for each element.
          // This option is only valid for diagonalization
          // methods
          if( esdfParam.isOutputEigvecCoef && solutionMethod_ == "diag" ) {
            if( mpirankRow == 0 ){
              statusOFS << std::endl 
                << "Output the eigenvector coefficients after diagonalization."
                << std::endl;
              std::ostringstream eigvecStream;      
              DblNumMat& eigvecCoef = hamDG.EigvecCoef().LocalMap()[key];

              serialize( key, eigvecStream, NO_MASK );
              serialize( eigvecCoef, eigvecStream, NO_MASK );
              SeparateWrite( "EIGVEC", eigvecStream, mpirankCol );
            } // if( mpirankRow == 0 )
          }

        } // (own this element)
  } // for (i)

  GetTime( timeOutputEnd );
  statusOFS << std::endl 
    << "Time for outputing data is = " << timeOutputEnd - timeOutputSta
    << " [s]" << std::endl;

  return;
}         // -----  end of method SCFDG::Iterate  -----

void
SCFDG::InnerIterate    ( Int outerIter )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );
  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  Real timeSta, timeEnd;
  Real timeIterStart, timeIterEnd;

  HamiltonianDG&  hamDG = *hamDGPtr_;

  bool isInnerSCFConverged = false;

  // The first inner iteration does not update the potential, and
  // construct the global Hamiltonian matrix from scratch
  GetTime(timeSta);

  // *********************************************************************
  // Enter the inner SCF iterations with fixed ALBs.
  // This is similar with SIESTA/CP2K/FHI-aims, but the outer SCF 
  // iteration is not needed since the basis set is given and fixed.
  // A mixing scheme is required to accelerate the SCF convergence.
  // Now we add Anderson/Broyden mixing for density, potential,
  // density matrix and Hamiltonian.
  // Hybrid DFT can only uses density matrix and Hamiltonian mixing.
  // *********************************************************************

  if( hamDG.IsHybrid() && hamDG.IsEXXActive() ) {
    scfInnerMaxIter_   = HybridscfInnerMaxIter_;
    scfInnerTolerance_ = HybridscfInnerTolerance_;
    InnermixVariable_  = HybridInnermixVariable_;
    mixType_ = HybridmixType_;  
  }
  else {
    scfInnerMaxIter_   = DFTscfInnerMaxIter_;
    scfInnerTolerance_ = DFTscfInnerTolerance_;
  }

  statusOFS << "  InnermixVariable_ "<< InnermixVariable_ << std::endl;
  statusOFS << " mixType_ " << mixType_ << std::endl;

  for( Int innerIter = 1; innerIter <= scfInnerMaxIter_; innerIter++ ){
    if ( isInnerSCFConverged ) break;
    scfTotalInnerIter_++;

    GetTime( timeIterStart );
    statusOFS << std::endl << "Inner SCF iteration #"  
      << innerIter << " starts." << std::endl << std::endl;

    // *********************************************************************
    // Update potential and construct/update the DG matrix
    // *********************************************************************

    if( innerIter == 1 ){
      // The first inner iteration does not update the potential, and
      // construct the global Hamiltonian matrix from scratch
      GetTime(timeSta);
      hamDG.CalculateDGMatrix( );
      GetTime( timeEnd );
      statusOFS << "Time for constructing the DG matrix is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;

      if( hamDG.IsEXXActive()){
        if(esdfParam.isDGHFISDF){
          hamDG.CalculateDGHFXMatrix_ISDF( Ehfx_, distDMMat_ );
        }
        else{
          hamDG.CalculateDGHFXMatrix( Ehfx_, distDMMat_ );
        }
        GetTime( timeEnd );
        statusOFS << "InnerSCF: Time for constructing the DGHFX matrix is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
      }// End if ( hamDG.IsHybrid() )
    }
    else{
      // The consequent inner iterations update the potential in the
      // element, and only update the global Hamiltonian matrix

      // Update the potential in the element (and the extended element)

      GetTime(timeSta);
//      // Save the old potential on the LGL grid
//      for( Int k = 0; k < numElem_[2]; k++ )
//        for( Int j = 0; j < numElem_[1]; j++ )
//          for( Int i = 0; i < numElem_[0]; i++ ){
//            Index3 key( i, j, k );
//            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
//              Index3 numLGLGrid     = hamDG.NumLGLGridElem();
//              blas::Copy( numLGLGrid.prod(),
//                  hamDG.VtotLGL().LocalMap()[key].Data(), 1,
//                  vtotLGLSave_.LocalMap()[key].Data(), 1 );
//            } // if (own this element)
//      } // for (i)

      // Update the local potential on the extended element and on the
      // element.
      UpdateElemLocalPotential();

      // Save the difference of the potential on the LGL grid into vtotLGLSave_
      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              Index3 numLGLGrid     = hamDG.NumLGLGridElem();
              Real *ptrNew = hamDG.VtotLGL().LocalMap()[key].Data();
              Real *ptrDif = vtotLGLSave_.LocalMap()[key].Data();
              for( Int p = 0; p < numLGLGrid.prod(); p++ ){
                (*ptrDif) = (*ptrNew) - (*ptrDif);
                ptrNew++;
                ptrDif++;
              }
            } // if (own this element)
          } // for (i)

      GetTime( timeEnd );
      statusOFS << "InnerSCF:: Time for updating the local potential in the extended element and the element is " <<
        timeEnd - timeSta << " [s]" << std::endl;

      // Update the DG Matrix
//      if( !hamDG.IsEXXActive() ) {
        GetTime(timeSta);
        hamDG.UpdateDGMatrix( vtotLGLSave_ );
        MPI_Barrier( domain_.comm );
        GetTime( timeEnd );
        statusOFS << "Time for updating the DG matrix is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
      //}
      // Recalculate DG Matrix
      if( hamDG.IsHybrid() && hamDG.IsEXXActive()){
//        GetTime(timeSta);
//        hamDG.CalculateDGMatrix( );
//        MPI_Barrier( domain_.comm );
//        GetTime( timeEnd );
//        statusOFS << "Time for recalculating the DG matrix is " <<
//          timeEnd - timeSta << " [s]" << std::endl << std::endl;
//
        GetTime( timeSta );

        if(esdfParam.isDGHFISDF){
          hamDG.CalculateDGHFXMatrix_ISDF( Ehfx_, distDMMat_ );
        }
        else{
          hamDG.CalculateDGHFXMatrix( Ehfx_, distDMMat_ );
        }

        GetTime( timeEnd );
        statusOFS << "InnerSCF: Time for constructing the DGHFX matrix is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
      }// End if ( hamDG.IsHybrid() )
    } // InnerIter > 1

    // *********************************************************************
    // Write the Hamiltonian matrix to a file (if needed) 

    if( esdfParam.isOutputHMatrix ){
      // Only the first processor column participates in the conversion
      if( mpirankRow == 0 ){
        DistSparseMatrix<Real>  HSparseMat;

        GetTime(timeSta);
        DistElemMatToDistSparseMat( 
            hamDG.HMat(),
            hamDG.NumBasisTotal(),
            HSparseMat,
            hamDG.ElemBasisIdx(),
            domain_.colComm );
        GetTime(timeEnd);
        statusOFS << "InnerSCF: Time for converting the DG matrix to DistSparseMatrix format is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;

        GetTime(timeSta);
        ParaWriteDistSparseMatrix( "H.csc", HSparseMat );
              //            WriteDistSparseMatrixFormatted( "H.matrix", HSparseMat );
        GetTime(timeEnd);
        statusOFS << "InnerSCF: Time for writing the matrix in parallel is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
      }

      MPI_Barrier( domain_.comm );
    }

    // *********************************************************************
    //  Save the mixing variable first
    {
      // Save the old potential on the LGL grid
      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              Index3 numLGLGrid     = hamDG.NumLGLGridElem();
              blas::Copy( numLGLGrid.prod(),
                  hamDG.VtotLGL().LocalMap()[key].Data(), 1,
                  vtotLGLSave_.LocalMap()[key].Data(), 1 );
              //statusOFS << " up Vtot1 " << hamDG.VtotLGL().LocalMap()[key] << std::endl;

            } // if (own this element)
      } // for (i)

      if( InnermixVariable_ == "density" || InnermixVariable_ == "potential" ){
        for( Int k = 0; k < numElem_[2]; k++ )
          for( Int j = 0; j < numElem_[1]; j++ )
            for( Int i = 0; i < numElem_[0]; i++ ){
              Index3 key( i, j, k );
              if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                if( InnermixVariable_ == "density" ){
                  DblNumVec& oldVec = hamDG.Density().LocalMap()[key];
                  mixInnerSave_.LocalMap()[key] = oldVec;
                }
                else if( InnermixVariable_ == "potential" ){
                  DblNumVec& oldVec = hamDG.Vtot().LocalMap()[key];
                  mixInnerSave_.LocalMap()[key] = oldVec;
                }
              } // own this element
        } // for (i)
      }
      else if( InnermixVariable_ == "hamiltonian" || InnermixVariable_ == "densitymatrix" ){
        if( InnermixVariable_ == "hamiltonian" ){
          for(typename std::map<ElemMatKey, DblNumMat >::iterator
            Ham_iterator = hamDG.HMat().LocalMap().begin();
            Ham_iterator != hamDG.HMat().LocalMap().end();
            ++ Ham_iterator )
          {
            ElemMatKey matkey = (*Ham_iterator).first;
            DblNumMat& oldMat = hamDG.HMat().LocalMap()[matkey];
            std::map<ElemMatKey, DblNumMat>::iterator mi =
              distmixInnerSave_.LocalMap().find( matkey );
              if( mi == distmixInnerSave_.LocalMap().end() ){
                distmixInnerSave_.LocalMap()[matkey] = oldMat;
              }
              else{
                DblNumMat&  mat = (*mi).second;
                blas::Copy( mat.Size(), oldMat.Data(), 1,
                  mat.Data(), 1);
              }
          }
        }
        else if ( InnermixVariable_ == "densitymatrix" ){
            for(typename std::map<ElemMatKey, DblNumMat >::iterator
              DM_iterator =  distDMMat_.LocalMap().begin();
              DM_iterator !=  distDMMat_.LocalMap().end();
              ++ DM_iterator ) 
            {
              ElemMatKey matkey = (*DM_iterator).first;
              DblNumMat& oldMat = distDMMat_.LocalMap()[matkey];
              std::map<ElemMatKey, DblNumMat>::iterator mi =
                distmixInnerSave_.LocalMap().find( matkey );
                if( mi == distmixInnerSave_.LocalMap().end() ){
                  distmixInnerSave_.LocalMap()[matkey] = oldMat;
                }
                else{
                  DblNumMat&  mat = (*mi).second;
                  blas::Copy( mat.Size(), oldMat.Data(), 1,
                    mat.Data(), 1);
                }
            }
        } // ---- End of if( InnermixVariable_ == "hamiltonian" )
      }
    }

    // *********************************************************************
    // Solve the eigenvalue problem
    // Standard cubic-scaling DIAG/CheFSI methods for eigenpairs
    // or reduced-scaling algoriths fordensity matrix.
    // *********************************************************************

    // Method 1: Using diagonalization method
    // With a versatile choice of processors for using ScaLAPACK.
    // Or using Chebyshev filtering

    if( solutionMethod_ == "diag" ){
      
      if(Diag_SCFDG_by_Cheby_ == 1 ){
        // Chebyshev filtering based diagonalization
        GetTime(timeSta);
        if(scfdg_ion_dyn_iter_ != 0){
          if(SCFDG_use_comp_subspace_ == 1){
            if((scfdg_ion_dyn_iter_ % SCFDG_CS_ioniter_regular_cheby_freq_ == 0)
               && (outerIter <= Second_SCFDG_ChebyOuterIter_ / 2)){
              // Just some adhoc criterion used here
              // Usual CheFSI to help corrrect drift / SCF convergence
#if ( _DEBUGlevel_ >= 1 )                
              statusOFS << std::endl << " Calling Second stage Chebyshev Iter in iondynamics step to improve drift / SCF convergence ..." << std::endl;    
#endif
              scfdg_GeneralChebyStep(Second_SCFDG_ChebyCycleNum_, Second_SCFDG_ChebyFilterOrder_);

              SCFDG_comp_subspace_engaged_ = 0;
            }
            else{  
            // Decide serial or parallel version here
              if(SCFDG_comp_subspace_parallel_ == 0){  
#if ( _DEBUGlevel_ >= 1 )
                statusOFS << std::endl << " Calling Complementary Subspace Strategy (serial subspace version) ...  " << std::endl;
#endif
                scfdg_complementary_subspace_serial(General_SCFDG_ChebyFilterOrder_);
              }
              else{
#if ( _DEBUGlevel_ >= 1 )
                statusOFS << std::endl << " Calling Complementary Subspace Strategy (parallel subspace version) ...  " << std::endl;
#endif
                scfdg_complementary_subspace_parallel(General_SCFDG_ChebyFilterOrder_);                     
              }
              // Set the engaged flag 
              SCFDG_comp_subspace_engaged_ = 1;
            }
          } // if(SCFDG_use_comp_subspace_ == 1)
          else{
            // Just some adhoc criterion used here
            if(outerIter <= Second_SCFDG_ChebyOuterIter_ / 2){
            // Need to re-use current guess, so do not call the first Cheby step
#if ( _DEBUGlevel_ >= 1 )              
              statusOFS << std::endl << " Calling Second stage Chebyshev Iter in iondynamics step " << std::endl;         
#endif
              scfdg_GeneralChebyStep(Second_SCFDG_ChebyCycleNum_, Second_SCFDG_ChebyFilterOrder_);
            }
            else{     
            // Subsequent MD Steps
#if ( _DEBUGlevel_ >= 1 )              
              statusOFS << std::endl << " Calling General Chebyshev Iter in iondynamics step " << std::endl;
#endif
              scfdg_GeneralChebyStep(General_SCFDG_ChebyCycleNum_, General_SCFDG_ChebyFilterOrder_); 
            }
          } //if(SCFDG_use_comp_subspace_ != 1)
        } // if (scfdg_ion_dyn_iter_ != 0)
        else{    
          // 0th MD / Geometry Optimization step (or static calculation)     
          if(outerIter == 1){
#if ( _DEBUGlevel_ >= 1 )
            statusOFS << std::endl << " Calling First Chebyshev Iter  " << std::endl;
#endif
            scfdg_FirstChebyStep(First_SCFDG_ChebyCycleNum_, First_SCFDG_ChebyFilterOrder_);
          }
          else if(outerIter > 1 && outerIter <= Second_SCFDG_ChebyOuterIter_){
#if ( _DEBUGlevel_ >= 1 )
            statusOFS << std::endl << " Calling Second Stage Chebyshev Iter  " << std::endl;
#endif
            scfdg_GeneralChebyStep(Second_SCFDG_ChebyCycleNum_, Second_SCFDG_ChebyFilterOrder_);
          }
          else{  
            if(SCFDG_use_comp_subspace_ == 1){
            // Decide serial or parallel version here
              if(SCFDG_comp_subspace_parallel_ == 0){  
#if ( _DEBUGlevel_ >= 1 )
                statusOFS << std::endl << " Calling Complementary Subspace Strategy (serial subspace version)  " << std::endl;
#endif
                scfdg_complementary_subspace_serial(General_SCFDG_ChebyFilterOrder_);
              }
              else{
#if ( _DEBUGlevel_ >= 1 )
                statusOFS << std::endl << " Calling Complementary Subspace Strategy (parallel subspace version)  " << std::endl;
#endif
                scfdg_complementary_subspace_parallel(General_SCFDG_ChebyFilterOrder_);       
              }
              // Now set the engaged flag 
              SCFDG_comp_subspace_engaged_ = 1;
            }
            else{
#if ( _DEBUGlevel_ >= 1 )                  
              statusOFS << std::endl << " Calling General Chebyshev Iter  " << std::endl;
#endif
              scfdg_GeneralChebyStep(General_SCFDG_ChebyCycleNum_, General_SCFDG_ChebyFilterOrder_); 
            }
          }
        } // end of if(scfdg_ion_dyn_iter_ != 0)

        GetTime( timeEnd );

        if(SCFDG_comp_subspace_engaged_ == 1){
          statusOFS << std::endl << " Total time for Complementary Subspace Method is " << timeEnd - timeSta << " [s]" << std::endl << std::endl;
        }
        else{
                  statusOFS << std::endl << " Total time for diag DG matrix via Chebyshev filtering is " << timeEnd - timeSta << " [s]" << std::endl << std::endl;
        }
        DblNumVec& eigval = hamDG.EigVal();          

      } // if(Diag_SCFDG_by_Cheby_ == 1 )
      else // DIAG :: call the ELSI interface and old Scalapack interface 
      {
        GetTime(timeSta);
        Int sizeH = hamDG.NumBasisTotal(); // used for the size of Hamitonian. 
        DblNumVec& eigval = hamDG.EigVal(); 
        eigval.Resize( hamDG.NumStateTotal() );        

        for( Int k = 0; k < numElem_[2]; k++ )
          for( Int j = 0; j < numElem_[1]; j++ )
            for( Int i = 0; i < numElem_[0]; i++ ){
              Index3 key( i, j, k );
              if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                const std::vector<Int>&  idx = hamDG.ElemBasisIdx()(i, j, k); 
                DblNumMat& localCoef  = hamDG.EigvecCoef().LocalMap()[key];
                localCoef.Resize( idx.size(), hamDG.NumStateTotal() );        
              }
            } 

        scalapack::Descriptor descH;
        if( contxt_ >= 0 ){
          descH.Init( sizeH, sizeH, scaBlockSize_, scaBlockSize_, 
              0, 0, contxt_ );
        }

        scalapack::ScaLAPACKMatrix<Real>  scaH, scaZ;

        std::vector<Int> mpirankElemVec(dmCol_);
        std::vector<Int> mpirankScaVec( numProcScaLAPACK_ );

        // The processors in the first column are the source
        for( Int i = 0; i < dmCol_; i++ ){
          mpirankElemVec[i] = i * dmRow_;
        }
        // The first numProcScaLAPACK processors are the target
        for( Int i = 0; i < numProcScaLAPACK_; i++ ){
          mpirankScaVec[i] = i;
        }

#if ( _DEBUGlevel_ >= 2 )
        statusOFS << "mpirankElemVec = " << mpirankElemVec << std::endl;
        statusOFS << "mpirankScaVec = " << mpirankScaVec << std::endl;
#endif

        Real timeConversionSta, timeConversionEnd;

        GetTime( timeConversionSta );
        DistElemMatToScaMat2( hamDG.HMat(), descH,
            scaH, hamDG.ElemBasisIdx(), domain_.comm,
            domain_.colComm, mpirankElemVec,
            mpirankScaVec );
        GetTime( timeConversionEnd );

#if ( _DEBUGlevel_ >= 1 )
        statusOFS << "Time for converting from DistElemMat to ScaMat is " <<
          timeConversionEnd - timeConversionSta << " [s]" << std::endl << std::endl;
#endif
        if(contxt_ >= 0){
          std::vector<Real> eigs(sizeH);
          double * Smatrix = NULL;
          GetTime( timeConversionSta );

          // allocate memory for the scaZ. and call ELSI: ELPA
          if( diagSolutionMethod_ == "scalapack"){
            scalapack::Syevd('U', scaH, eigs, scaZ);
          }
          else // by default to use ELPA
          {
#ifdef ELSI
            scaZ.SetDescriptor(scaH.Desc());
            c_elsi_ev_real(scaH.Data(), Smatrix, &eigs[0], scaZ.Data()); 
#endif
          }
                
          GetTime( timeConversionEnd );
#if ( _DEBUGlevel_ >= 2 )
          if( diagSolutionMethod_ == "scalapack"){
             statusOFS << "InnerSCF: Time for Scalapack::diag " <<
               timeConversionEnd - timeConversionSta << " [s]" 
               << std::endl << std::endl;
          }
          else{
            statusOFS << "InnerSCF: Time for ELSI::ELPA  Diag " <<
              timeConversionEnd - timeConversionSta << " [s]" 
              << std::endl << std::endl;
          }
#endif
          for( Int i = 0; i < hamDG.NumStateTotal(); i++ ){
            eigval[i] = eigs[i];
          }
        } //if(contxt_ >= -1)

        GetTime( timeConversionSta );
        ScaMatToDistNumMat2( scaZ, hamDG.Density().Prtn(), 
            hamDG.EigvecCoef(), hamDG.ElemBasisIdx(), domain_.comm,
            domain_.colComm, mpirankElemVec, mpirankScaVec, 
            hamDG.NumStateTotal() );
        GetTime( timeConversionEnd );
#if ( _DEBUGlevel_ >= 1 )
        statusOFS << "Time for converting from ScaMat to DistNumMat is " <<
          timeConversionEnd - timeConversionSta << " [s]" << std::endl << std::endl;
#endif
        GetTime( timeConversionSta );
        for( Int k = 0; k < numElem_[2]; k++ )
          for( Int j = 0; j < numElem_[1]; j++ )
            for( Int i = 0; i < numElem_[0]; i++ ){
              Index3 key( i, j, k );
              if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                DblNumMat& localCoef  = hamDG.EigvecCoef().LocalMap()[key];
                MPI_Bcast(localCoef.Data(), localCoef.m() * localCoef.n(), 
                    MPI_DOUBLE, 0, domain_.rowComm);
              }
        } 
        GetTime( timeConversionEnd );
#if ( _DEBUGlevel_ >= 1 )
        statusOFS << "Time for MPI_Bcast eigval and localCoef is " <<
          timeConversionEnd - timeConversionSta << " [s]" << std::endl << std::endl;
#endif

        MPI_Barrier( domain_.comm );
        MPI_Barrier( domain_.rowComm );
        MPI_Barrier( domain_.colComm );

        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        if( diagSolutionMethod_ == "scalapack"){
          statusOFS << "InnerSCF: Time for diag DG matrix via ScaLAPACK is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
        }
        else{
          statusOFS << "InnerSCF: Time for diag DG matrix via ELSI:ELPA is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
        }
#endif

        // Communicate the eigenvalues
        Int mpirankScaSta = mpirankScaVec[0];
        MPI_Bcast(eigval.Data(), hamDG.NumStateTotal(), MPI_DOUBLE, 
           mpirankScaVec[0], domain_.comm);

      } // End of ELSI

    // Post processing

    Evdw_ = 0.0;

    if(SCFDG_comp_subspace_engaged_ == 1)
    {
      // Calculate Harris energy without computing the occupations
      CalculateHarrisEnergy();
    }        
    else{
     // Compute the occupation rate - specific smearing types dealt with within this function
      statusOFS << "Call for Full DM After DIAG " << std::endl;
      CalculateOccupationRate( hamDG.EigVal(), hamDG.OccupationRate() );

      if( InnermixVariable_ == "densitymatrix"  ||  hamDG.IsEXXActive() ) {
        GetTime(timeSta);
        scfdg_compute_fullDM();
        GetTime( timeEnd );
        statusOFS << "InnerSCF: Recalculate density matrix for diag method " << std::endl;
      }

#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "InnerSCF: DIAG Time for computing full density matrix " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      // Compute the Harris energy functional.  
      // NOTE: In computing the Harris energy, the density and the
      // potential must be the INPUT density and potential without ANY
      // update.
      GetTime(timeSta);
      CalculateHarrisEnergy();
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "InnerSCF: Time for computing harris energy " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    }

    // Calculate the new electron density

    GetTime( timeSta );

    if(SCFDG_comp_subspace_engaged_ == 1){
      // Density calculation for complementary subspace method
      statusOFS << std::endl << " Using complementary subspace method for electron density ... " << std::endl;
      Real GetTime_extra_sta, GetTime_extra_end;          
      Real GetTime_fine_sta, GetTime_fine_end;

      GetTime(GetTime_extra_sta);
      statusOFS << std::endl << " Forming diagonal blocks of density matrix : ";
      GetTime(GetTime_fine_sta);

      // Compute the diagonal blocks of the density matrix
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn> cheby_diag_dmat;  
      cheby_diag_dmat.Prtn()     = hamDG.HMat().Prtn();
      cheby_diag_dmat.SetComm(domain_.colComm);

      // Copy eigenvectors to temp bufer
      DblNumMat &eigvecs_local = (hamDG.EigvecCoef().LocalMap().begin())->second;

      DblNumMat temp_local_eig_vec;
      temp_local_eig_vec.Resize(eigvecs_local.m(), eigvecs_local.n());
      blas::Copy((eigvecs_local.m() * eigvecs_local.n()), eigvecs_local.Data(), 1, temp_local_eig_vec.Data(), 1);

      // First compute the X*X^T portion
      // Multiply out to obtain diagonal block of density matrix
      ElemMatKey diag_block_key = std::make_pair(my_cheby_eig_vec_key_, my_cheby_eig_vec_key_);
      cheby_diag_dmat.LocalMap()[diag_block_key].Resize( temp_local_eig_vec.m(),  temp_local_eig_vec.m());

      blas::Gemm( 'N', 'T', temp_local_eig_vec.m(), temp_local_eig_vec.m(), temp_local_eig_vec.n(),
          1.0, 
          temp_local_eig_vec.Data(), temp_local_eig_vec.m(), 
          temp_local_eig_vec.Data(), temp_local_eig_vec.m(),
          0.0, 
          cheby_diag_dmat.LocalMap()[diag_block_key].Data(),  temp_local_eig_vec.m());

      GetTime(GetTime_fine_end);
      statusOFS << std::endl << " X * X^T computed in " << (GetTime_fine_end - GetTime_fine_sta) << " s.";

      GetTime(GetTime_fine_sta);
      if(SCFDG_comp_subspace_N_solve_ != 0)
      {
        // Now compute the X * C portion
        DblNumMat XC_mat;
        XC_mat.Resize(eigvecs_local.m(), SCFDG_comp_subspace_N_solve_);

        blas::Gemm( 'N', 'N', temp_local_eig_vec.m(), SCFDG_comp_subspace_N_solve_, temp_local_eig_vec.n(),
                    1.0, 
                    temp_local_eig_vec.Data(), temp_local_eig_vec.m(), 
                    SCFDG_comp_subspace_matC_.Data(), SCFDG_comp_subspace_matC_.m(),
                    0.0, 
                    XC_mat.Data(),  XC_mat.m());

        // Subtract XC*XC^T from DM
        blas::Gemm( 'N', 'T', XC_mat.m(), XC_mat.m(), XC_mat.n(),
                    -1.0, 
                    XC_mat.Data(), XC_mat.m(), 
                    XC_mat.Data(), XC_mat.m(),
                    1.0, 
                    cheby_diag_dmat.LocalMap()[diag_block_key].Data(),  temp_local_eig_vec.m());
      }
      GetTime(GetTime_fine_end);
      statusOFS << std::endl << " X*C and XC * (XC)^T computed in " << (GetTime_fine_end - GetTime_fine_sta) << " s.";
      
      
      GetTime(GetTime_extra_end);
      statusOFS << std::endl << " Total time for computing diagonal blocks of DM = " << (GetTime_extra_end - GetTime_extra_sta)  << " s." << std::endl ;
      statusOFS << std::endl;

      // Make the call evaluate this on the real space grid 
      hamDG.CalculateDensityDM2(hamDG.Density(), hamDG.DensityLGL(), cheby_diag_dmat );
    } // (SCFDG_comp_subspace_engaged_ == 1)
    else {
      Int temp_m = hamDG.NumBasisTotal() / (numElem_[0] * numElem_[1] * numElem_[2]); // Average no. of ALBs per element
      Int temp_n = hamDG.NumStateTotal();
      if((Diag_SCFDG_by_Cheby_ == 1) && (temp_m < temp_n))
      {  
        statusOFS << std::endl << " Using alternate routine for electron density: " << std::endl;

        Real GetTime_extra_sta, GetTime_extra_end;                
        GetTime(GetTime_extra_sta);
        statusOFS << std::endl << " Forming diagonal blocks of density matrix ... ";

        // Compute the diagonal blocks of the density matrix
        DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn> cheby_diag_dmat;  
        cheby_diag_dmat.Prtn()     = hamDG.HMat().Prtn();
        cheby_diag_dmat.SetComm(domain_.colComm);

        // Copy eigenvectors to temp bufer
        DblNumMat &eigvecs_local = (hamDG.EigvecCoef().LocalMap().begin())->second;

        DblNumMat scal_local_eig_vec;
        scal_local_eig_vec.Resize(eigvecs_local.m(), eigvecs_local.n());
        blas::Copy((eigvecs_local.m() * eigvecs_local.n()), eigvecs_local.Data(), 1, scal_local_eig_vec.Data(), 1);

        // Scale temp buffer by occupation square root
        for(Int iter_scale = 0; iter_scale < eigvecs_local.n(); iter_scale ++)
        {
          blas::Scal(  scal_local_eig_vec.m(),  sqrt(hamDG.OccupationRate()[iter_scale]), scal_local_eig_vec.Data() + iter_scale * scal_local_eig_vec.m(), 1 );
        }

        // Multiply out to obtain diagonal block of density matrix
        ElemMatKey diag_block_key = std::make_pair(my_cheby_eig_vec_key_, my_cheby_eig_vec_key_);
        cheby_diag_dmat.LocalMap()[diag_block_key].Resize( scal_local_eig_vec.m(),  scal_local_eig_vec.m());

        blas::Gemm( 'N', 'T', scal_local_eig_vec.m(), scal_local_eig_vec.m(), scal_local_eig_vec.n(),
            1.0, 
            scal_local_eig_vec.Data(), scal_local_eig_vec.m(), 
            scal_local_eig_vec.Data(), scal_local_eig_vec.m(),
            0.0, 
            cheby_diag_dmat.LocalMap()[diag_block_key].Data(),  scal_local_eig_vec.m());

        GetTime(GetTime_extra_end);
        statusOFS << " Done. ( " << (GetTime_extra_end - GetTime_extra_sta)  << " s) " << std::endl ;

        // Make the call evaluate this on the real space grid 
        hamDG.CalculateDensityDM2(hamDG.Density(), hamDG.DensityLGL(), cheby_diag_dmat );
      }
      else
      {  
        // FIXME 
        // Do not need the conversion from column to row partition as well

        statusOFS << "Call CalculateDensity After DIAG " << std::endl;
//        statusOFS << "  hamDG.OccupationRate() " <<  hamDG.OccupationRate()  <<  std::endl;

        // xmqin 20240218 calculate density using different methods
        if( InnermixVariable_ == "densitymatrix"  ||  hamDG.IsEXXActive() ) {
          hamDG.CalculateDensityDM2(hamDG.Density(), hamDG.DensityLGL(), distDMMat_ );

          statusOFS << "InnerSCF: Recalculate density from density matrix" << std::endl;
        }
        else{
          
          hamDG.CalculateDensity( hamDG.Density(), hamDG.DensityLGL() );
        }

        // 2016/11/20: Add filtering of the density. Impacts
        // convergence at the order of 1e-5 for the LiH dimer
        // example and therefore is not activated
        if(0){
          DistFourier& fft = *distfftPtr_;
          Int ntot      = fft.numGridTotal;
          Int ntotLocal = fft.numGridLocal;

          DblNumVec  tempVecLocal;
          DistNumVecToDistRowVec(
              hamDG.Density(),
              tempVecLocal,
              domain_.numGridFine,
              numElem_,
              fft.localNzStart,
              fft.localNz,
              fft.isInGrid,
              domain_.colComm );

          if( fft.isInGrid ){
            for( Int i = 0; i < ntotLocal; i++ ){
              fft.inputComplexVecLocal(i) = Complex( 
                  tempVecLocal(i), 0.0 );
            }

            fftw_execute( fft.forwardPlan );

            // Filter out high frequency modes
            for( Int i = 0; i < ntotLocal; i++ ){
              if( fft.gkkLocal(i) > std::pow(densityGridFactor_,2.0) * ecutWavefunction_ ){
                fft.outputComplexVecLocal(i) = Z_ZERO;
              }
            }

            fftw_execute( fft.backwardPlan );


            for( Int i = 0; i < ntotLocal; i++ ){
              tempVecLocal(i) = fft.inputComplexVecLocal(i).real() / ntot;
            }
          }

          DistRowVecToDistNumVec( 
              tempVecLocal,
              hamDG.Density(),
              domain_.numGridFine,
              numElem_,
              fft.localNzStart,
              fft.localNz,
              fft.isInGrid,
              domain_.colComm );


          // Compute the sum of density and normalize again.
          Real sumRhoLocal = 0.0, sumRho = 0.0;
          for( Int k = 0; k < numElem_[2]; k++ )
            for( Int j = 0; j < numElem_[1]; j++ )
              for( Int i = 0; i < numElem_[0]; i++ ){
                Index3 key( i, j, k );
                if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                  DblNumVec& localRho = hamDG.Density().LocalMap()[key];

                  Real* ptrRho = localRho.Data();
                  for( Int p = 0; p < localRho.Size(); p++ ){
                    sumRhoLocal += ptrRho[p];
                  }
                }
              }

          sumRhoLocal *= domain_.Volume() / domain_.NumGridTotalFine(); 
          mpi::Allreduce( &sumRhoLocal, &sumRho, 1, MPI_SUM, domain_.colComm );

#if ( _DEBUGlevel_ >= 0 )
          statusOFS << std::endl;
          Print( statusOFS, "Sum Rho on uniform grid (after Fourier filtering) = ", sumRho );
          statusOFS << std::endl;
#endif
          Real fac = hamDG.NumSpin() * hamDG.NumOccupiedState() / sumRho;
          sumRhoLocal = 0.0, sumRho = 0.0;
          for( Int k = 0; k < numElem_[2]; k++ )
            for( Int j = 0; j < numElem_[1]; j++ )
              for( Int i = 0; i < numElem_[0]; i++ ){
                Index3 key( i, j, k );
                if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                  DblNumVec& localRho = hamDG.Density().LocalMap()[key];
                  blas::Scal(  localRho.Size(),  fac, localRho.Data(), 1 );

                  Real* ptrRho = localRho.Data();
                  for( Int p = 0; p < localRho.Size(); p++ ){
                    sumRhoLocal += ptrRho[p];
                  }
                }
          }

          sumRhoLocal *= domain_.Volume() / domain_.NumGridTotalFine(); 
          mpi::Allreduce( &sumRhoLocal, &sumRho, 1, MPI_SUM, domain_.colComm );

#if ( _DEBUGlevel_ >= 0 )
          statusOFS << " fac " << fac << std::endl;
          Print( statusOFS, "Sum Rho on uniform grid (after normalization again) = ", sumRho );
          statusOFS << std::endl;
#endif
        } //if(0)

      }// ((Diag_SCFDG_by_Cheby_ == 1) && (temp_m < temp_n))

    } // if(SCFDG_comp_subspace_engaged_ == 1)

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "InnerSCF: Time for computing density 1 in the global domain is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    // *******************************************************************************
    // Update Step :: Potential, KS and 2rd Energy
    //
    // Update the output potential, and the KS and second order accurate
    // energy
    if(InnermixVariable_ == "potential"){
      // Update the Hartree energy and the exchange correlation energy and
      // potential for computing the KS energy and the second order
      // energy.
      // NOTE Vtot should not be updated until finishing the computation
      // of the energies.

      if( isCalculateGradRho_  ){
        GetTime( timeSta );
        hamDG.CalculateGradDensity(  *distfftPtr_ );
        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for calculating gradient of density is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
      }

      GetTime( timeSta );
      hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), *distfftPtr_ );

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for computing Exc in the global domain is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      GetTime( timeSta );

      hamDG.CalculateHartree( hamDG.Vhart(), *distfftPtr_ );

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "InnerSCF: Time for computing Vhart in the global domain is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
      // Compute the second order accurate energy functional.
      // NOTE: In computing the second order energy, the density and the
      // potential must be the OUTPUT density and potential without ANY
      // MIXING.
      CalculateSecondOrderEnergy();
      // Compute the KS energy 
      GetTime( timeSta );

      CalculateKSEnergy();

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "InnerSCF: Time for computing KSEnergy in the global domain is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      // Update the total potential AFTER updating the energy

      // No external potential

      // Compute the new total potential

      GetTime( timeSta );

      hamDG.CalculateVtot( hamDG.Vtot() );

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "InnerSCF: Time for computing Vtot in the global domain is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    }
    // **********************************************************************************
    // Atomic Forces
    //
    // Compute the force at every step
    if( esdfParam.isCalculateForceEachSCF ){

      // Compute force
      GetTime( timeSta );

      if(SCFDG_comp_subspace_engaged_ == false)
      {
        if(1)
        {
          statusOFS << std::endl << "InnerSCF: Computing forces using eigenvectors ... " << std::endl;
          hamDG.CalculateForce( *distfftPtr_ );
        }
        else
        {         
          // Alternate (highly unusual) routine for debugging purposes
          // Compute the Full DM (from eigenvectors) and call the PEXSI force evaluator

          double extra_timeSta, extra_timeEnd;

          statusOFS << std::endl << "InnerSCF: Computing forces using Density Matrix ... ";
          statusOFS << std::endl << "InnerSCF: Computing full Density Matrix from eigenvectors ...";
          GetTime(extra_timeSta);

          distDMMat_.Prtn()     = hamDG.HMat().Prtn();

          // Compute the full DM 
          scfdg_compute_fullDM();

          GetTime(extra_timeEnd);

          statusOFS << std::endl << " Computation took " << (extra_timeEnd - extra_timeSta) << " s." << std::endl;

          // Call the PEXSI force evaluator
          hamDG.CalculateForceDM( *distfftPtr_, distDMMat_ );        
        }
      }
      else
      {
        double extra_timeSta, extra_timeEnd;

        statusOFS << std::endl << " Computing forces using Density Matrix ... ";

        statusOFS << std::endl << " Computing full Density Matrix for Complementary Subspace method ...";
        GetTime(extra_timeSta);

        // Compute the full DM in the complementary subspace method
        scfdg_complementary_subspace_compute_fullDM();

        GetTime(extra_timeEnd);

        statusOFS << std::endl << " DM Computation took " << (extra_timeEnd - extra_timeSta) << " s." << std::endl;

        // Call the PEXSI force evaluator
        hamDG.CalculateForceDM( *distfftPtr_, distDMMat_ );        

      } //  if(SCFDG_comp_subspace_engaged_ == false)

      GetTime( timeEnd );
      statusOFS << "Time for computing the force is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;

      // Print out the force
      // Only master processor output information containing all atoms
      if( mpirank == 0 ){
        PrintBlock( statusOFS, "Atomic Force" );
        {
          Point3 forceCM(0.0, 0.0, 0.0);
          std::vector<Atom>& atomList = hamDG.AtomList();
          Int numAtom = atomList.size();
          for( Int a = 0; a < numAtom; a++ ){
            Print( statusOFS, "atom", a, "force", atomList[a].force );
            forceCM += atomList[a].force;
          }
          statusOFS << std::endl;
          Print( statusOFS, "force for centroid: ", forceCM );
          statusOFS << std::endl;
        }
      } // output

    } //  if( esdfParam.isCalculateForceEachSCF )

    // Compute the a posteriori error estimator at every step
    // FIXME This is not used when intra-element parallelization is
    // used.
    if( esdfParam.isCalculateAPosterioriEachSCF && 0 )
    {
      GetTime( timeSta );
      DblNumTns  eta2Total, eta2Residual, eta2GradJump, eta2Jump;
      hamDG.CalculateAPosterioriError( 
          eta2Total, eta2Residual, eta2GradJump, eta2Jump );
      GetTime( timeEnd );
      statusOFS << "Time for computing the a posteriori error is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;

      // Only master processor output information containing all atoms
      if( mpirank == 0 ){
        PrintBlock( statusOFS, "A Posteriori error" );
        {
          statusOFS << std::endl << "Total a posteriori error:" << std::endl;
          statusOFS << eta2Total << std::endl;
          statusOFS << std::endl << "Residual term:" << std::endl;
          statusOFS << eta2Residual << std::endl;
          statusOFS << std::endl << "Jump of gradient term:" << std::endl;
          statusOFS << eta2GradJump << std::endl;
          statusOFS << std::endl << "Jump of function value term:" << std::endl;
          statusOFS << eta2Jump << std::endl;
        }
      }
    }
    // Atomic Forces
    //**********************************************************************************
  } // IF (DIAG) 

  // Method 2: Using the pole expansion and selected inversion (PEXSI) method
  // FIXME Currently it is assumed that all processors used by DG will be used by PEXSI.
#ifdef _USE_PEXSI_
  // The following version is with intra-element parallelization
  DistDblNumVec VtotHist; // check check
  // check check
  Real difNumElectron = 0.0;
  if( solutionMethod_ == "pexsi" ){
    // Initialize the history of vtot , check check
    for( Int k=0; k< numElem_[2]; k++ )
      for( Int j=0; j< numElem_[1]; j++ )
        for( Int i=0; i< numElem_[0]; i++ ) {
          Index3 key = Index3(i,j,k);
          if( distEigSolPtr_->Prtn().Owner(key) == (mpirank / dmRow_) ){
            DistDblNumVec& vtotCur = hamDG.Vtot();
            VtotHist.LocalMap()[key] = vtotCur.LocalMap()[key];
            //VtotHist.LocalMap()[key] = mixInnerSave_.LocalMap()[key];
          } // owns this element
    } // for (i)

    Real timePEXSISta, timePEXSIEnd;
    GetTime( timePEXSISta );

    Real numElectronExact = hamDG.NumOccupiedState() * hamDG.NumSpin();
    Real muMinInertia, muMaxInertia;
    Real muPEXSI, numElectronPEXSI;
    Int numTotalInertiaIter = 0, numTotalPEXSIIter = 0;

    std::vector<Int> mpirankSparseVec( numProcPEXSICommCol_ );

    // FIXME 
    // Currently, only the first processor column participate in the
    // communication between PEXSI and DGDFT For the first processor
    // column involved in PEXSI, the first numProcPEXSICommCol_
    // processors are involved in the data communication between PEXSI
    // and DGDFT

    for( Int i = 0; i < numProcPEXSICommCol_; i++ ){
      mpirankSparseVec[i] = i;
    }

#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "mpirankSparseVec = " << mpirankSparseVec << std::endl;
#endif

    Int info;

    // Temporary matrices 
    DistSparseMatrix<Real>  HSparseMat;
    DistSparseMatrix<Real>  DMSparseMat;
    DistSparseMatrix<Real>  EDMSparseMat;
    DistSparseMatrix<Real>  FDMSparseMat;

    if( mpirankRow == 0 ){

      // Convert the DG matrix into the distributed CSC format
      GetTime(timeSta);
      DistElemMatToDistSparseMat3( 
          hamDG.HMat(),
          hamDG.NumBasisTotal(),
          HSparseMat,
          hamDG.ElemBasisIdx(),
          domain_.colComm,
          mpirankSparseVec );
      GetTime(timeEnd);

#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for converting the DG matrix to DistSparseMatrix format is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

#if ( _DEBUGlevel_ >= 0 )
      if( mpirankCol < numProcPEXSICommCol_ ){
        statusOFS << "H.size = " << HSparseMat.size << std::endl;
        statusOFS << "H.nnz  = " << HSparseMat.nnz << std::endl;
        statusOFS << "H.nnzLocal  = " << HSparseMat.nnzLocal << std::endl;
        statusOFS << "H.colptrLocal.m() = " << HSparseMat.colptrLocal.m() << std::endl;
        statusOFS << "H.rowindLocal.m() = " << HSparseMat.rowindLocal.m() << std::endl;
        statusOFS << "H.nzvalLocal.m() = " << HSparseMat.nzvalLocal.m() << std::endl;
      }
#endif
    }// if( mpirankRow == 0)

    // So energy must be obtained from DM as in totalEnergyH
    // and free energy is nothing but energy..
    Real totalEnergyH, totalFreeEnergy;
    //if( (mpirankRow < numProcPEXSICommRow_) && (mpirankCol < numProcPEXSICommCol_) )
    {
      // Load the matrices into PEXSI.  
      // Only the processors with mpirankCol == 0 need to carry the
      // nonzero values of HSparseMat

#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "numProcPEXSICommRow_ = " << numProcPEXSICommRow_ << std::endl;
      statusOFS << "numProcPEXSICommCol_ = " << numProcPEXSICommCol_ << std::endl;
      statusOFS << "mpirankRow = " << mpirankRow << std::endl;
      statusOFS << "mpirankCol = " << mpirankCol << std::endl;
#endif
      GetTime( timeSta );

#ifndef ELSI                
      PPEXSILoadRealHSMatrix(
          plan_,
          pexsiOptions_,
          HSparseMat.size,
          HSparseMat.nnz,
          HSparseMat.nnzLocal,
          HSparseMat.colptrLocal.m() - 1,
          HSparseMat.colptrLocal.Data(),
          HSparseMat.rowindLocal.Data(),
          HSparseMat.nzvalLocal.Data(),
          1,  // isSIdentity
          NULL,
          &info );
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for loading the matrix into PEXSI is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      if( info != 0 ){
        std::ostringstream msg;
        msg 
          << "PEXSI loading H matrix returns info " << info << std::endl;
        ErrorHandling( msg.str().c_str() );
      }
#endif         

      // PEXSI solver
      {
        if( outerIter >= inertiaCountSteps_ ){
          pexsiOptions_.isInertiaCount = 0;
        }
        // Note: Heuristics strategy for dynamically adjusting the
        // tolerance
        pexsiOptions_.muInertiaTolerance = 
          std::min( std::max( muInertiaToleranceTarget_, 0.1 * scfOuterNorm_ ), 0.01 );
        pexsiOptions_.numElectronPEXSITolerance = 
          std::min( std::max( numElectronPEXSIToleranceTarget_, 1.0 * scfOuterNorm_ ), 0.5 );

        // Only perform symbolic factorization for the first outer SCF. 
        // Reuse the previous Fermi energy as the initial guess for mu.
        if( outerIter == 1 ){
          pexsiOptions_.isSymbolicFactorize = 1;
          pexsiOptions_.mu0 = 0.5 * (pexsiOptions_.muMin0 + pexsiOptions_.muMax0);
        }
        else{
          pexsiOptions_.isSymbolicFactorize = 0;
          pexsiOptions_.mu0 = fermi_;
        }

        statusOFS << std::endl 
          << "muInertiaTolerance        = " << pexsiOptions_.muInertiaTolerance << std::endl
          << "numElectronPEXSITolerance = " << pexsiOptions_.numElectronPEXSITolerance << std::endl
          << "Symbolic factorization    =  " << pexsiOptions_.isSymbolicFactorize << std::endl;
      }
#ifdef ELSI
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << std::endl << "ELSI PEXSI set sparsity start" << std::endl<< std::flush;
#endif
#endif

#ifdef ELSI
      c_elsi_set_sparsity( HSparseMat.nnz,
                           HSparseMat.nnzLocal,
                           HSparseMat.colptrLocal.m() - 1,
                           HSparseMat.rowindLocal.Data(),
                           HSparseMat.colptrLocal.Data() );

      c_elsi_customize_pexsi( pexsiOptions_.temperature,
                              pexsiOptions_.gap,
                              pexsiOptions_.deltaE,
                              pexsiOptions_.numPole,
                              numProcPEXSICommCol_,  // # n_procs_per_pole
                              pexsiOptions_.maxPEXSIIter,
                              pexsiOptions_.muMin0,
                              pexsiOptions_.muMax0,
                              pexsiOptions_.mu0,
                              pexsiOptions_.muInertiaTolerance,
                              pexsiOptions_.muInertiaExpansion,
                              pexsiOptions_.muPEXSISafeGuard,
                              pexsiOptions_.numElectronPEXSITolerance,
                              pexsiOptions_.matrixType,
                              pexsiOptions_.isSymbolicFactorize,
                              pexsiOptions_.ordering,
                              pexsiOptions_.npSymbFact,
                              pexsiOptions_.verbosity);

#if ( _DEBUGlevel_ >= 0 )
      statusOFS << std::endl << "ELSI PEXSI Customize Done " << std::endl;
#endif

      if( mpirankRow == 0 )
         CopyPattern( HSparseMat, DMSparseMat );
      statusOFS << std::endl << "ELSI PEXSI Copy pattern done" << std::endl;
      c_elsi_dm_real_sparse(HSparseMat.nzvalLocal.Data(), NULL, DMSparseMat.nzvalLocal.Data());

      GetTime( timeEnd );
      statusOFS << std::endl << "ELSI PEXSI real sparse done" << std::endl;

      if( mpirankRow == 0 ){
        CopyPattern( HSparseMat, EDMSparseMat );
        CopyPattern( HSparseMat, FDMSparseMat );
        c_elsi_collect_pexsi(&fermi_,EDMSparseMat.nzvalLocal.Data(),FDMSparseMat.nzvalLocal.Data());
        statusOFS << std::endl << "ELSI PEXSI collecte done " << std::endl;
      }
      statusOFS << std::endl << "Time for ELSI PEXSI = " << 
        timeEnd - timeSta << " [s]" << std::endl << std::endl<<std::flush;
#endif

#ifndef ELSI
      GetTime( timeSta );

      // New version of PEXSI driver, uses inertia count + pole update
      // strategy. No Newton's iteration. But this is not very stable.
      pexsiOptions_.method = esdfParam.pexsiMethod;
      pexsiOptions_.nPoints = esdfParam.pexsiNpoint;

      PPEXSIDFTDriver2(
          plan_,
          &pexsiOptions_,
          numElectronExact,
          &muPEXSI,
          &numElectronPEXSI,         
          &numTotalInertiaIter,
          &info );

      // New version of PEXSI driver, use inertia count + pole update.
      // two method of pole expansion. default is 2

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for the main PEXSI Driver is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      if( info != 0 ){
        std::ostringstream msg;
        msg 
          << "PEXSI main driver returns info " << info << std::endl;
        ErrorHandling( msg.str().c_str() );
      }

      // Update the fermi level 
      fermi_ = muPEXSI;
      difNumElectron = std::abs(numElectronPEXSI - numElectronExact);

      // Heuristics for the next step
      //pexsiOptions_.muMin0 = muMinInertia - 5.0 * pexsiOptions_.temperature;
      //pexsiOptions_.muMax0 = muMaxInertia + 5.0 * pexsiOptions_.temperature;

      // Retrieve the PEXSI data

      // FIXME: Hack: in PEXSIDriver3, only DM is available.

      if( ( mpirankRow == 0 ) && (mpirankCol < numProcPEXSICommCol_) ){
      // if( mpirankRow == 0 ){
        Real totalEnergyS;

        GetTime( timeSta );

        CopyPattern( HSparseMat, DMSparseMat );
        CopyPattern( HSparseMat, EDMSparseMat );
        CopyPattern( HSparseMat, FDMSparseMat );

        statusOFS << "Before retrieve" << std::endl;
        PPEXSIRetrieveRealDFTMatrix(
            // pexsiPlan_,
            plan_,
            DMSparseMat.nzvalLocal.Data(),
            EDMSparseMat.nzvalLocal.Data(),
            FDMSparseMat.nzvalLocal.Data(),
            &totalEnergyH,
            &totalEnergyS,
            &totalFreeEnergy,
            &info );
        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for retrieving PEXSI data is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
        // FIXME: Hack: there is no free energy really. totalEnergyS is to be added later
        statusOFS << "NOTE: Free energy = Energy in PPEXSIDFTDriver3!" << std::endl;

        statusOFS << std::endl
          << "Results obtained from PEXSI:" << std::endl
          << "Total energy (H*DM)         = " << totalEnergyH << std::endl
          << "Total energy (S*EDM)        = " << totalEnergyS << std::endl
          << "Total free energy           = " << totalFreeEnergy << std::endl 
          << "InertiaIter                 = " << numTotalInertiaIter << std::endl
          << "mu                          = " << muPEXSI << std::endl
          << "numElectron                 = " << numElectronPEXSI << std::endl 
          << std::endl;

        if( info != 0 ){
          std::ostringstream msg;
          msg 
            << "PEXSI data retrieval returns info " << info << std::endl;
          ErrorHandling( msg.str().c_str() );
        }
      }  //  if( ( mpirankRow == 0 ) && (mpirankCol < numProcPEXSICommCol_) )
#endif
    } // if( (mpirankRow < numProcPEXSICommRow_) && (mpirankCol < numProcPEXSICommCol_) )

    // Broadcast the total energy Tr[H*DM] and free energy (which is energy)
    MPI_Bcast( &totalEnergyH, 1, MPI_DOUBLE, 0, domain_.comm );
    MPI_Bcast( &totalFreeEnergy, 1, MPI_DOUBLE, 0, domain_.comm );
    // Broadcast the Fermi level
    MPI_Bcast( &fermi_, 1, MPI_DOUBLE, 0, domain_.comm );
    MPI_Bcast( &difNumElectron, 1, MPI_DOUBLE, 0, domain_.comm );

    if( mpirankRow == 0 )
    {
      GetTime(timeSta);
      // Convert the density matrix from DistSparseMatrix format to the
      // DistElemMat format
      DistSparseMatToDistElemMat3(
          DMSparseMat,
          hamDG.NumBasisTotal(),
          hamDG.HMat().Prtn(),
          distDMMat_,
          hamDG.ElemBasisIdx(),
          hamDG.ElemBasisInvIdx(),
          domain_.colComm,
          mpirankSparseVec );

      // Convert the energy density matrix from DistSparseMatrix
      // format to the DistElemMat format

      DistSparseMatToDistElemMat3(
          EDMSparseMat,
          hamDG.NumBasisTotal(),
          hamDG.HMat().Prtn(),
          distEDMMat_,
          hamDG.ElemBasisIdx(),
          hamDG.ElemBasisInvIdx(),
          domain_.colComm,
          mpirankSparseVec );

      // Convert the free energy density matrix from DistSparseMatrix
      // format to the DistElemMat format
      DistSparseMatToDistElemMat3(
          FDMSparseMat,
          hamDG.NumBasisTotal(),
          hamDG.HMat().Prtn(),
          distFDMMat_,
          hamDG.ElemBasisIdx(),
          hamDG.ElemBasisInvIdx(),
          domain_.colComm,
          mpirankSparseVec );

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for converting the DistSparseMatrices to DistElemMat " << 
        "for post-processing is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    }

    // Broadcast the distElemMat matrices
    // FIXME this is not a memory efficient implementation
    GetTime(timeSta);
    {
      Int sstrSize;
      std::vector<char> sstr;
      if( mpirankRow == 0 ){
        std::stringstream distElemMatStream;
        Int cnt = 0;
        for( typename std::map<ElemMatKey, DblNumMat >::iterator mi  = distDMMat_.LocalMap().begin();
            mi != distDMMat_.LocalMap().end(); ++mi ){ 
          cnt++;
        } // for (mi)
        serialize( cnt, distElemMatStream, NO_MASK );
        for( typename std::map<ElemMatKey, DblNumMat >::iterator mi  = distDMMat_.LocalMap().begin();
            mi != distDMMat_.LocalMap().end(); ++mi ){
          ElemMatKey key = (*mi).first;
          serialize( key, distElemMatStream, NO_MASK );
          serialize( distDMMat_.LocalMap()[key], distElemMatStream, NO_MASK );

          serialize( distEDMMat_.LocalMap()[key], distElemMatStream, NO_MASK ); 
          serialize( distFDMMat_.LocalMap()[key], distElemMatStream, NO_MASK ); 

        } // for (mi)
        sstr.resize( Size( distElemMatStream ) );
        distElemMatStream.read( &sstr[0], sstr.size() );
        sstrSize = sstr.size();
      }

      MPI_Bcast( &sstrSize, 1, MPI_INT, 0, domain_.rowComm );
      sstr.resize( sstrSize );
      MPI_Bcast( &sstr[0], sstrSize, MPI_BYTE, 0, domain_.rowComm );

      if( mpirankRow != 0 ){
        std::stringstream distElemMatStream;
        distElemMatStream.write( &sstr[0], sstrSize );
        Int cnt;
        deserialize( cnt, distElemMatStream, NO_MASK );
        for( Int i = 0; i < cnt; i++ ){
          ElemMatKey key;
          DblNumMat mat;
          deserialize( key, distElemMatStream, NO_MASK );
          deserialize( mat, distElemMatStream, NO_MASK );
          distDMMat_.LocalMap()[key] = mat;

          deserialize( mat, distElemMatStream, NO_MASK );
          distEDMMat_.LocalMap()[key] = mat;
          deserialize( mat, distElemMatStream, NO_MASK );
          distFDMMat_.LocalMap()[key] = mat;
        } // for (mi)
      }
    }

    GetTime(timeSta);
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for broadcasting the density matrix for post-processing is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    Evdw_ = 0.0;

    // Compute the Harris energy functional.  
    // NOTE: In computing the Harris energy, the density and the
    // potential must be the INPUT density and potential without ANY
    // update.
    GetTime( timeSta );
    CalculateHarrisEnergyDM( totalFreeEnergy, distFDMMat_ );
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for calculating the Harris energy is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    // Evaluate the electron density
    GetTime( timeSta );
    hamDG.CalculateDensityDM2( 
    hamDG.Density(), hamDG.DensityLGL(), distDMMat_ );
    MPI_Barrier( domain_.comm );
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing density in the global domain is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumVec&  density      = hamDG.Density().LocalMap()[key];
          } // own this element
    } // for (i)

    // Update the output potential, and the KS and second order accurate
    // energy
    GetTime(timeSta);
    {
      // Update the Hartree energy and the exchange correlation energy and
      // potential for computing the KS energy and the second order
      // energy.
      // NOTE Vtot should not be updated until finishing the computation
      // of the energies.

      if( isCalculateGradRho_  ){
        GetTime( timeSta );
        hamDG.CalculateGradDensity(  *distfftPtr_ );
        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for calculating gradient of density is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
      }

      hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), *distfftPtr_ );

      hamDG.CalculateHartree( hamDG.Vhart(), *distfftPtr_ );

      if( hamDG.IsHybrid() && hamDG.IsEXXActive()){
        GetTime( timeSta );
        if(esdfParam.isDGHFISDF){
          hamDG.CalculateDGHFXMatrix_ISDF( Ehfx_, distDMMat_ );
        }
        else{
          hamDG.CalculateDGHFXMatrix( Ehfx_, distDMMat_ );
        }
        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "InnerSCF: Time for computing HFX in the global domain is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
      }

      // Compute the second order accurate energy functional.
      // NOTE: In computing the second order energy, the density and the
      // potential must be the OUTPUT density and potential without ANY
      // MIXING.
      //        CalculateSecondOrderEnergy();

      // Compute the KS energy 
      CalculateKSEnergyDM( totalEnergyH, distEDMMat_, distFDMMat_ );

      // Update the total potential AFTER updating the energy

      // No external potential

      // Compute the new total potential

      hamDG.CalculateVtot( hamDG.Vtot() );

    }
    GetTime(timeEnd);
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing the potential is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      // Compute the force at every step
      //      if( esdfParam.isCalculateForceEachSCF ){
      //        // Compute force
      //        GetTime( timeSta );
      //        hamDG.CalculateForceDM( *distfftPtr_, distDMMat_ );
      //        GetTime( timeEnd );
      //        statusOFS << "Time for computing the force is " <<
      //          timeEnd - timeSta << " [s]" << std::endl << std::endl;
      //
      //        // Print out the force
      //        // Only master processor output information containing all atoms
      //        if( mpirank == 0 ){
      //          PrintBlock( statusOFS, "Atomic Force" );
      //          {
      //            Point3 forceCM(0.0, 0.0, 0.0);
      //            std::vector<Atom>& atomList = hamDG.AtomList();
      //            Int numAtom = atomList.size();
      //            for( Int a = 0; a < numAtom; a++ ){
      //              Print( statusOFS, "atom", a, "force", atomList[a].force );
      //              forceCM += atomList[a].force;
      //            }
      //            statusOFS << std::endl;
      //            Print( statusOFS, "force for centroid: ", forceCM );
      //            statusOFS << std::endl;
      //          }
      //        }
      //      }

      // TODO Evaluate the a posteriori error estimator

    GetTime( timePEXSIEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PEXSI evaluation is " <<
      timePEXSIEnd - timePEXSISta << " [s]" << std::endl << std::endl;
#endif
  } //if( solutionMethod_ == "pexsi" )

#endif

  // **************************************************************************************
  // Compute the error of the mixing variable
  GetTime(timeSta);
  {
    Real normMixDifLocal = 0.0, normMixOldLocal = 0.0;
    Real normMixDif, normMixOld;

    Real MaxDifLocal = 0.0, MaxDif = 0.0;

    if( InnermixVariable_ == "density" || InnermixVariable_ == "potential" ){
      for( Int k = 0; k < numElem_[2]; k++ ){
        for( Int j = 0; j < numElem_[1]; j++ ){
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              if( InnermixVariable_ == "density" ){
                DblNumVec& oldVec = mixInnerSave_.LocalMap()[key];
                DblNumVec& newVec = hamDG.Density().LocalMap()[key];

                for( Int p = 0; p < oldVec.m(); p++ ){
                  normMixDifLocal += pow( oldVec(p) - newVec(p), 2.0 );
                  normMixOldLocal += pow( oldVec(p), 2.0 );
                }
              }
              else if( InnermixVariable_ == "potential" ){
                DblNumVec& oldVec = mixInnerSave_.LocalMap()[key];
                DblNumVec& newVec = hamDG.Vtot().LocalMap()[key];

                for( Int p = 0; p < oldVec.m(); p++ ){
                  normMixDifLocal += pow( oldVec(p) - newVec(p), 2.0 );
                  normMixOldLocal += pow( oldVec(p), 2.0 );
                }
              }
            } // own this element
          } // for (i)
        } // for(j)
      } // for (k)

      mpi::Allreduce( &normMixDifLocal, &normMixDif, 1, MPI_SUM,
        domain_.colComm );
      mpi::Allreduce( &normMixOldLocal, &normMixOld, 1, MPI_SUM,
        domain_.colComm );
  
      normMixDif = std::sqrt( normMixDif );
      normMixOld = std::sqrt( normMixOld );
  
      scfInnerNorm_    = normMixDif / normMixOld;
#if ( _DEBUGlevel_ >= 0 )
      Print(statusOFS, "norm(MixDif)          = ", normMixDif );
      Print(statusOFS, "norm(MixOld)          = ", normMixOld );
      Print(statusOFS, "norm(out-in)/norm(in) = ", scfInnerNorm_ );
#endif
      if( scfInnerNorm_ < scfInnerTolerance_ ){
        /* converged */
        Print( statusOFS, "Inner SCF is converged!\n" );
        isInnerSCFConverged = true;
      }
    }
    else{  
      if( InnermixVariable_ == "hamiltonian" ){
        for(typename std::map<ElemMatKey, DblNumMat >::iterator
          Ham_iterator = hamDG.HMat().LocalMap().begin();
          Ham_iterator != hamDG.HMat().LocalMap().end();
            ++ Ham_iterator ) {
          ElemMatKey matkey = (*Ham_iterator).first;
          DblNumMat& oldMat = distmixInnerSave_.LocalMap()[matkey];
          DblNumMat& newMat = hamDG.HMat().LocalMap()[matkey];

          for( Int q = 0; q < oldMat.n(); q++ ){
            for( Int p = 0; p < oldMat.m(); p++ ){
               Real diffMat = std::abs( oldMat(p, q) - newMat(p, q) );
               MaxDifLocal = std::max( MaxDifLocal, diffMat );
            }
          }
        }
      }
      else if( InnermixVariable_ == "densitymatrix" ){
          for(typename std::map<ElemMatKey, DblNumMat >::iterator
            Ham_iterator = distDMMat_.LocalMap().begin();
            Ham_iterator !=  distDMMat_.LocalMap().end();
            ++ Ham_iterator ) {
            ElemMatKey matkey = (*Ham_iterator).first;
            DblNumMat& oldMat = distmixInnerSave_.LocalMap()[matkey];
            DblNumMat& newMat =  distDMMat_.LocalMap()[matkey];

            for( Int q = 0; q < oldMat.n(); q++ ){
              for( Int p = 0; p < oldMat.m(); p++ ){
                Real diffMat = std::abs( oldMat(p, q) - newMat(p, q) );
                MaxDifLocal = std::max( MaxDifLocal, diffMat );
              }
            }
          }
      }

      mpi::Allreduce( &MaxDifLocal, &MaxDif, 1, MPI_MAX, domain_.colComm );
      scfInnerMaxDif_    = MaxDif;
#if ( _DEBUGlevel_ >= 0 )
      Print(statusOFS, "Inner MaxDiff(out-in) = ", scfInnerMaxDif_ );
#endif

      if( scfInnerMaxDif_ < scfInnerTolerance_ ){
         /* converged */
         Print( statusOFS, "Inner SCF is converged!\n" );
         isInnerSCFConverged = true;
      }
    }                          
  }

  MPI_Barrier( domain_.colComm );
  MPI_Barrier( domain_.rowComm ); 
  MPI_Barrier( domain_.comm ); 
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing the SCF residual is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  // Mixing for the inner SCF iteration.
  GetTime( timeSta );

  // The number of iterations used for Anderson mixing
  Int numAndersonIter;

  if( scfInnerMaxIter_ == 1 ){
    // Maximum inner iteration = 1 means there is no distinction of
    // inner/outer SCF.  Anderson mixing uses the global history
    numAndersonIter = scfTotalInnerIter_;
  }
  else{
    // If more than one inner iterations is used, then Anderson only
    // uses local history.  For explanation see 
    numAndersonIter = innerIter;
  }

  if( InnermixVariable_ == "density" ){
    if( mixType_ == "anderson" ||
        mixType_ == "kerker+anderson"    ){
      statusOFS << " anderson density mixing " << std::endl;
      AndersonMix2(
          numAndersonIter, 
          mixStepLength_,
          hamDG.Density(),
          mixInnerSave_,
          hamDG.Density(),
          dfInnerMat_,
          dvInnerMat_);
    }
    else if( mixType_ == "broyden" ){
      statusOFS << " broyden density mixing " << std::endl;
      BroydenMix(
          numAndersonIter,
          mixStepLength_,
          hamDG.Density(),
          mixInnerSave_,
          hamDG.Density(),
          dfInnerMat_,
          dvInnerMat_,
          cdfInnerMat_);
    }
    else{
      ErrorHandling("Invalid density mixing type.");
    }
  }
  else if( InnermixVariable_ == "potential" ){
    if( mixType_ == "anderson" ||
        mixType_ == "kerker+anderson"    ){
      statusOFS << " anderson potential mixing " << std::endl;
      AndersonMix2(
          numAndersonIter, 
          mixStepLength_,
          hamDG.Vtot(),
          mixInnerSave_,
          hamDG.Vtot(),
          dfInnerMat_,
          dvInnerMat_);
    }
    else if( mixType_ == "broyden" ){
      statusOFS << " broyden potential mixing " << std::endl;
      BroydenMix(
          numAndersonIter,
          mixStepLength_,
          hamDG.Vtot(),
          mixInnerSave_,
          hamDG.Vtot(),
          dfInnerMat_,
          dvInnerMat_,
          cdfInnerMat_);
    }
    else{
      ErrorHandling("Invalid potential mixing type.");
    }
  }
  else if( InnermixVariable_ == "densitymatrix" ){
    if( mixType_ == "anderson" ){
      statusOFS << " anderson densitymatrix mixing " << std::endl;
      AndersonMix2(
            numAndersonIter,
            mixStepLength_,
            distDMMat_,
            distmixInnerSave_,
            distDMMat_,
            distdfInnerMat_,
            distdvInnerMat_);
    } 
    else if( mixType_ == "broyden" ){
      statusOFS << " broyden densitymatrix mixing " << std::endl;
      BroydenMix(
            numAndersonIter,
            mixStepLength_,
            distDMMat_,
            distmixInnerSave_,
            distDMMat_,
            distdfInnerMat_,
            distdvInnerMat_,
            distcdfInnerMat_);
    }
    else{
      ErrorHandling("Invalid densitymatrix mixing type.");
    } 
  }
  else if( InnermixVariable_ == "hamiltonian" ){
    if( mixType_ == "anderson" ){
      statusOFS << " anderson hamiltonian mixing " << std::endl;
      AndersonMix2(
          numAndersonIter,
          mixStepLength_,
          hamDG.HMat(),
          distmixInnerSave_,
          hamDG.HMat(),
          distdfInnerMat_,
          distdvInnerMat_);
    } 
    else if( mixType_ == "broyden" ){
      statusOFS << " broyden hamiltonian mixing " << std::endl;
      BroydenMix(
            numAndersonIter,
            mixStepLength_,
            hamDG.HMat(),
            distmixInnerSave_,
            hamDG.HMat(),
            distdfInnerMat_,
            distdvInnerMat_,
            distcdfInnerMat_);
    }
    else{
      ErrorHandling("Invalid hamiltonian mixing type.");
    } 
  }

  MPI_Barrier( domain_.comm );
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "InnerSCF: Time for mixing is " <<
    timeEnd - timeSta << " [s]" << std::endl;
#endif

  // Post processing for the density mixing. Make sure that the
  // density is positive, and compute the potential again. 
  // This is only used for density mixing.
  if( InnermixVariable_ == "densitymatrix" ){
    statusOFS << "InnerSCF: Recalculate density from mixed density matrix " << std::endl;
    hamDG.CalculateDensityDM2( hamDG.Density(), hamDG.DensityLGL(), distDMMat_ );
  }

  if( (InnermixVariable_ == "density") || (InnermixVariable_ == "densitymatrix") )
  {
    Real sumRhoLocal = 0.0;
    Real sumRho;
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumVec&  density      = hamDG.Density().LocalMap()[key];

            for (Int p=0; p < density.Size(); p++) {
              density(p) = std::max( density(p), 0.0 );
              sumRhoLocal += density(p);
            }
          } // own this element
        } // for (i)

    mpi::Allreduce( &sumRhoLocal, &sumRho, 1, MPI_SUM, domain_.colComm );
    sumRho *= domain_.Volume() / domain_.NumGridTotalFine();

    Real rhofac = hamDG.NumSpin() * hamDG.NumOccupiedState() / sumRho;
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << std::endl;
    Print( statusOFS, "Numer of Occupied State ",  hamDG.NumOccupiedState() );
    Print( statusOFS, "Rho factor after mixing (raw data) = ", rhofac );
    Print( statusOFS, "Sum Rho after mixing (raw data)    = ", sumRho );
    statusOFS << std::endl;
#endif
    // Normalize the electron density in the global domain
#if 0
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              DblNumVec& localRho = hamDG.Density().LocalMap()[key];
              blas::Scal( localRho.Size(), rhofac, localRho.Data(), 1 );
            } // own this element
    } // for (i)
#endif
    // Update the potential after mixing for the next iteration.  
    // This is only used for potential mixing

    // Compute the exchange-correlation potential and energy from the
    // new density

    if( isCalculateGradRho_  ){
      GetTime( timeSta );
      hamDG.CalculateGradDensity(  *distfftPtr_ );
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for calculating gradient of density is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    }

    GetTime( timeSta );
    hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), *distfftPtr_ );
    statusOFS << "Exc after DIAG " << Exc_ << std::endl;
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing Exc in the global domain is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      GetTime( timeSta );

      hamDG.CalculateHartree( hamDG.Vhart(), *distfftPtr_ );

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "InnerSCF: Time for computing Vhart in the global domain is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
      // Compute the second order accurate energy functional.
      // NOTE: In computing the second order energy, the density and the
      // potential must be the OUTPUT density and potential without ANY
      // MIXING.
      CalculateSecondOrderEnergy();
      // Compute the KS energy
      GetTime( timeSta );

      CalculateKSEnergy();

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "InnerSCF: Time for computing KSEnergy in the global domain is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    hamDG.CalculateVtot( hamDG.Vtot() );

  } // if( InnermixVariable_ == "density" || ( InnermixVariable_ == "densitymatrix" && domixingThisstep ) )

#ifdef _USE_PEXSI_
  if( solutionMethod_ == "pexsi" )
  {
    Real deltaVmin = 0.0;
    Real deltaVmax = 0.0;

    for( Int k=0; k< numElem_[2]; k++ )
      for( Int j=0; j< numElem_[1]; j++ )
        for( Int i=0; i< numElem_[0]; i++ ) {
          Index3 key = Index3(i,j,k);
          if( distEigSolPtr_->Prtn().Owner(key) == (mpirank / dmRow_) ){
            DblNumVec vtotCur;
            vtotCur = hamDG.Vtot().LocalMap()[key];
            DblNumVec& oldVtot = VtotHist.LocalMap()[key];
            blas::Axpy( vtotCur.m(), -1.0, oldVtot.Data(),
                            1, vtotCur.Data(), 1);
            deltaVmin = std::min( deltaVmin, findMin(vtotCur) );
            deltaVmax = std::max( deltaVmax, findMax(vtotCur) );
          }
      }

      {
        Int color = mpirank % dmRow_;
        MPI_Comm elemComm;
        std::vector<Real> vlist(mpisize/dmRow_);

        MPI_Comm_split( domain_.comm, color, mpirank, &elemComm );
        MPI_Allgather( &deltaVmin, 1, MPI_DOUBLE, &vlist[0], 1, MPI_DOUBLE, elemComm);
        deltaVmin = 0.0;
        for(Int i =0; i < mpisize/dmRow_; i++)
          if(deltaVmin > vlist[i])
            deltaVmin = vlist[i];

        MPI_Allgather( &deltaVmax, 1, MPI_DOUBLE, &vlist[0], 1, MPI_DOUBLE, elemComm);
        deltaVmax = 0.0;
        for(Int i =0; i < mpisize/dmRow_; i++)
          if(deltaVmax < vlist[i])
            deltaVmax = vlist[i];

        pexsiOptions_.muMin0 += deltaVmin;
        pexsiOptions_.muMax0 += deltaVmax;
        MPI_Comm_free( &elemComm);
      }
    }
#endif
    // Print out the state variables of the current iteration

    // Only master processor output information containing all atoms
    if( mpirank == 0 ){
      PrintState( );
    }

    GetTime( timeIterEnd );

    statusOFS << "Time for this inner SCF iteration = " << timeIterEnd - timeIterStart
      << " [s]" << std::endl;

  } // for (innerIter)

  return ;
}         // -----  end of method SCFDG::InnerIterate  ----- 

// This routine calculates the full density matrix
void SCFDG::scfdg_compute_fullDM()
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  HamiltonianDG&  hamDG = *hamDGPtr_;
  std::vector<Index3>  getKeys_list;

  DistDblNumMat& my_dist_mat = hamDG.EigvecCoef();

  // Check that vectors provided only contain one entry in the local map
  // This is a safeguard to ensure that we are really dealing with distributed matrices
  if((my_dist_mat.LocalMap().size() != 1))
  {
    statusOFS << std::endl << " Eigenvector not formatted correctly !!"
      << std::endl << " Aborting ... " << std::endl;
    exit(1);
  }

  // Copy eigenvectors to temp bufer
  DblNumMat &eigvecs_local = (hamDG.EigvecCoef().LocalMap().begin())->second;

  DblNumMat scal_local_eig_vec;
  scal_local_eig_vec.Resize(eigvecs_local.m(), eigvecs_local.n());
  blas::Copy((eigvecs_local.m() * eigvecs_local.n()), eigvecs_local.Data(), 1, scal_local_eig_vec.Data(), 1);

  // Scale temp buffer by occupation * numspin
  for(int iter_scale = 0; iter_scale < eigvecs_local.n(); iter_scale ++)
  {
    blas::Scal(  scal_local_eig_vec.m(),  
        (hamDG.NumSpin()* hamDG.OccupationRate()[iter_scale]), 
        scal_local_eig_vec.Data() + iter_scale * scal_local_eig_vec.m(), 1 );
  }

  // Obtain key based on my_dist_mat : This assumes that my_dist_mat is formatted correctly
  // based on processor number, etc.
  Index3 key = (my_dist_mat.LocalMap().begin())->first;

  // Obtain keys of neighbors using the Hamiltonian matrix
  for(typename std::map<ElemMatKey, DblNumMat >::iterator 
      get_neighbors_from_Ham_iterator = hamDG.HMat().LocalMap().begin();
      get_neighbors_from_Ham_iterator != hamDG.HMat().LocalMap().end();
      get_neighbors_from_Ham_iterator ++)
  {
    Index3 neighbor_key = (get_neighbors_from_Ham_iterator->first).second;

    if(neighbor_key == key)
      continue;
    else
      getKeys_list.push_back(neighbor_key);
  }

  // Do the communication necessary to get the information from
  // procs holding the neighbors
  my_dist_mat.GetBegin( getKeys_list, NO_MASK ); 
  my_dist_mat.GetEnd( NO_MASK );

  // First compute the diagonal block
  {
    DblNumMat &mat_local = my_dist_mat.LocalMap()[key];
    ElemMatKey diag_block_key = std::make_pair(key, key);

    // Compute the X*X^T portion
    distDMMat_.LocalMap()[diag_block_key].Resize( mat_local.m(),  mat_local.m());

    blas::Gemm( 'N', 'T', mat_local.m(), mat_local.m(), mat_local.n(),
        1.0, 
        scal_local_eig_vec.Data(), scal_local_eig_vec.m(), 
        mat_local.Data(), mat_local.m(),
        0.0, 
        distDMMat_.LocalMap()[diag_block_key].Data(),  mat_local.m());
  }

  // Now handle the off-diagonal blocks
  {
    DblNumMat &mat_local = my_dist_mat.LocalMap()[key];
    for(Int off_diag_iter = 0; off_diag_iter < getKeys_list.size(); off_diag_iter ++)
    {
      DblNumMat &mat_neighbor = my_dist_mat.LocalMap()[getKeys_list[off_diag_iter]];
      ElemMatKey off_diag_key = std::make_pair(key, getKeys_list[off_diag_iter]);

      // First compute the Xi * Xj^T portion
      distDMMat_.LocalMap()[off_diag_key].Resize( scal_local_eig_vec.m(),  mat_neighbor.m());

      blas::Gemm( 'N', 'T', scal_local_eig_vec.m(), mat_neighbor.m(), scal_local_eig_vec.n(),
          1.0, 
          scal_local_eig_vec.Data(), scal_local_eig_vec.m(), 
          mat_neighbor.Data(), mat_neighbor.m(),
          0.0, 
          distDMMat_.LocalMap()[off_diag_key].Data(),  mat_local.m());
    }
  }

  // Need to clean up extra entries in my_dist_mat
  typename std::map<Index3, DblNumMat >::iterator it;
  for(Int delete_iter = 0; delete_iter <  getKeys_list.size(); delete_iter ++)
  {
    it = my_dist_mat.LocalMap().find(getKeys_list[delete_iter]);
    (my_dist_mat.LocalMap()).erase(it);
  }

  return;
} // calculate density matrix

void
SCFDG::UpdateElemLocalPotential    (  )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  HamiltonianDG&  hamDG = *hamDGPtr_;
  // vtot gather the neighborhood
  DistDblNumVec&  vtot = hamDG.Vtot();

  std::set<Index3> neighborSet;
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){
          std::vector<Index3>   idx(3);

          for( Int d = 0; d < DIM; d++ ){
            // Previous
            if( key[d] == 0 ) 
              idx[0][d] = numElem_[d]-1; 
            else 
              idx[0][d] = key[d]-1;

            // Current
            idx[1][d] = key[d];

            // Next
            if( key[d] == numElem_[d]-1) 
              idx[2][d] = 0;
            else
              idx[2][d] = key[d] + 1;
          } // for (d)

          // Tensor product 
          for( Int c = 0; c < 3; c++ )
            for( Int b = 0; b < 3; b++ )
              for( Int a = 0; a < 3; a++ ){
                // Not the element key itself
                if( idx[a][0] != i || idx[b][1] != j || idx[c][2] != k ){
                  neighborSet.insert( Index3( idx[a][0], idx[b][1], idx[c][2] ) );
                }
              } // for (a)
        } // own this element
      } // for (i)
  std::vector<Index3>  neighborIdx;
  neighborIdx.insert( neighborIdx.begin(), neighborSet.begin(), neighborSet.end() );

  // communicate
  vtot.Prtn()   = elemPrtn_;
  vtot.SetComm(domain_.colComm);
  vtot.GetBegin( neighborIdx, NO_MASK );
  vtot.GetEnd( NO_MASK );

  // Update of the local potential in each extended element locally.
  // The nonlocal potential does not need to be updated
  //
  // Also update the local potential on the LGL grid in hamDG.
  //
  // NOTE:
  //
  // 1. It is hard coded that the extended element is 1 or 3
  // times the size of the element
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
          // Skip the calculation if there is no adaptive local
          // basis function.  
          if( eigSol.Psi().NumState() == 0 )
            continue;

          Hamiltonian&  hamExtElem  = eigSol.Ham();
          DblNumVec&    vtotExtElem = hamExtElem.Vtot();
          SetValue( vtotExtElem, 0.0 );

          Index3 numGridElem = hamDG.NumUniformGridElemFine();
          Index3 numGridExtElem = eigSol.FFT().domain.numGridFine;

          // Update the potential in the extended element
          for(std::map<Index3, DblNumVec>::iterator 
              mi = vtot.LocalMap().begin();
              mi != vtot.LocalMap().end(); ++mi ){
            Index3      keyElem = (*mi).first;
            DblNumVec&  vtotElem = (*mi).second;
            // Determine the shiftIdx which maps the position of vtotElem to 
            // vtotExtElem
            Index3 shiftIdx;
            for( Int d = 0; d < DIM; d++ ){
              shiftIdx[d] = keyElem[d] - key[d];
              shiftIdx[d] = shiftIdx[d] - IRound( Real(shiftIdx[d]) / 
                  numElem_[d] ) * numElem_[d];
              // FIXME Adjustment  
              if( numElem_[d] > 1 ) shiftIdx[d] ++;
              shiftIdx[d] *= numGridElem[d];
            }

            Int ptrExtElem, ptrElem;
            for( Int k = 0; k < numGridElem[2]; k++ )
              for( Int j = 0; j < numGridElem[1]; j++ )
                for( Int i = 0; i < numGridElem[0]; i++ ){
                  ptrExtElem = (shiftIdx[0] + i) + 
                    ( shiftIdx[1] + j ) * numGridExtElem[0] +
                    ( shiftIdx[2] + k ) * numGridExtElem[0] * numGridExtElem[1];
                  ptrElem    = i + j * numGridElem[0] + k * numGridElem[0] * numGridElem[1];
                  vtotExtElem( ptrExtElem ) = vtotElem( ptrElem );
                } // for (i)
          } // for (mi)

          // Loop over the neighborhood

        } // own this element
      } // for (i)

  // Clean up vtot not owned by this element
  std::vector<Index3>  eraseKey;
  for( std::map<Index3, DblNumVec>::iterator 
      mi  = vtot.LocalMap().begin();
      mi != vtot.LocalMap().end(); ++mi ){
    Index3 key = (*mi).first;
    if( vtot.Prtn().Owner(key) != (mpirank / dmRow_) ){
      eraseKey.push_back( key );
    }
  }

  for( std::vector<Index3>::iterator vi = eraseKey.begin();
      vi != eraseKey.end(); ++vi ){
    vtot.LocalMap().erase( *vi );
  }

  // Modify the potential in the extended element.  Current options are
  //
  // 1. Add barrier
  // 2. Periodize the potential
  //
  // Numerical results indicate that option 2 seems to be better.
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
          Index3 numGridExtElemFine = eigSol.FFT().domain.numGridFine;

          // Add the external barrier potential. CANNOT be used
          // together with periodization option
          if( esdfParam.isPotentialBarrier ){
            Domain& dmExtElem = eigSol.FFT().domain;
            DblNumVec& vext = eigSol.Ham().Vext();
            SetValue( vext, 0.0 );
            for( Int gk = 0; gk < dmExtElem.numGridFine[2]; gk++)
              for( Int gj = 0; gj < dmExtElem.numGridFine[1]; gj++ )
                for( Int gi = 0; gi < dmExtElem.numGridFine[0]; gi++ ){
                  Int idx = gi + gj * dmExtElem.numGridFine[0] + 
                    gk * dmExtElem.numGridFine[0] * dmExtElem.numGridFine[1];
                  vext[idx] = vBarrier_[0][gi] + vBarrier_[1][gj] + vBarrier_[2][gk];
                } // for (gi)
            // NOTE:
            // Directly modify the vtot.  vext is not used in the
            // matrix-vector multiplication in the eigensolver.
            blas::Axpy( numGridExtElemFine.prod(), 1.0, eigSol.Ham().Vext().Data(), 1,
                eigSol.Ham().Vtot().Data(), 1 );
          }

          // Periodize the external potential. CANNOT be used together
          // with the barrier potential option
          if( esdfParam.isPeriodizePotential ){
            Domain& dmExtElem = eigSol.FFT().domain;
            // Get the potential
            DblNumVec& vext = eigSol.Ham().Vext();
            DblNumVec& vtot = eigSol.Ham().Vtot();

            // Find the max of the potential in the extended element
            Real vtotMax = *std::max_element( &vtot[0], &vtot[0] + vtot.Size() );
            Real vtotAvg = 0.0;
            for(Int i = 0; i < vtot.Size(); i++){
              vtotAvg += vtot[i];
            }
            vtotAvg /= Real(vtot.Size());
            Real vtotMin = *std::min_element( &vtot[0], &vtot[0] + vtot.Size() );

            SetValue( vext, 0.0 );
            for( Int gk = 0; gk < dmExtElem.numGridFine[2]; gk++)
              for( Int gj = 0; gj < dmExtElem.numGridFine[1]; gj++ )
                for( Int gi = 0; gi < dmExtElem.numGridFine[0]; gi++ ){
                  Int idx = gi + gj * dmExtElem.numGridFine[0] + 
                    gk * dmExtElem.numGridFine[0] * dmExtElem.numGridFine[1];
                  // Bring the potential to the vacuum level
                  vext[idx] = ( vtot[idx] - 0.0 ) * 
                    ( vBubble_[0][gi] * vBubble_[1][gj] * vBubble_[2][gk] - 1.0 );
                } // for (gi)
            // NOTE:
            // Directly modify the vtot.  vext is not used in the
            // matrix-vector multiplication in the eigensolver.
            blas::Axpy( numGridExtElemFine.prod(), 1.0, eigSol.Ham().Vext().Data(), 1,
                eigSol.Ham().Vtot().Data(), 1 );
          } // if ( isPeriodizePotential_ ) 
        } // own this element
      } // for (i)

  // Update the potential in element on LGL grid
  //
  // The local potential on the LGL grid is done by using Fourier
  // interpolation from the extended element to the element. Gibbs
  // phenomena MAY be there but at least this is better than
  // Lagrange interpolation on a uniform grid.
  //
  // NOTE: The interpolated potential on the LGL grid is taken to be the
  // MODIFIED potential with vext on the extended element. Therefore it
  // is important that the artificial vext vanishes inside the element.
  // When periodization option is used, it can potentially reduce the
  // effect of Gibbs phenomena.

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
          Index3 numGridExtElemFine = eigSol.FFT().domain.numGridFine;

          DblNumVec&  vtotLGLElem = hamDG.VtotLGL().LocalMap()[key];
          Index3 numLGLGrid       = hamDG.NumLGLGridElem();

          DblNumVec&    vtotExtElem = eigSol.Ham().Vtot();

          InterpPeriodicUniformFineToLGL( 
              numGridExtElemFine,
              numLGLGrid,
              vtotExtElem.Data(),
              vtotLGLElem.Data() );
        } // own this element
      } // for (i)

  return ;
}         // -----  end of method SCFDG::UpdateElemLocalPotential  ----- 

void
SCFDG::CalculateOccupationRate    ( DblNumVec& eigVal, DblNumVec& occupationRate )
{
  // For a given finite temperature, update the occupation number */
  Real tol = 1e-16;
  Int maxiter = 200;

  if(SmearingScheme_ == "FD")
  {
    Int npsi       = hamDGPtr_->NumStateTotal();
    Int nOccStates = hamDGPtr_->NumOccupiedState();

    if( eigVal.m() != npsi ){
      std::ostringstream msg;
      msg 
        << "The number of eigenstates do not match."  << std::endl
        << "eigVal         ~ " << eigVal.m() << std::endl
        << "numStateTotal  ~ " << npsi << std::endl;
      ErrorHandling( msg.str().c_str() );
    }

    if( occupationRate.m() != npsi ) occupationRate.Resize( npsi );

    DblNumVec eigValTotal( npsi );
    blas::Copy( npsi, eigVal.Data(), 1, eigValTotal.Data(), 1 );

    Sort( eigValTotal );

    if( npsi == nOccStates ){
      for( Int j = 0; j < npsi; j++ ){
         occupationRate(j) = 1.0;
      }
      fermi_ = eigValTotal(npsi-1);
    }
    else if( npsi > nOccStates ){
      if( esdfParam.temperature == 0.0 ){
        fermi_ = eigValTotal(nOccStates-1);
        for( Int j = 0; j < npsi; j++ ){
          if( eigVal[j] <= fermi_ ){
            occupationRate(j) = 1.0;
          }
          else{
            occupationRate(j) = 0.0;
          }
        }
      }
      else{
        Real lb, ub, flb, fub, occsum;
        Int ilb, iub, iter;

        ilb = 1;
        iub = npsi;
        lb = eigValTotal(ilb-1);
        ub = eigValTotal(iub-1);

        fermi_ = (lb+ub)*0.5;
        occsum = 0.0;
        for(Int j = 0; j < npsi; j++){
          occsum += 1.0 / (1.0 + exp(Tbeta_*(eigValTotal(j) - fermi_)));
        }

        iter = 1;
        while( (fabs(occsum - nOccStates) > tol) && (iter < maxiter) ) {
          if( occsum < nOccStates ) {lb = fermi_;}
          else {ub = fermi_;}

          fermi_ = (lb+ub)*0.5;
          occsum = 0.0;
          for(Int j = 0; j < npsi; j++){
            occsum += 1.0 / (1.0 + exp(Tbeta_*(eigValTotal(j) - fermi_)));
          }
          iter++;
        }

        for(Int j = 0; j < npsi; j++){
          occupationRate(j) = 1.0 / (1.0 + exp(Tbeta_*(eigVal(j) - fermi_)));
        }
      }
    }
    else{
      ErrorHandling( "The number of eigenvalues in ev should be larger than nocc" );
    }
  }
  else{
    // MP and GB type smearing
    Int npsi       = hamDGPtr_->NumStateTotal();
    Int nOccStates = hamDGPtr_->NumOccupiedState();

    if( eigVal.m() != npsi ){
      std::ostringstream msg;
      msg 
        << "The number of eigenstates do not match."  << std::endl
        << "eigVal         ~ " << eigVal.m() << std::endl
        << "numStateTotal  ~ " << npsi << std::endl;
      ErrorHandling( msg.str().c_str() );
    }

    if( occupationRate.m() != npsi ) 
      occupationRate.Resize( npsi );

    Real lb, ub, flb, fub, fx;
    Int  iter;

    if( npsi > nOccStates )  
    { 
      // Set up the bounds
      lb = eigVal(0);
      ub = eigVal(npsi - 1);

      // Set up the function bounds
      flb = mp_occupations_residual(eigVal, lb, nOccStates, Tsigma_, MP_smearing_order_);
      fub = mp_occupations_residual(eigVal, ub, nOccStates, Tsigma_, MP_smearing_order_);

      if(flb * fub > 0.0)
        ErrorHandling( "Bisection method for finding Fermi level cannot proceed !!" );

      fermi_ = (lb + ub) * 0.5;

      /* Start bisection iteration */
      iter = 1;
      fx = mp_occupations_residual(eigVal, fermi_, nOccStates,Tsigma_, MP_smearing_order_);

      // Iterate using the bisection method
      while( (fabs(fx) > tol) && (iter < maxiter) ) 
      {
        flb = mp_occupations_residual(eigVal, lb, nOccStates, Tsigma_, MP_smearing_order_);
        fub = mp_occupations_residual(eigVal, ub, nOccStates, Tsigma_, MP_smearing_order_);

        if( (flb * fx) < 0.0 )
          ub = fermi_;
        else
          lb = fermi_;

        fermi_ = (lb + ub) * 0.5;
        fx = mp_occupations_residual(eigVal, fermi_, nOccStates, Tsigma_, MP_smearing_order_);

        iter++;
      }

      if(iter >= maxiter)
        ErrorHandling( "Bisection method for finding Fermi level does not appear to converge !!" );
      else
      {
        // Bisection method seems to have converged
        // Fill up the occupations
        populate_mp_occupations(eigVal, occupationRate, fermi_, Tsigma_, MP_smearing_order_);
      }        
    } // end of if(npsi > nOccStates)
    else 
    {
      if (npsi == nOccStates ) 
      {
        for(Int j = 0; j < npsi; j++) 
          occupationRate(j) = 1.0;

          fermi_ = eigVal(npsi-1);
      }
      else 
      {
        // npsi < nOccStates
        ErrorHandling( "The number of top eigenvalues should be larger than number of occupied states !! " );
      }
    } // end of if(npsi > nOccStates) ... else
  } // end of if(SmearingScheme_ == "FD") ... else

  return ;
}         // -----  end of method SCFDG::CalculateOccupationRate  ----- 

void
SCFDG::CalculateOccupationRateExtElem    ( DblNumVec& eigVal, DblNumVec& occupationRate, Int npsi, Int nOccStates )
{

  if( eigVal.m() != npsi ){
    std::ostringstream msg;
    msg
      << "The number of eigenstates do not match."  << std::endl
      << "eigVal         ~ " << eigVal.m() << std::endl
      << "numStateTotal  ~ " << npsi << std::endl;
    ErrorHandling( msg.str().c_str() );
  }

  if( occupationRate.m() != npsi ) occupationRate.Resize( npsi );

  DblNumVec eigValTotal( npsi );
  blas::Copy( npsi, eigVal.Data(), 1, eigValTotal.Data(), 1 );

  Sort( eigValTotal );

  if( npsi == nOccStates ){
    for( Int j = 0; j < npsi; j++ ){
      occupationRate(j) = 1.0;
    }
    fermi_ = eigValTotal(npsi-1);
  }
  else if( npsi > nOccStates ){
    if( esdfParam.temperature == 0.0 ){
      fermi_ = eigValTotal(nOccStates-1);
      for( Int j = 0; j < npsi; j++ ){
        if( eigVal[j] <= fermi_ ){
          occupationRate(j) = 1.0;
        }
        else{
          occupationRate(j) = 0.0;
        }
      }
    }
    else{
      Real tol = 1e-16;
      Int maxiter = 200;

      Real lb, ub, flb, fub, occsum;
      Int ilb, iub, iter;

      ilb = 1;
      iub = npsi;
      lb = eigValTotal(ilb-1);
      ub = eigValTotal(iub-1);

      fermi_ = (lb+ub)*0.5;
      occsum = 0.0;
      for(Int j = 0; j < npsi; j++){
        occsum += 1.0 / (1.0 + exp(Tbeta_*(eigValTotal(j) - fermi_)));
      }

      iter = 1;
      while( (fabs(occsum - nOccStates) > tol) && (iter < maxiter) ) {
        if( occsum < nOccStates ) {lb = fermi_;}
        else {ub = fermi_;}

        fermi_ = (lb+ub)*0.5;
        occsum = 0.0;
        for(Int j = 0; j < npsi; j++){
          occsum += 1.0 / (1.0 + exp(Tbeta_*(eigValTotal(j) - fermi_)));
        }
        iter++;
      }

      for(Int j = 0; j < npsi; j++){
        occupationRate(j) = 1.0 / (1.0 + exp(Tbeta_*(eigVal(j) - fermi_)));
      }
    }
  }
  else{
    ErrorHandling( "The number of eigenvalues in ev should be larger than nocc" );
  }

  return ;
}         // -----  end of method SCFDG::CalculateOccupationRateExtElm  -----

void
SCFDG::InterpPeriodicUniformToLGL    ( 
    const Index3& numUniformGrid, 
    const Index3& numLGLGrid, 
    const Real*   psiUniform, 
    Real*         psiLGL )
{

  Index3 Ns1 = numUniformGrid;
  Index3 Ns2 = numLGLGrid;

  DblNumVec  tmp1( Ns2[0] * Ns1[1] * Ns1[2] );
  DblNumVec  tmp2( Ns2[0] * Ns2[1] * Ns1[2] );
  SetValue( tmp1, 0.0 );
  SetValue( tmp2, 0.0 );

  // x-direction, use Gemm
  {
    Int m = Ns2[0], n = Ns1[1] * Ns1[2], k = Ns1[0];
    blas::Gemm( 'N', 'N', m, n, k, 1.0, PeriodicUniformToLGLMat_[0].Data(),
        m, psiUniform, k, 0.0, tmp1.Data(), m );
  }

  // y-direction, use Gemv
  {
    Int   m = Ns2[1], n = Ns1[1];
    Int   ptrShift1, ptrShift2;
    Int   inc = Ns2[0];
    for( Int k = 0; k < Ns1[2]; k++ ){
      for( Int i = 0; i < Ns2[0]; i++ ){
        ptrShift1 = i + k * Ns2[0] * Ns1[1];
        ptrShift2 = i + k * Ns2[0] * Ns2[1];
        blas::Gemv( 'N', m, n, 1.0, 
            PeriodicUniformToLGLMat_[1].Data(), m, 
            tmp1.Data() + ptrShift1, inc, 0.0, 
            tmp2.Data() + ptrShift2, inc );
      } // for (i)
    } // for (k)
  }


  // z-direction, use Gemm
  {
    Int m = Ns2[0] * Ns2[1], n = Ns2[2], k = Ns1[2]; 
    blas::Gemm( 'N', 'T', m, n, k, 1.0, 
        tmp2.Data(), m, 
        PeriodicUniformToLGLMat_[2].Data(), n, 0.0, psiLGL, m );
  }


  return ;
}         // -----  end of method SCFDG::InterpPeriodicUniformToLGL  ----- 

void
SCFDG::InterpPeriodicUniformFineToLGL    ( 
    const Index3& numUniformGridFine, 
    const Index3& numLGLGrid, 
    const Real*   rhoUniform, 
    Real*         rhoLGL )
{

  Index3 Ns1 = numUniformGridFine;
  Index3 Ns2 = numLGLGrid;

  DblNumVec  tmp1( Ns2[0] * Ns1[1] * Ns1[2] );
  DblNumVec  tmp2( Ns2[0] * Ns2[1] * Ns1[2] );
  SetValue( tmp1, 0.0 );
  SetValue( tmp2, 0.0 );

  // x-direction, use Gemm
  {
    Int m = Ns2[0], n = Ns1[1] * Ns1[2], k = Ns1[0];
    blas::Gemm( 'N', 'N', m, n, k, 1.0, PeriodicUniformFineToLGLMat_[0].Data(),
        m, rhoUniform, k, 0.0, tmp1.Data(), m );
  }

  // y-direction, use Gemv
  {
    Int   m = Ns2[1], n = Ns1[1];
    Int   rhoShift1, rhoShift2;
    Int   inc = Ns2[0];
    for( Int k = 0; k < Ns1[2]; k++ ){
      for( Int i = 0; i < Ns2[0]; i++ ){
        rhoShift1 = i + k * Ns2[0] * Ns1[1];
        rhoShift2 = i + k * Ns2[0] * Ns2[1];
        blas::Gemv( 'N', m, n, 1.0, 
            PeriodicUniformFineToLGLMat_[1].Data(), m, 
            tmp1.Data() + rhoShift1, inc, 0.0, 
            tmp2.Data() + rhoShift2, inc );
      } // for (i)
    } // for (k)
  }


  // z-direction, use Gemm
  {
    Int m = Ns2[0] * Ns2[1], n = Ns2[2], k = Ns1[2]; 
    blas::Gemm( 'N', 'T', m, n, k, 1.0, 
        tmp2.Data(), m, 
        PeriodicUniformFineToLGLMat_[2].Data(), n, 0.0, rhoLGL, m );
  }


  return ;
}         // -----  end of method SCFDG::InterpPeriodicUniformFineToLGL  ----- 

void
SCFDG::InterpPeriodicGridExtElemToGridElem ( 
    const Index3& numUniformGridFineExtElem, 
    const Index3& numUniformGridFineElem, 
    const Real*   rhoUniformExtElem, 
    Real*         rhoUniformElem )
{

  Index3 Ns1 = numUniformGridFineExtElem;
  Index3 Ns2 = numUniformGridFineElem;

  DblNumVec  tmp1( Ns2[0] * Ns1[1] * Ns1[2] );
  DblNumVec  tmp2( Ns2[0] * Ns2[1] * Ns1[2] );
  SetValue( tmp1, 0.0 );
  SetValue( tmp2, 0.0 );

  // x-direction, use Gemm
  {
    Int m = Ns2[0], n = Ns1[1] * Ns1[2], k = Ns1[0];
    blas::Gemm( 'N', 'N', m, n, k, 1.0, PeriodicGridExtElemToGridElemMat_[0].Data(),
        m, rhoUniformExtElem, k, 0.0, tmp1.Data(), m );
  }

  // y-direction, use Gemv
  {
    Int   m = Ns2[1], n = Ns1[1];
    Int   rhoShift1, rhoShift2;
    Int   inc = Ns2[0];
    for( Int k = 0; k < Ns1[2]; k++ ){
      for( Int i = 0; i < Ns2[0]; i++ ){
        rhoShift1 = i + k * Ns2[0] * Ns1[1];
        rhoShift2 = i + k * Ns2[0] * Ns2[1];
        blas::Gemv( 'N', m, n, 1.0, 
            PeriodicGridExtElemToGridElemMat_[1].Data(), m, 
            tmp1.Data() + rhoShift1, inc, 0.0, 
            tmp2.Data() + rhoShift2, inc );
      } // for (i)
    } // for (k)
  }


  // z-direction, use Gemm
  {
    Int m = Ns2[0] * Ns2[1], n = Ns2[2], k = Ns1[2]; 
    blas::Gemm( 'N', 'T', m, n, k, 1.0, 
        tmp2.Data(), m, 
        PeriodicGridExtElemToGridElemMat_[2].Data(), n, 0.0, rhoUniformElem, m );
  }

  return ;
}         // -----  end of method SCFDG::InterpPeriodicGridExtElemToGridElem  ----- 

void
SCFDG::CalculateKSEnergy    (  )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  HamiltonianDG&  hamDG = *hamDGPtr_;

  DblNumVec&  eigVal         = hamDG.EigVal();
  DblNumVec&  occupationRate = hamDG.OccupationRate();

  // Band energy
  Int numSpin = hamDG.NumSpin();


  if(Begin_DG_TDDFT_==1)
{
//Ekin_=Ekin_;
//debug
//std::cout<<" Ekin mult: "<<Ekin_<<std::endl;

double Ekin2=0.0;
for (Int i=0; i < eigVal.m(); i++) {
  Ekin2  += numSpin * eigVal(i) * occupationRate(i);
}
//std::cout<<" Ekin2 : "<<Ekin2<<std::endl;
Ekin_=Ekin2;


//std::cout<<"Ekin_ "<<Ekin_<<std::endl;

////
}
  else if(SCFDG_comp_subspace_engaged_ == 1)
  {
    double HC_part = 0.0;

    for(Int sum_iter = 0; sum_iter < SCFDG_comp_subspace_N_solve_; sum_iter ++)
      HC_part += (1.0 - SCFDG_comp_subspace_top_occupations_(sum_iter)) * SCFDG_comp_subspace_top_eigvals_(sum_iter);

    Ekin_ = numSpin * (SCFDG_comp_subspace_trace_Hmat_ - HC_part) ;
  }
  else
  {  
    Ekin_ = 0.0;
    for (Int i=0; i < eigVal.m(); i++) {
      Ekin_  += numSpin * eigVal(i) * occupationRate(i);
    }
  }

  // Self energy part
  Eself_ = 0.0;
  std::vector<Atom>&  atomList = hamDG.AtomList();
  for(Int a=0; a< atomList.size() ; a++) {
    Int type = atomList[a].type;
    Eself_ += ptablePtr_->SelfIonInteraction(type);
  }

  // Hartree and XC part
  Ehart_ = 0.0;
  EVxc_  = 0.0;

  Real EhartLocal = 0.0, EVxcLocal = 0.0;

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          DblNumVec&  density      = hamDG.Density().LocalMap()[key];
          DblNumVec&  vxc          = hamDG.Vxc().LocalMap()[key];
          DblNumVec&  pseudoCharge = hamDG.PseudoCharge().LocalMap()[key];
          DblNumVec&  vhart        = hamDG.Vhart().LocalMap()[key];

          for (Int p=0; p < density.Size(); p++) {
            EVxcLocal  += vxc(p) * density(p);
            EhartLocal += 0.5 * vhart(p) * ( density(p) + pseudoCharge(p) );
          }

        } // own this element
      } // for (i)

  mpi::Allreduce( &EVxcLocal, &EVxc_, 1, MPI_SUM, domain_.colComm );
  mpi::Allreduce( &EhartLocal, &Ehart_, 1, MPI_SUM, domain_.colComm );

  Ehart_ *= domain_.Volume() / domain_.NumGridTotalFine();
  EVxc_  *= domain_.Volume() / domain_.NumGridTotalFine();

  // Correction energy
  Ecor_   = (Exc_ - Ehfx_ - EVxc_) - Ehart_ - Eself_;
  if( esdfParam.isUseVLocal == true ){
    Ecor_ += hamDG.EIonSR();
  }

  // Total energy
  Etot_ = Ekin_ + Ecor_;

  // Helmholtz free energy
  if( hamDG.NumOccupiedState() == 
      hamDG.NumStateTotal() ){
    // Zero temperature
    Efree_ = Etot_;
  }
  else{
    // Finite temperature
    Efree_ = 0.0;
    Real fermi = fermi_;
    Real Tbeta = Tbeta_;

    if(SCFDG_comp_subspace_engaged_ == 1)
    {
      double occup_energy_part = 0.0;
      double occup_tol = 1e-12;
      double fl;
      for(Int l=0; l < SCFDG_comp_subspace_top_occupations_.m(); l++)
      {
        fl = SCFDG_comp_subspace_top_occupations_(l);
        if((fl > occup_tol) && ((1.0 - fl) > occup_tol))
          occup_energy_part += fl * log(fl) + (1.0 - fl) * log(1 - fl);
      }

      Efree_ = Ekin_ + Ecor_ + (numSpin / Tbeta) * occup_energy_part;
    }
    else
    {  
      for(Int l=0; l< eigVal.m(); l++) {
        Real eig = eigVal(l);
        if( eig - fermi >= 0){
          Efree_ += -numSpin /Tbeta*log(1.0+exp(-Tbeta*(eig - fermi))); 
        }
        else{
          Efree_ += numSpin * (eig - fermi) - numSpin / Tbeta*log(1.0+exp(Tbeta*(eig-fermi)));
        } 
      }

      Efree_ += Ecor_ + fermi * hamDG.NumOccupiedState() * numSpin; 
    }
  }

  return ;
}         // -----  end of method SCFDG::CalculateKSEnergy  ----- 

void
  SCFDG::CalculateKSEnergyDM (
      Real totalEnergyH,
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distEDMMat,
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distFDMMat )
  {
    Int mpirank, mpisize;
    MPI_Comm_rank( domain_.comm, &mpirank );
    MPI_Comm_size( domain_.comm, &mpisize );

    HamiltonianDG&  hamDG = *hamDGPtr_;

    DblNumVec&  eigVal         = hamDG.EigVal();
    DblNumVec&  occupationRate = hamDG.OccupationRate();

    // Kinetic energy
    Int numSpin = hamDG.NumSpin();

    // Self energy part
    Eself_ = 0.0;
    std::vector<Atom>&  atomList = hamDG.AtomList();
    for(Int a=0; a< atomList.size() ; a++) {
      Int type = atomList[a].type;
      Eself_ += ptablePtr_->SelfIonInteraction(type);
    }

    // Hartree and XC part
    Ehart_ = 0.0;
    EVxc_  = 0.0;

    Real EhartLocal = 0.0, EVxcLocal = 0.0;

    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumVec&  density      = hamDG.Density().LocalMap()[key];
            DblNumVec&  vxc          = hamDG.Vxc().LocalMap()[key];
            DblNumVec&  pseudoCharge = hamDG.PseudoCharge().LocalMap()[key];
            DblNumVec&  vhart        = hamDG.Vhart().LocalMap()[key];

            for (Int p=0; p < density.Size(); p++) {
              EVxcLocal  += vxc(p) * density(p);
              EhartLocal += 0.5 * vhart(p) * ( density(p) + pseudoCharge(p) );
            }

          } // own this element
        } // for (i)

    mpi::Allreduce( &EVxcLocal, &EVxc_, 1, MPI_SUM, domain_.colComm );
    mpi::Allreduce( &EhartLocal, &Ehart_, 1, MPI_SUM, domain_.colComm );

    Ehart_ *= domain_.Volume() / domain_.NumGridTotalFine();
    EVxc_  *= domain_.Volume() / domain_.NumGridTotalFine();

    // Correction energy
    Ecor_   = (Exc_ - Ehfx_ - EVxc_) - Ehart_ - Eself_;
    if( esdfParam.isUseVLocal == true ){
      Ecor_ += hamDG.EIonSR();
    }

    // Kinetic energy and helmholtz free energy, calculated from the
    // energy and free energy density matrices.
    // Here 
    // 
    //   Ekin = Tr[H 2/(1+exp(beta(H-mu)))] 
    // and
    //   Ehelm = -2/beta Tr[log(1+exp(mu-H))] + mu*N_e
    // FIXME Put the above documentation to the proper place like the hpp
    // file

    Real Ehelm = 0.0, EhelmLocal = 0.0, EkinLocal = 0.0;

    // FIXME Ekin is not used later.
    if( 1 ) {
      // Compute the trace of the energy density matrix in each element
      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){

              DblNumMat& localEDM = distEDMMat.LocalMap()[
                ElemMatKey(key, key)];
              DblNumMat& localFDM = distFDMMat.LocalMap()[
                ElemMatKey(key, key)];

              for( Int a = 0; a < localEDM.m(); a++ ){
                EkinLocal  += localEDM(a,a);
                EhelmLocal += localFDM(a,a);
              }
            } // own this element
          } // for (i)

      // Reduce the results 
      mpi::Allreduce( &EkinLocal, &Ekin_, 
          1, MPI_SUM, domain_.colComm );

      mpi::Allreduce( &EhelmLocal, &Ehelm, 
          1, MPI_SUM, domain_.colComm );

      // Add the mu*N term for the free energy
      Ehelm += fermi_ * hamDG.NumOccupiedState() * numSpin;

    }

    // FIXME In order to be compatible with PPEXSIDFTDriver3, the
    // Tr[H*DM] part is directly read from totalEnergyH
    Ekin_ = totalEnergyH;

    // Total energy
    Etot_ = Ekin_ + Ecor_;

    // Free energy at finite temperature
    // FIXME PPEXSIDFTDriver3 does not have free energy
    Ehelm = totalEnergyH;
    Efree_ = Ehelm + Ecor_;
    


    return ;
  }         // -----  end of method SCFDG::CalculateKSEnergyDM  ----- 

void
SCFDG::CalculateHarrisEnergy    (  )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  HamiltonianDG&  hamDG = *hamDGPtr_;

  DblNumVec&  eigVal         = hamDG.EigVal();
  DblNumVec&  occupationRate = hamDG.OccupationRate();

  // NOTE: To avoid confusion, all energies in this routine are
  // temporary variables other than EfreeHarris_.
  //
  // The related energies will be computed again in the routine
  //
  // CalculateKSEnergy()

  Real Ekin, Eself, Ehart, EVxc, Exc, Exx, Ecor;

  // Kinetic energy from the new density matrix.
  Int numSpin = hamDG.NumSpin();
  Ekin = 0.0;

  if(SCFDG_comp_subspace_engaged_ == 1)
  {
    // This part is the same irrespective of smearing type
    double HC_part = 0.0;

    for(Int sum_iter = 0; sum_iter < SCFDG_comp_subspace_N_solve_; sum_iter ++)
      HC_part += (1.0 - SCFDG_comp_subspace_top_occupations_(sum_iter)) * SCFDG_comp_subspace_top_eigvals_(sum_iter);

    Ekin = numSpin * (SCFDG_comp_subspace_trace_Hmat_ - HC_part) ;
  }
  else
  {  
    for (Int i=0; i < eigVal.m(); i++) {
      Ekin  += numSpin * eigVal(i) * occupationRate(i);
    }
  }
  // Self energy part
  Eself = 0.0;
  std::vector<Atom>&  atomList = hamDG.AtomList();
  for(Int a=0; a< atomList.size() ; a++) {
    Int type = atomList[a].type;
    Eself += ptablePtr_->SelfIonInteraction(type);
  }

  // Nonlinear correction part.  This part uses the Hartree energy and
  // XC correlation energy from the old electron density.

  Real EhartLocal = 0.0, EVxcLocal = 0.0;

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          DblNumVec&  density      = hamDG.Density().LocalMap()[key];
          DblNumVec&  vxc          = hamDG.Vxc().LocalMap()[key];
          DblNumVec&  pseudoCharge = hamDG.PseudoCharge().LocalMap()[key];
          DblNumVec&  vhart        = hamDG.Vhart().LocalMap()[key];

          for (Int p=0; p < density.Size(); p++) {
            EVxcLocal  += vxc(p) * density(p);
            EhartLocal += 0.5 * vhart(p) * ( density(p) + pseudoCharge(p) );
          }

        } // own this element
      } // for (i)

  mpi::Allreduce( &EVxcLocal, &EVxc, 1, MPI_SUM, domain_.colComm );
  mpi::Allreduce( &EhartLocal, &Ehart, 1, MPI_SUM, domain_.colComm );

  Ehart *= domain_.Volume() / domain_.NumGridTotalFine();
  EVxc  *= domain_.Volume() / domain_.NumGridTotalFine();
  // Use the previous exchange-correlation energy
  Exc    = Exc_;

  // Correction energy.  
  Ecor   = (Exc - Ehfx_ - EVxc) - Ehart - Eself;
  if( esdfParam.isUseVLocal == true ){
    Ecor  += hamDG.EIonSR();
  }

  // Harris free energy functional
  if( hamDG.NumOccupiedState() == 
      hamDG.NumStateTotal() ){
    // Zero temperature
    EfreeHarris_ = Ekin + Ecor;
  }
  else{
    // Finite temperature
    EfreeHarris_ = 0.0;
    Real fermi = fermi_;
    Real Tbeta = Tbeta_;

    if(SCFDG_comp_subspace_engaged_ == 1)
    {
      // Complementary subspace technique in use

      double occup_energy_part = 0.0;
      double occup_tol = 1e-12;
      double fl, x;

      if(SmearingScheme_ == "FD")
      {
        for(Int l=0; l < SCFDG_comp_subspace_top_occupations_.m(); l++)
        {
          fl = SCFDG_comp_subspace_top_occupations_(l);
          if((fl > occup_tol) && ((1.0 - fl) > occup_tol))
            occup_energy_part += fl * log(fl) + (1.0 - fl) * log(1 - fl);
        }

        EfreeHarris_ = Ekin + Ecor + (numSpin / Tbeta) * occup_energy_part;
      }
      else
      {
        // Other kinds of smearing

        for(Int l=0; l < SCFDG_comp_subspace_top_occupations_.m(); l++)
        {
          fl = SCFDG_comp_subspace_top_occupations_(l);
          if((fl > occup_tol) && ((1.0 - fl) > occup_tol))
          {
            x = (SCFDG_comp_subspace_top_eigvals_(l) - fermi_) / Tsigma_ ;
            occup_energy_part += mp_entropy(x, MP_smearing_order_);
          }
        }

        EfreeHarris_ = Ekin + Ecor + (numSpin / Tbeta) * occup_energy_part;

      }
    }  
    else
    { 
      // Complementary subspace technique not in use : full spectrum available
      if(SmearingScheme_ == "FD")
      {
        for(Int l=0; l< eigVal.m(); l++) {
          Real eig = eigVal(l);
          if( eig - fermi >= 0){
            EfreeHarris_ += -numSpin /Tbeta*log(1.0+exp(-Tbeta*(eig - fermi))); 
          }
          else{
            EfreeHarris_ += numSpin * (eig - fermi) - numSpin / Tbeta*log(1.0+exp(Tbeta*(eig-fermi)));
          }
        }
        EfreeHarris_ += Ecor + fermi * hamDG.NumOccupiedState() * numSpin; 
      }
      else
      {
        // GB or MP schemes in use
        double occup_energy_part = 0.0;
        double occup_tol = 1e-12;
        double fl, x;

        for(Int l=0; l < eigVal.m(); l++)
        {
          fl = occupationRate(l);
          if((fl > occup_tol) && ((1.0 - fl) > occup_tol))
          { 
            x = (eigVal(l) - fermi_) / Tsigma_ ;
            occup_energy_part += mp_entropy(x, MP_smearing_order_) ;
          }
        }

        EfreeHarris_ = Ekin + Ecor + (numSpin * Tsigma_) * occup_energy_part;

      }

    } // end of full spectrum available calculation
  } // end of finite temperature calculation

  return ;
}         // -----  end of method SCFDG::CalculateHarrisEnergy  ----- 

void
SCFDG::CalculateHarrisEnergyDM(
    Real totalFreeEnergy,
    DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distFDMMat )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  HamiltonianDG&  hamDG = *hamDGPtr_;

  // NOTE: To avoid confusion, all energies in this routine are
  // temporary variables other than EfreeHarris_.
  //
  // The related energies will be computed again in the routine
  //
  // CalculateKSEnergy()

  Real Ehelm, Eself, Ehart, EVxc, Exc, Exx, Ecor;

  Int numSpin = hamDG.NumSpin();

  // Self energy part
  Eself = 0.0;
  std::vector<Atom>&  atomList = hamDG.AtomList();
  for(Int a=0; a< atomList.size() ; a++) {
    Int type = atomList[a].type;
    Eself += ptablePtr_->SelfIonInteraction(type);
  }

  // Nonlinear correction part.  This part uses the Hartree energy and
  // XC correlation energy from the old electron density.

  Real EhartLocal = 0.0, EVxcLocal = 0.0;

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          DblNumVec&  density      = hamDG.Density().LocalMap()[key];
          DblNumVec&  vxc          = hamDG.Vxc().LocalMap()[key];
          DblNumVec&  pseudoCharge = hamDG.PseudoCharge().LocalMap()[key];
          DblNumVec&  vhart        = hamDG.Vhart().LocalMap()[key];

          for (Int p=0; p < density.Size(); p++) {
            EVxcLocal  += vxc(p) * density(p);
            EhartLocal += 0.5 * vhart(p) * ( density(p) + pseudoCharge(p) );
          }
        } // own this element
      } // for (i)

  mpi::Allreduce( &EVxcLocal, &EVxc, 1, MPI_SUM, domain_.colComm );
  mpi::Allreduce( &EhartLocal, &Ehart, 1, MPI_SUM, domain_.colComm );

  Ehart *= domain_.Volume() / domain_.NumGridTotalFine();
  EVxc  *= domain_.Volume() / domain_.NumGridTotalFine();
  // Use the previous exchange-correlation energy
  Exc    = Exc_;

  // Correction energy.  
  Ecor   = (Exc - Ehfx_ - EVxc) - Ehart - Eself;
  if( esdfParam.isUseVLocal == true ){
    Ecor  += hamDG.EIonSR();
  }

  // The Helmholtz part of the free energy
  // Ehelm = -2/beta Tr[log(1+exp(mu-H))] + mu*N_e
  // FIXME Put the above documentation to the proper place like the hpp
  // file
  Real EhelmLocal = 0.0;
  Ehelm = 0.0;

  // Compute the trace of the energy density matrix in each element
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          DblNumMat& localFDM = distFDMMat.LocalMap()[
          ElemMatKey(key, key)];

          for( Int a = 0; a < localFDM.m(); a++ ){
            EhelmLocal += localFDM(a,a);
          }
        } // own this element
  } // for (i)

  mpi::Allreduce( &EhelmLocal, &Ehelm, 
      1, MPI_SUM, domain_.colComm );

  // Add the mu*N term
  Ehelm += fermi_ * hamDG.NumOccupiedState() * numSpin;

  // Harris free energy functional. This has to be the finite
  // temperature formulation

  // FIXME
  Ehelm = totalFreeEnergy;
  EfreeHarris_ = Ehelm + Ecor;

  return ;
}         // -----  end of method SCFDG::CalculateHarrisEnergyDM  ----- 

void
SCFDG::CalculateSecondOrderEnergy  (  )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  HamiltonianDG&  hamDG = *hamDGPtr_;

  DblNumVec&  eigVal         = hamDG.EigVal();
  DblNumVec&  occupationRate = hamDG.OccupationRate();

  // NOTE: To avoid confusion, all energies in this routine are
  // temporary variables other than EfreeSecondOrder_.
  // 
  // This is similar to the situation in 
  //
  // CalculateHarrisEnergy()

  Real Ekin, Eself, Ehart, EVtot, Exc, Exx, Ecor;

  // Kinetic energy from the new density matrix.
  Int numSpin = hamDG.NumSpin();
  //Ekin = 0.0;

   if(Begin_DG_TDDFT_==1)
 {
double Ekin2=0.0;
for (Int i=0; i < eigVal.m(); i++) {
  Ekin2  += numSpin * eigVal(i) * occupationRate(i);
}
//std::cout<<" Ekin2 : "<<Ekin2<<std::endl;
Ekin_=Ekin2;
//


 }
  else if(SCFDG_comp_subspace_engaged_ == 1)
  {
	Ekin_=0.0;
    // This part is the same, irrespective of smearing type
    double HC_part = 0.0;

    for(Int sum_iter = 0; sum_iter < SCFDG_comp_subspace_N_solve_; sum_iter ++)
      HC_part += (1.0 - SCFDG_comp_subspace_top_occupations_(sum_iter)) * SCFDG_comp_subspace_top_eigvals_(sum_iter);

    Ekin = numSpin * (SCFDG_comp_subspace_trace_Hmat_ - HC_part) ;
  }
  else
  {
Ekin_=0.0;  
    for (Int i=0; i < eigVal.m(); i++) {
      Ekin  += numSpin * eigVal(i) * occupationRate(i);
    }
  }

  // Self energy part
  Eself = 0.0;
  std::vector<Atom>&  atomList = hamDG.AtomList();
  for(Int a=0; a< atomList.size() ; a++) {
    Int type = atomList[a].type;
    Eself_ += ptablePtr_->SelfIonInteraction(type);
  }

  // Nonlinear correction part.  This part uses the Hartree energy and
  // XC correlation energy from the OUTPUT electron density, but the total
  // potential is the INPUT one used in the diagonalization process.
  // The density is also the OUTPUT density.
  //
  // NOTE the sign flip in Ehart, which is different from those in KS
  // energy functional and Harris energy functional.

  Real EhartLocal = 0.0, EVtotLocal = 0.0;

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          DblNumVec&  density      = hamDG.Density().LocalMap()[key];
          DblNumVec&  vext         = hamDG.Vext().LocalMap()[key];
          DblNumVec&  vtot         = hamDG.Vtot().LocalMap()[key];
          DblNumVec&  pseudoCharge = hamDG.PseudoCharge().LocalMap()[key];
          DblNumVec&  vhart        = hamDG.Vhart().LocalMap()[key];

          for (Int p=0; p < density.Size(); p++) {
            EVtotLocal  += (vtot(p) - vext(p)) * density(p);
            // NOTE the sign flip
            EhartLocal  += 0.5 * vhart(p) * ( density(p) - pseudoCharge(p) );
          }
        } // own this element
      } // for (i)

  mpi::Allreduce( &EVtotLocal, &EVtot, 1, MPI_SUM, domain_.colComm );
  mpi::Allreduce( &EhartLocal, &Ehart, 1, MPI_SUM, domain_.colComm );

  Ehart *= domain_.Volume() / domain_.NumGridTotalFine();
  EVtot *= domain_.Volume() / domain_.NumGridTotalFine();

  // Use the exchange-correlation energy with respect to the new
  // electron density
  Exc = Exc_;
  // Correction energy.  
  // NOTE The correction energy in the second order method means
  // differently from that in Harris energy functional or the KS energy
  // functional.
  Ecor   = (Exc - Ehfx_ + Ehart - Eself) - EVtot;
  // FIXME
  //    statusOFS
  //        << "Component energy for second order correction formula = " << std::endl
  //        << "Exc     = " << Exc      << std::endl
  //        << "Ehart   = " << Ehart    << std::endl
  //        << "Eself   = " << Eself    << std::endl
  //        << "EVtot   = " << EVtot    << std::endl
  //        << "Ecor    = " << Ecor     << std::endl;
  //    

  // Second order accurate free energy functional
  if( hamDG.NumOccupiedState() == 
      hamDG.NumStateTotal() ){
    // Zero temperature
    EfreeSecondOrder_ = Ekin + Ecor;
  }
  else{
    // Finite temperature
    EfreeSecondOrder_ = 0.0;
    Real fermi = fermi_;
    Real Tbeta = Tbeta_;

    if(SCFDG_comp_subspace_engaged_ == 1)
    {
      double occup_energy_part = 0.0;
      double occup_tol = 1e-12;
      double fl, x;

      if(SmearingScheme_ == "FD")
      {
        for(Int l = 0; l < SCFDG_comp_subspace_top_occupations_.m(); l++)
        {
          fl = SCFDG_comp_subspace_top_occupations_(l);
          if((fl > occup_tol) && ((1.0 - fl) > occup_tol))
            occup_energy_part += fl * log(fl) + (1.0 - fl) * log(1 - fl);
        }

        EfreeSecondOrder_ = Ekin + Ecor + (numSpin / Tbeta) * occup_energy_part;
      }
      else
      {
        // MP and GB smearing    
        for(Int l = 0; l < SCFDG_comp_subspace_top_occupations_.m(); l++)
        {
          fl = SCFDG_comp_subspace_top_occupations_(l);
          if((fl > occup_tol) && ((1.0 - fl) > occup_tol))
          {
            x = (SCFDG_comp_subspace_top_eigvals_(l) - fermi_) / Tsigma_ ;
            occup_energy_part += mp_entropy(x, MP_smearing_order_);
          }
        }

        EfreeSecondOrder_ = Ekin + Ecor + (numSpin / Tbeta) * occup_energy_part;            
      }
    }
    else
    {
      // Complementary subspace technique not in use : full spectrum available
      if(SmearingScheme_ == "FD")
      {
        for(Int l=0; l< eigVal.m(); l++) {
          Real eig = eigVal(l);
          if( eig - fermi >= 0){
            EfreeSecondOrder_ += -numSpin /Tbeta*log(1.0+exp(-Tbeta*(eig - fermi))); 
          }
          else{
            EfreeSecondOrder_ += numSpin * (eig - fermi) - numSpin / Tbeta*log(1.0+exp(Tbeta*(eig-fermi)));
          }
        }

        EfreeSecondOrder_ += Ecor + fermi * hamDG.NumOccupiedState() * numSpin; 
      }
      else
      {
        // GB or MP schemes in use
        double occup_energy_part = 0.0;
        double occup_tol = 1e-12;
        double fl, x;

        for(Int l=0; l < eigVal.m(); l++)
        {
          fl = occupationRate(l);
          if((fl > occup_tol) && ((1.0 - fl) > occup_tol))
          { 
            x = (eigVal(l) - fermi_) / Tsigma_ ;
            occup_energy_part += mp_entropy(x, MP_smearing_order_) ;
          }
        }

        EfreeSecondOrder_ = Ekin + Ecor + (numSpin * Tsigma_) * occup_energy_part;
      }  // end of full spectrum available calculation
    }
  } // end of finite temperature calculation

  return ;
}         // -----  end of method SCFDG::CalculateSecondOrderEnergy  ----- 

void
SCFDG::CalculateVDW    ( Real& VDWEnergy, DblNumMat& VDWForce )
{
  HamiltonianDG&  hamDG = *hamDGPtr_;
  std::vector<Atom>& atomList = hamDG.AtomList();
  Evdw_ = 0.0;
  forceVdw_.Resize( atomList.size(), DIM );
  SetValue( forceVdw_, 0.0 );

  Int numAtom = atomList.size();

  Domain& dm = domain_;

  if( VDWType_ == "DFT-D2"){

    const Int vdw_nspecies = 55;
    Int ia,is1,is2,is3,itypat,ja,jtypat,npairs,nshell;
    bool need_gradient,newshell;
    const Real vdw_d = 20.0;
    const Real vdw_tol_default = 1e-10;
    const Real vdw_s_pbe = 0.75;
    Real c6,c6r6,ex,fr,fred1,fred2,fred3,gr,grad,r0,r1,r2,r3,rcart1,rcart2,rcart3;

    double vdw_c6_dftd2[vdw_nspecies] = 
    {  0.14, 0.08, 1.61, 1.61, 3.13, 1.75, 1.23, 0.70, 0.75, 0.63,
      5.71, 5.71,10.79, 9.23, 7.84, 5.57, 5.07, 4.61,10.80,10.80,
      10.80,10.80,10.80,10.80,10.80,10.80,10.80,10.80,10.80,10.80,
      16.99,17.10,16.37,12.64,12.47,12.01,24.67,24.67,24.67,24.67,
      24.67,24.67,24.67,24.67,24.67,24.67,24.67,24.67,37.32,38.71,
      38.44,31.74,31.50,29.99, 0.00 };

    double vdw_r0_dftd2[vdw_nspecies] =
    { 1.001,1.012,0.825,1.408,1.485,1.452,1.397,1.342,1.287,1.243,
      1.144,1.364,1.639,1.716,1.705,1.683,1.639,1.595,1.485,1.474,
      1.562,1.562,1.562,1.562,1.562,1.562,1.562,1.562,1.562,1.562,
      1.650,1.727,1.760,1.771,1.749,1.727,1.628,1.606,1.639,1.639,
      1.639,1.639,1.639,1.639,1.639,1.639,1.639,1.639,1.672,1.804,
      1.881,1.892,1.892,1.881,1.000 };

    for(Int i=0; i<vdw_nspecies; i++) {
      vdw_c6_dftd2[i] = vdw_c6_dftd2[i] / 2625499.62 * pow(10/0.52917706, 6);
      vdw_r0_dftd2[i] = vdw_r0_dftd2[i] / 0.52917706;
    }

    DblNumMat vdw_c6(vdw_nspecies, vdw_nspecies);
    DblNumMat vdw_r0(vdw_nspecies, vdw_nspecies);
    SetValue( vdw_c6, 0.0 );
    SetValue( vdw_r0, 0.0 );

    for(Int i=0; i<vdw_nspecies; i++) {
      for(Int j=0; j<vdw_nspecies; j++) {
        vdw_c6(i,j) = std::sqrt( vdw_c6_dftd2[i] * vdw_c6_dftd2[j] );
        vdw_r0(i,j) = vdw_r0_dftd2[i] + vdw_r0_dftd2[j];
      }
    }

    Real vdw_s;
    if (XCType_ == "XC_GGA_XC_PBE") {
      vdw_s=vdw_s_pbe;
    }
    else {
      ErrorHandling( "Van der Waals DFT-D2 correction in only compatible with GGA-PBE!" );
    }

    for(Int ii=-1; ii<2; ii++) {
      for(Int jj=-1; jj<2; jj++) {
        for(Int kk=-1; kk<2; kk++) {

          for(Int i=0; i<atomList.size(); i++) {
            Int iType = atomList[i].type;
            for(Int j=0; j<(i+1); j++) {
              Int jType = atomList[j].type;

              Real rx = atomList[i].pos[0] - atomList[j].pos[0] + ii * dm.length[0];
              Real ry = atomList[i].pos[1] - atomList[j].pos[1] + jj * dm.length[1];
              Real rz = atomList[i].pos[2] - atomList[j].pos[2] + kk * dm.length[2];
              Real rr = std::sqrt( rx * rx + ry * ry + rz * rz );

              if ( ( rr > 0.0001 ) && ( rr < 75.0 ) ) {

                Real sfact = vdw_s;
                if ( i == j ) sfact = sfact * 0.5;

                Real c6 = vdw_c6(iType-1, jType-1);
                Real r0 = vdw_r0(iType-1, jType-1);

                Real ex = exp( -vdw_d * ( rr / r0 - 1 ));
                Real fr = 1.0 / ( 1.0 + ex );
                Real c6r6 = c6 / pow(rr, 6.0);

                // Contribution to energy
                Evdw_ = Evdw_ - sfact * fr * c6r6;

                // Contribution to force
                if( i != j ) {

                  Real gr = ( vdw_d / r0 ) * ( fr * fr ) * ex;
                  Real grad = sfact * ( gr - 6.0 * fr / rr ) * c6r6 / rr; 

                  Real fx = grad * rx;
                  Real fy = grad * ry;
                  Real fz = grad * rz;

                  forceVdw_( i, 0 ) = forceVdw_( i, 0 ) + fx; 
                  forceVdw_( i, 1 ) = forceVdw_( i, 1 ) + fy; 
                  forceVdw_( i, 2 ) = forceVdw_( i, 2 ) + fz; 
                  forceVdw_( j, 0 ) = forceVdw_( j, 0 ) - fx; 
                  forceVdw_( j, 1 ) = forceVdw_( j, 1 ) - fy; 
                  forceVdw_( j, 2 ) = forceVdw_( j, 2 ) - fz; 

                } // end for i != j

              } // end if

            } // end for j
          } // end for i

        } // end for ii
      } // end for jj
    } // end for kk
  } // If DFT-D2

  VDWEnergy = Evdw_;
  VDWForce = forceVdw_;

  return ;
}         // -----  end of method SCFDG::CalculateVDW  ----- 

void
SCFDG::AndersonMix    ( 
    Int             iter, 
    Real            mixStepLength,
    std::string     mixType,
    DistDblNumVec&  distvMix,
    DistDblNumVec&  distvOld,
    DistDblNumVec&  distvNew,
    DistDblNumMat&  dfMat,
    DistDblNumMat&  dvMat )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  distvMix.SetComm(domain_.colComm);
  distvOld.SetComm(domain_.colComm);
  distvNew.SetComm(domain_.colComm);
  dfMat.SetComm(domain_.colComm);
  dvMat.SetComm(domain_.colComm);

  // Residual 
  DistDblNumVec distRes;
  // Optimal input potential in Anderon mixing.
  DistDblNumVec distvOpt; 
  // Optimal residual in Anderson mixing
  DistDblNumVec distResOpt; 
  // Preconditioned optimal residual in Anderson mixing
  DistDblNumVec distPrecResOpt;

  distRes.SetComm(domain_.colComm);
  distvOpt.SetComm(domain_.colComm);
  distResOpt.SetComm(domain_.colComm);
  distPrecResOpt.SetComm(domain_.colComm);

  // *********************************************************************
  // Initialize
  // *********************************************************************
  Int ntot  = hamDGPtr_->NumUniformGridElemFine().prod();

  // Number of iterations used, iter should start from 1
  Int iterused = std::min( iter-1, mixMaxDim_ ); 
  // The current position of dfMat, dvMat
  Int ipos = iter - 1 - ((iter-2)/ mixMaxDim_ ) * mixMaxDim_;
  // The next position of dfMat, dvMat
  Int inext = iter - ((iter-1)/ mixMaxDim_) * mixMaxDim_;

  distRes.Prtn()          = elemPrtn_;
  distvOpt.Prtn()         = elemPrtn_;
  distResOpt.Prtn()       = elemPrtn_;
  distPrecResOpt.Prtn()   = elemPrtn_;

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          DblNumVec  emptyVec( ntot );
          SetValue( emptyVec, 0.0 );
          distRes.LocalMap()[key]        = emptyVec;
          distvOpt.LocalMap()[key]       = emptyVec;
          distResOpt.LocalMap()[key]     = emptyVec;
          distPrecResOpt.LocalMap()[key] = emptyVec;
        } // if ( own this element )
      } // for (i)

  // *********************************************************************
  // Anderson mixing
  // *********************************************************************

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          // res(:) = vOld(:) - vNew(:) is the residual
          distRes.LocalMap()[key] = distvOld.LocalMap()[key];
          blas::Axpy( ntot, -1.0, distvNew.LocalMap()[key].Data(), 1, 
              distRes.LocalMap()[key].Data(), 1 );

          distvOpt.LocalMap()[key]   = distvOld.LocalMap()[key];
          distResOpt.LocalMap()[key] = distRes.LocalMap()[key];


          // dfMat(:, ipos-1) = res(:) - dfMat(:, ipos-1);
          // dvMat(:, ipos-1) = vOld(:) - dvMat(:, ipos-1);
          if( iter > 1 ){
            blas::Scal( ntot, -1.0, dfMat.LocalMap()[key].VecData(ipos-1), 1 );
            blas::Axpy( ntot, 1.0,  distRes.LocalMap()[key].Data(), 1, 
                dfMat.LocalMap()[key].VecData(ipos-1), 1 );
            blas::Scal( ntot, -1.0, dvMat.LocalMap()[key].VecData(ipos-1), 1 );
            blas::Axpy( ntot, 1.0,  distvOld.LocalMap()[key].Data(),  1, 
                dvMat.LocalMap()[key].VecData(ipos-1), 1 );
          }
        } // own this element
      } // for (i)

  // For iter == 1, Anderson mixing is the same as simple mixing. 
  if( iter > 1){

    Int nrow = iterused;

    // Normal matrix FTF = F^T * F
    DblNumMat FTFLocal( nrow, nrow ), FTF( nrow, nrow );
    SetValue( FTFLocal, 0.0 );
    SetValue( FTF, 0.0 );

    // Right hand side FTv = F^T * vout
    DblNumVec FTvLocal( nrow ), FTv( nrow );
    SetValue( FTvLocal, 0.0 );
    SetValue( FTv, 0.0 );

    // Local construction of FTF and FTv
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumMat& df     = dfMat.LocalMap()[key];
            DblNumVec& res    = distRes.LocalMap()[key];
            for( Int q = 0; q < nrow; q++ ){
              FTvLocal(q) += blas::Dot( ntot, df.VecData(q), 1,
                  res.Data(), 1 );

              for( Int p = q; p < nrow; p++ ){
                FTFLocal(p, q) += blas::Dot( ntot, df.VecData(p), 1, 
                    df.VecData(q), 1 );
                if( p > q )
                  FTFLocal(q,p) = FTFLocal(p,q);
              } // for (p)
            } // for (q)

          } // own this element
        } // for (i)

    // Reduce the data
    mpi::Allreduce( FTFLocal.Data(), FTF.Data(), nrow * nrow, 
        MPI_SUM, domain_.colComm );
    mpi::Allreduce( FTvLocal.Data(), FTv.Data(), nrow, 
        MPI_SUM, domain_.colComm );

    // All processors solve the least square problem

    // FIXME Magic number for pseudo-inverse
    Real rcond = 1e-12;
    Int rank;

    DblNumVec  S( nrow );

    // FTv = pinv( FTF ) * res
    lapack::SVDLeastSquare( nrow, nrow, 1, 
        FTF.Data(), nrow, FTv.Data(), nrow,
        S.Data(), rcond, &rank );

    statusOFS << "Rank of dfmat = " << rank <<
      ", rcond = " << rcond << std::endl;

    // Update vOpt, resOpt. 
    // FTv = Y^{\dagger} r as in the usual notation.
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            // vOpt   -= dv * FTv
            blas::Gemv('N', ntot, nrow, -1.0, dvMat.LocalMap()[key].Data(),
                ntot, FTv.Data(), 1, 1.0, 
                distvOpt.LocalMap()[key].Data(), 1 );

            // resOpt -= df * FTv
            blas::Gemv('N', ntot, nrow, -1.0, dfMat.LocalMap()[key].Data(),
                ntot, FTv.Data(), 1, 1.0, 
                distResOpt.LocalMap()[key].Data(), 1 );
          } // own this element
        } // for (i)
  } // (iter > 1)

  if( mixType == "kerker+anderson" ){
    KerkerPrecond( distPrecResOpt, distResOpt );
  }
  else if( mixType == "anderson" ){
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            distPrecResOpt.LocalMap()[key] = 
              distResOpt.LocalMap()[key];
          } // own this element
        } // for (i)
  }
  else{
    ErrorHandling("Invalid mixing type.");
  }

  // Update dfMat, dvMat, vMix 
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          // dfMat(:, inext-1) = res(:)
          // dvMat(:, inext-1) = vOld(:)
          blas::Copy( ntot, distRes.LocalMap()[key].Data(), 1, 
              dfMat.LocalMap()[key].VecData(inext-1), 1 );
          blas::Copy( ntot, distvOld.LocalMap()[key].Data(),  1, 
              dvMat.LocalMap()[key].VecData(inext-1), 1 );

          // vMix(:) = vOpt(:) - mixStepLength * precRes(:)
          distvMix.LocalMap()[key] = distvOpt.LocalMap()[key];
          blas::Axpy( ntot, -mixStepLength, 
              distPrecResOpt.LocalMap()[key].Data(), 1, 
              distvMix.LocalMap()[key].Data(), 1 );
        } // own this element
  } // for (i)

  return ;
}         // -----  end of method SCFDG::AndersonMix  ----- 

void
SCFDG::AndersonMix2    ( 
    Int             iter, 
    Real            mixStepLength,
    DistDblNumVec&  distvMix,
    DistDblNumVec&  distvOld,
    DistDblNumVec&  distvNew,
    DistDblNumMat&  dfMat,
    DistDblNumMat&  dvMat )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  distvMix.SetComm(domain_.colComm);
  distvOld.SetComm(domain_.colComm);
  distvNew.SetComm(domain_.colComm);
  dfMat.SetComm(domain_.colComm);
  dvMat.SetComm(domain_.colComm);

  // Residual 
  DistDblNumVec distRes;
  // Optimal input potential in Anderon mixing.
  DistDblNumVec distvRes; 
  DistDblNumVec distTemp; 
  // Optimal residual in Anderson mixing
  //
  distRes.SetComm(domain_.colComm);
  distvRes.SetComm(domain_.colComm);
  distTemp.SetComm(domain_.colComm);

  // *********************************************************************
  // Initialize
  // *********************************************************************
  Int ntot  = hamDGPtr_->NumUniformGridElemFine().prod();
//  Int numBasis = esdfParam.numALBElem(0, 0, 0);
  Int pos = ((iter - 1) % mixMaxDim_ - 1 + mixMaxDim_ ) % mixMaxDim_;
  Int next = (pos + 1) % mixMaxDim_;

  distRes.Prtn()          = elemPrtn_;
  distvRes.Prtn()         = elemPrtn_;
  distTemp.Prtn()         = elemPrtn_;

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          DblNumVec  emptyVec( ntot );
          SetValue( emptyVec, 0.0 );
          distRes.LocalMap()[key]        = emptyVec;
          distvRes.LocalMap()[key]       = emptyVec;
          distTemp.LocalMap()[key]       = emptyVec;
        } // if ( own this element )
  } // for (i)

  // *********************************************************************
  // Anderson mixing
  // *********************************************************************

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          // res(:) = vOld(:) - vNew(:) is the residual
          DblNumVec& vOld = distvOld.LocalMap()[key];
          DblNumVec& vNew = distvNew.LocalMap()[key];
          DblNumVec&  Res = distRes.LocalMap()[key];
          DblNumVec& vRes = distvRes.LocalMap()[key];
          DblNumMat&  dv  = dvMat.LocalMap()[key];
          DblNumMat&  df  = dfMat.LocalMap()[key];
          
          blas::Axpy( ntot, -1.0, vOld.Data(), 1, vNew.Data(), 1 );
          blas::Copy( ntot, vOld.Data(), 1, Res.Data(), 1);
          blas::Copy( ntot, vNew.Data(), 1, vRes.Data(), 1);
          
          // dfMat(:, ipos-1) = res(:) - dfMat(:, ipos-1);
          // dvMat(:, ipos-1) = vOld(:) - dvMat(:, ipos-1);
          if( iter > 1 ){
            blas::Axpy( ntot, -1.0,  vOld.Data(), 1, dv.VecData(pos), 1 );
            blas::Axpy( ntot, -1.0,  vNew.Data(), 1, df.VecData(pos), 1 );
          }
        } // own this element
  } // for (i)

  // For iter == 1, Anderson mixing is the same as simple mixing. 
  if( iter > 1)
  {
    Int dim=std::min(iter-1,mixMaxDim_);
    // Normal matrix FTF = F^T * F
    DblNumMat FTFLocal( dim, dim ), FTF( dim, dim );
    SetValue( FTFLocal, 0.0 );
    SetValue( FTF, 0.0 );

    // Right hand side FTv = F^T * vout
    DblNumVec FTvLocal( dim ), FTv( dim );
    SetValue( FTvLocal, 0.0 );
    SetValue( FTv, 0.0 );

    // Local construction of FTF and FTv
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumVec& vNew   = distvNew.LocalMap()[key];
            DblNumMat& df     = dfMat.LocalMap()[key];
            for( Int q = 0; q < dim; q++ ){
              FTvLocal(q) += blas::Dot( ntot, df.VecData(q), 1,
                  vNew.Data(), 1 );

              for( Int p = q; p < dim; p++ ){
                FTFLocal(p, q) += blas::Dot( ntot, df.VecData(p), 1, 
                    df.VecData(q), 1 );
                if( p > q )
                  FTFLocal(q,p) = FTFLocal(p,q);
              } // for (p)
            } // for (q)

          } // own this element
    } // for (i)

    // Reduce the data
    mpi::Allreduce( FTFLocal.Data(), FTF.Data(), dim * dim, 
        MPI_SUM, domain_.colComm );
    mpi::Allreduce( FTvLocal.Data(), FTv.Data(), dim, 
        MPI_SUM, domain_.colComm );

    // All processors solve the least square problem

    lapack::Potrf('L', dim, FTF.Data(), dim );

    lapack::Potrs('L', dim, FTF.Data(), dim, I_ONE, FTv.Data(), dim);

    // Update vOpt, resOpt. 
    // FTv = Y^{\dagger} r as in the usual notation.
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumVec& vOld   = distvOld.LocalMap()[key];
            DblNumVec& vNew   = distvNew.LocalMap()[key];
            DblNumMat& df     = dfMat.LocalMap()[key];
            DblNumMat& dv     = dvMat.LocalMap()[key];
            for(Int p = 0; p < dim; p++){
              blas::Axpy(ntot, -FTv(p),  dv.VecData(p),
                1, vOld.Data(), 1); // Xopt = Xin - \sum\gamma\Delta X
              blas::Axpy(ntot, -FTv(p),  df.VecData(p),
                1, vNew.Data(), 1); // Fopt = F - \sum\gamma\Delta F
            }
          } // own this element
    } // for (i)
  } // (iter > 1)

  // Update dfMat, dvMat, vMix 
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){

          DblNumVec& vOld   = distvOld.LocalMap()[key];
          DblNumVec& vNew   = distvNew.LocalMap()[key];
          DblNumVec& vMix   = distvMix.LocalMap()[key];
          DblNumVec& Res    = distRes.LocalMap()[key];
          DblNumVec& vRes   = distvRes.LocalMap()[key];
          DblNumMat& df     = dfMat.LocalMap()[key];
          DblNumMat& dv     = dvMat.LocalMap()[key];
          DblNumVec& Temp = distTemp.LocalMap()[key];

          blas::Copy(ntot, vOld.Data(), 1, Temp.Data(), 1);
          blas::Axpy(ntot, mixStepLength, vNew.Data(), 1, Temp.Data(), 1);
          blas::Copy(ntot, Temp.Data(), 1, vMix.Data(), 1);

          blas::Copy(ntot, Res.Data(), 1, dv.VecData(next), 1);
          blas::Copy(ntot, vRes.Data(), 1, df.VecData(next), 1);

        } // own this element
  } // for (i)

  return ;
}         // -----  end of method SCFDG::AndersonMix2  ----- 

void 
SCFDG::AndersonMix    (
        Int             iter,
        Real            mixStepLength,
        std::string     mixType,
        DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvMix,
        DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvOld,
        DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvNew,
        DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>&  dfMat,
        DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>&  dvMat )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  distvMix.SetComm(domain_.colComm);
  distvOld.SetComm(domain_.colComm);
  distvNew.SetComm(domain_.colComm);
  dfMat.SetComm(domain_.colComm);
  dvMat.SetComm(domain_.colComm);

  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn> distRes;
  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn> distvOpt;
  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn> distResOpt;

  distRes.LocalMap().clear();
  distvOpt.LocalMap().clear();
  distResOpt.LocalMap().clear();

  distRes.Prtn()          = distvMix.Prtn();
  distvOpt.Prtn()         = distvMix.Prtn();
  distResOpt.Prtn()       = distvMix.Prtn();

  distRes.SetComm(domain_.colComm);
  distvOpt.SetComm(domain_.colComm);
  distResOpt.SetComm(domain_.colComm);

  // *********************************************************************
  // Initialize
  // *********************************************************************
  Int ntot = esdfParam.numALBElem(0, 0, 0) * esdfParam.numALBElem(0, 0, 0);
  Int numBasis = esdfParam.numALBElem(0, 0, 0);
  Int iterused = std::min( iter-1, mixMaxDim_ );
  Int ipos = iter - 1 - ((iter-2)/ mixMaxDim_ ) * mixMaxDim_;
  Int inext = iter - ((iter-1)/ mixMaxDim_) * mixMaxDim_;

//  statusOFS << " numBasis " << numBasis <<std::endl;
//  statusOFS << "  ntot " << ntot << std::endl;
//  statusOFS << " iterused " << iterused << std::endl;
//  statusOFS << " ipos " << ipos << std::endl;
//  statusOFS << " inext " << inext << std::endl; 

  DblNumMat  emptyMat( numBasis,  numBasis );
  SetValue( emptyMat, 0.0 );

  for(typename std::map<ElemMatKey, DblNumMat >::iterator
    My_iterator = distvMix.LocalMap().begin();
    My_iterator != distvMix.LocalMap().end();
    ++ My_iterator )
  {
    ElemMatKey matkey = (*My_iterator).first;
    distRes.LocalMap()[matkey]     = emptyMat;
    distvOpt.LocalMap()[matkey]    = emptyMat;
    distResOpt.LocalMap()[matkey]  = emptyMat;
   }

  // *********************************************************************
  // Anderson mixing
  // *********************************************************************
  for(typename std::map<ElemMatKey, DblNumMat >::iterator
     My_iterator = distvMix.LocalMap().begin();
     My_iterator != distvMix.LocalMap().end();
     ++ My_iterator )
  {
    ElemMatKey matkey = (*My_iterator).first;
    DblNumMat& vOld = distvOld.LocalMap()[matkey];
    DblNumMat& Res = distRes.LocalMap()[matkey];
    DblNumMat& vNew = distvNew.LocalMap()[matkey];
    blas::Copy(ntot, vOld.Data(), 1, Res.Data(), 1 ); 
    blas::Axpy( ntot, -1.0, vNew.Data(), 1, Res.Data(), 1);

    DblNumMat& vOpt = distvOpt.LocalMap()[matkey];
    DblNumMat& ResOpt =  distResOpt.LocalMap()[matkey];

    blas::Copy(ntot, vOld.Data(), 1, vOpt.Data(), 1 );
    blas::Copy(ntot, Res.Data(), 1, ResOpt.Data(), 1 );

    if( iter > 1 ){
      DblNumTns& dfTns = dfMat.LocalMap()[matkey];
      DblNumTns& dvTns = dvMat.LocalMap()[matkey];
      blas::Scal( ntot, -1.0, dfTns.MatData(ipos-1), 1 );
      blas::Axpy( ntot, 1.0, Res.Data(), 1, dfTns.MatData(ipos-1), 1 );
      blas::Scal( ntot, -1.0, dvTns.MatData(ipos-1), 1 );
      blas::Axpy( ntot, 1.0, vOld.Data(), 1, dvTns.MatData(ipos-1), 1 );
    }
   }

  if( iter > 1){

    Int nrow = iterused;

    DblNumMat FTFLocal( nrow, nrow ), FTF( nrow, nrow );
    SetValue( FTFLocal, 0.0 );
    SetValue( FTF, 0.0 );

    DblNumVec FTvLocal( nrow ), FTv( nrow );
    SetValue( FTvLocal, 0.0 );
    SetValue( FTv, 0.0 );

    for(typename std::map<ElemMatKey, DblNumMat >::iterator
      My_iterator = distvMix.LocalMap().begin();
      My_iterator != distvMix.LocalMap().end();
      ++ My_iterator )
    {
      ElemMatKey matkey = (* My_iterator).first;
      DblNumTns& df     = dfMat.LocalMap()[matkey];
      DblNumMat& res    = distRes.LocalMap()[matkey];
      for( Int q = 0; q < nrow; q++ ){
        FTvLocal(q) += blas::Dot( ntot, df.MatData(q), 1,
            res.Data(), 1 );

        for( Int p = q; p < nrow; p++ ){
          FTFLocal(p, q) += blas::Dot( ntot, df.MatData(p), 1,
              df.MatData(q), 1 );
          if( p > q )
            FTFLocal(q,p) = FTFLocal(p,q);
        } // for (p)
      } // for (q)
    }

    mpi::Allreduce( FTFLocal.Data(), FTF.Data(), nrow * nrow,
        MPI_SUM, domain_.colComm );
    mpi::Allreduce( FTvLocal.Data(), FTv.Data(), nrow,
        MPI_SUM, domain_.colComm );

    Real rcond = 1e-12;
    Int rank;

    DblNumVec  S( nrow );

    lapack::SVDLeastSquare( nrow, nrow, 1,
        FTF.Data(), nrow, FTv.Data(), nrow,
        S.Data(), rcond, &rank );

    statusOFS << "Rank of dfmat = " << rank <<
      ", rcond = " << rcond << std::endl;
       
    for(typename std::map<ElemMatKey, DblNumMat >::iterator
      My_iterator = distvMix.LocalMap().begin();
      My_iterator != distvMix.LocalMap().end();
      ++ My_iterator )
    {
      ElemMatKey matkey = (*My_iterator).first;
      blas::Gemv('N', ntot, nrow, -1.0, dvMat.LocalMap()[matkey].Data(),
          ntot, FTv.Data(), 1, 1.0,
          distvOpt.LocalMap()[matkey].Data(), 1 );

      blas::Gemv('N', ntot, nrow, -1.0, dfMat.LocalMap()[matkey].Data(),
          ntot, FTv.Data(), 1, 1.0,
          distResOpt.LocalMap()[matkey].Data(), 1 );
    }
    
  } // ( iter > 1 )

  // Update dfMat, dvMat, vMix 
  for(typename std::map<ElemMatKey, DblNumMat >::iterator
    My_iterator = distvMix.LocalMap().begin();
    My_iterator != distvMix.LocalMap().end();
    ++ My_iterator )
  {
    ElemMatKey matkey = (*My_iterator).first;
    
    blas::Copy( ntot, distRes.LocalMap()[matkey].Data(), 1,
        dfMat.LocalMap()[matkey].MatData(inext-1), 1 );
    blas::Copy( ntot, distvOld.LocalMap()[matkey].Data(),  1,
        dvMat.LocalMap()[matkey].MatData(inext-1), 1 );

    Real fac = -1.0 * mixStepLength;      
    DblNumMat& vOpt = distvOpt.LocalMap()[matkey];
    DblNumMat& ResOpt = distResOpt.LocalMap()[matkey];

    distvMix.LocalMap()[matkey] = vOpt;
    blas::Axpy( ntot, fac, ResOpt.Data(), 1,
         distvMix.LocalMap()[matkey].Data(), 1 );
  } 
  
  return ;
}

void
SCFDG::AndersonMix2    ( 
       Int             iter, 
       Real            mixStepLength,
       DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvMix,
       DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvOld,
       DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvNew,
       DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>&  dfMat,
       DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>&  dvMat )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  distvMix.SetComm(domain_.colComm);
  distvOld.SetComm(domain_.colComm);
  distvNew.SetComm(domain_.colComm);
  dfMat.SetComm(domain_.colComm);
  dvMat.SetComm(domain_.colComm);

  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn> distRes;
  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn> distvRes;
  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn> distTemp;

  distRes.LocalMap().clear();
  distvRes.LocalMap().clear();
  distTemp.LocalMap().clear();

  distRes.Prtn()          = distvMix.Prtn();
  distvRes.Prtn()         = distvMix.Prtn();
  distTemp.Prtn()         = distvMix.Prtn();

  distRes.SetComm(domain_.colComm);
  distvRes.SetComm(domain_.colComm);
  distTemp.SetComm(domain_.colComm);

  Int ntot = esdfParam.numALBElem(0, 0, 0) * esdfParam.numALBElem(0, 0, 0);
  Int numBasis = esdfParam.numALBElem(0, 0, 0);
  Int pos = ((iter - 1) % mixMaxDim_ - 1 + mixMaxDim_ ) % mixMaxDim_;
  Int next = (pos + 1) % mixMaxDim_;

//  statusOFS << " numBasis " << numBasis <<std::endl;
//  statusOFS << " ntot " << ntot << std::endl;
//  statusOFS << " pos " << pos << std::endl;
//  statusOFS << " next " << next << std::endl;
  
  DblNumMat  emptyMat( numBasis,  numBasis );
  SetValue( emptyMat, 0.0 );

  for(typename std::map<ElemMatKey, DblNumMat >::iterator
    My_iterator = distvMix.LocalMap().begin();
    My_iterator != distvMix.LocalMap().end();
    ++ My_iterator )
  {
    ElemMatKey matkey = (*My_iterator).first;
    distRes.LocalMap()[matkey]    = emptyMat;
    distvRes.LocalMap()[matkey]   = emptyMat;
    distTemp.LocalMap()[matkey]   = emptyMat;
   }

  // *********************************************************************
  // Anderson mixing
  // *********************************************************************
  for(typename std::map<ElemMatKey, DblNumMat >::iterator
     My_iterator = distvMix.LocalMap().begin();
     My_iterator != distvMix.LocalMap().end();
     ++ My_iterator )
  {
    ElemMatKey matkey = (*My_iterator).first;
    DblNumMat& vOld = distvOld.LocalMap()[matkey];
    DblNumMat& vNew = distvNew.LocalMap()[matkey];
    DblNumMat&  Res = distRes.LocalMap()[matkey];
    DblNumMat& vRes = distvRes.LocalMap()[matkey];
    DblNumTns& df = dfMat.LocalMap()[matkey];
    DblNumTns& dv = dvMat.LocalMap()[matkey];

    // F = Xout- Xin
    blas::Axpy( ntot, -1.0, vOld.Data(), 1, vNew.Data(), 1);
    // Store the current Xin and F
    blas::Copy(ntot, vOld.Data(), 1, Res.Data(),1);
    blas::Copy(ntot, vNew.Data(), 1, vRes.Data(),1);

    if( iter > 1 ){
      // -\Delta X = Xin(i-1) - Xin(i)
      blas::Axpy(ntot, -1.0, vOld.Data(), 1, dv.MatData(pos), 1);
      // -\Delta F = F(i-1) - F(i)
      blas::Axpy(ntot, -1.0, vNew.Data(), 1, df.MatData(pos), 1);
     }
  }

  if( iter > 1 ){
    Int dim = std::min(iter - 1, mixMaxDim_ );

    DblNumMat FTFLocal( dim, dim ), FTF ( dim, dim );
    SetValue( FTFLocal, 0.0 );
    SetValue( FTF, 0.0 );

    DblNumVec FTvLocal( dim ), FTv( dim );
    SetValue( FTvLocal, 0.0 );
    SetValue( FTv, 0.0 );

    for(typename std::map<ElemMatKey, DblNumMat >::iterator
      My_iterator = distvMix.LocalMap().begin();
      My_iterator != distvMix.LocalMap().end();
      ++ My_iterator )
    {
      ElemMatKey matkey = (* My_iterator).first;
      DblNumMat& vNew   = distvNew.LocalMap()[matkey];
      DblNumTns& df     = dfMat.LocalMap()[matkey];

      for(Int q = 0; q < dim; q++){

        FTvLocal(q) += blas::Dot(ntot, df.MatData(q), 1, vNew.Data(), 1);  // -<\Delta F(i)|F>

        for(Int p = q; p < dim; p++){
//          statusOFS << " i, j " << i << " , " <<j <<std::endl; 
          // <\Delta F(i)|\Delta F(j)>
          FTFLocal(p,q) += blas::Dot(ntot, df.MatData(p), 1, df.MatData(q),1); 
//          statusOFS << " FTFLocal(j,i) "<< FTFLocal(j,i)  << std::endl;
          if( p > q )  FTFLocal(q, p) = FTFLocal(p, q);
        }
      }
    }

    mpi::Allreduce( FTFLocal.Data(), FTF.Data(), dim * dim,
        MPI_SUM, domain_.colComm );
    mpi::Allreduce( FTvLocal.Data(), FTv.Data(), dim,
        MPI_SUM, domain_.colComm );

    lapack::Potrf('L', dim, FTF.Data(), dim );

    lapack::Potrs('L', dim, FTF.Data(), dim, I_ONE, FTv.Data(), dim);

    for(typename std::map<ElemMatKey, DblNumMat >::iterator
      My_iterator = distvMix.LocalMap().begin();
      My_iterator != distvMix.LocalMap().end();
      ++ My_iterator )
    {
      ElemMatKey matkey = (*My_iterator).first;
      DblNumTns& df     = dfMat.LocalMap()[matkey];
      DblNumTns& dv     = dvMat.LocalMap()[matkey];
      DblNumMat& vOld   = distvOld.LocalMap()[matkey];
      DblNumMat& vNew   = distvNew.LocalMap()[matkey];

      for(Int i = 0; i < dim; i++){ 
        blas::Axpy(ntot, -FTv(i),  dv.MatData(i),
             1, vOld.Data(), 1); // Xopt = Xin - \sum\gamma\Delta X
        blas::Axpy(ntot, -FTv(i),  df.MatData(i),
             1, vNew.Data(), 1); // Fopt = F - \sum\gamma\Delta F
      }
    }

  } // ( iter > 1 )

  for(typename std::map<ElemMatKey, DblNumMat >::iterator
    My_iterator = distvMix.LocalMap().begin();
    My_iterator != distvMix.LocalMap().end();
    ++ My_iterator )
  {
    ElemMatKey matkey = (*My_iterator).first;
    DblNumMat& vOld = distvOld.LocalMap()[matkey];
    DblNumMat& vNew = distvNew.LocalMap()[matkey];
    DblNumMat&  Res = distRes.LocalMap()[matkey];
    DblNumMat& vRes = distvRes.LocalMap()[matkey];
    DblNumMat& vMix = distvMix.LocalMap()[matkey];
    DblNumMat& Temp = distTemp.LocalMap()[matkey];

    DblNumTns& df   = dfMat.LocalMap()[matkey];
    DblNumTns& dv   = dvMat.LocalMap()[matkey];

    blas::Copy(ntot, vOld.Data(), 1, Temp.Data(), 1);
    blas::Axpy(ntot, mixStepLength, vNew.Data(), 1, Temp.Data(), 1);
    blas::Copy(ntot, Temp.Data(), 1, vMix.Data(), 1);

    blas::Copy(ntot, Res.Data(), 1, dv.MatData(next), 1);
    blas::Copy(ntot, vRes.Data(), 1, df.MatData(next), 1);
  }

  return ;
}   

void
SCFDG::KerkerPrecond ( 
    DistDblNumVec&  distPrecResidual,
    const DistDblNumVec&  distResidual )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  DistFourier& fft = *distfftPtr_;
  //DistFourier.SetComm(domain_.colComm);

  Int ntot      = fft.numGridTotal;
  Int ntotLocal = fft.numGridLocal;

  Index3 numUniformGridElem = hamDGPtr_->NumUniformGridElem();

  // Convert distResidual to tempVecLocal in distributed row vector format
  DblNumVec  tempVecLocal;

  DistNumVecToDistRowVec(
      distResidual,
      tempVecLocal,
      domain_.numGridFine,
      numElem_,
      fft.localNzStart,
      fft.localNz,
      fft.isInGrid,
      domain_.colComm );

  // NOTE Fixed KerkerB parameter
  //
  // From the point of view of the elliptic preconditioner
  //
  // (-\Delta + 4 * pi * b) r_p = -Delta r
  //
  // The Kerker preconditioner in the Fourier space is
  //
  // k^2 / (k^2 + 4 * pi * b)
  //
  // or using gkk = k^2 /2 
  //
  // gkk / ( gkk + 2 * pi * b )
  //
  // Here we choose KerkerB to be a fixed number.

  // FIXME hard coded
  Real KerkerB = 0.08; 
  Real Amin = 0.4;

  if( fft.isInGrid ){

    for( Int i = 0; i < ntotLocal; i++ ){
      fft.inputComplexVecLocal(i) = Complex( 
          tempVecLocal(i), 0.0 );
    }
    fftw_execute( fft.forwardPlan );

    for( Int i = 0; i < ntotLocal; i++ ){
      // Do not touch the zero frequency
      // Procedure taken from VASP
      if( fft.gkkLocal(i) != 0 ){
        fft.outputComplexVecLocal(i) *= fft.gkkLocal(i) / 
          ( fft.gkkLocal(i) + 2.0 * PI * KerkerB );
        //                fft.outputComplexVecLocal(i) *= std::min(fft.gkkLocal(i) / 
        //                        ( fft.gkkLocal(i) + 2.0 * PI * KerkerB ), Amin);
      }
    }
    fftw_execute( fft.backwardPlan );

    for( Int i = 0; i < ntotLocal; i++ ){
      tempVecLocal(i) = fft.inputComplexVecLocal(i).real() / ntot;
    }
  } // if (fft.isInGrid)

  // Convert tempVecLocal to distPrecResidual in the DistNumVec format 

  DistRowVecToDistNumVec(
      tempVecLocal,
      distPrecResidual,
      domain_.numGridFine,
      numElem_,
      fft.localNzStart,
      fft.localNz,
      fft.isInGrid,
      domain_.colComm );



  return ;
}         // -----  end of method SCFDG::KerkerPrecond  ----- 


void
SCFDG::BroydenMix    (
       Int             iter,
       Real            mixStepLength,
       DistDblNumVec&  distvMix,
       DistDblNumVec&  distvOld,
       DistDblNumVec&  distvNew,
       DistDblNumMat&  dfMat,
       DistDblNumMat&  dvMat,
       DistDblNumMat&  cdfMat )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  distvMix.SetComm(domain_.colComm);
  distvOld.SetComm(domain_.colComm);
  distvNew.SetComm(domain_.colComm);
  dfMat.SetComm(domain_.colComm);
  dvMat.SetComm(domain_.colComm);
  cdfMat.SetComm(domain_.colComm);

  DistDblNumVec distGvMix;
//  DistDblNumVec distGvOld;
//  DistDblNumVec distGvNew;

  // Create space to store intermediate variables
  distGvMix.SetComm(domain_.colComm);
//  distGvOld.SetComm(domain_.colComm);
//  distGvNew.SetComm(domain_.colComm);

  distGvMix.Prtn()       = elemPrtn_;
//  distGvOld.Prtn()       = elemPrtn_;
//  distGvNew.Prtn()       = elemPrtn_;

  Int ntot  = hamDGPtr_->NumUniformGridElemFine().prod();

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          DblNumVec  emptyVec( ntot );
          SetValue( emptyVec, 0.0 );
          distGvMix.LocalMap()[key]     = emptyVec;
//          distGvOld.LocalMap()[key]     = distvOld.LocalMap()[key];
//          distGvNew.LocalMap()[key]     = distvNew.LocalMap()[key];
        } // if ( own this element )
  } // for (i)

  // *********************************************************************
  // Initialize
  // *********************************************************************

  // Number of iterations used, iter should start from 1
  Int iterused = std::min( iter-1, mixMaxDim_ );
  // The current position of dfMat, dvMat
  Int ipos = iter - 1 - ((iter-2)/ mixMaxDim_ ) * mixMaxDim_;

  // *********************************************************************
  // Broyden mixing
  // *********************************************************************
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          DblNumVec& vOld   = distvOld.LocalMap()[key];
          DblNumVec& vNew   = distvNew.LocalMap()[key];
          DblNumMat& cdf    = cdfMat.LocalMap()[key];
          DblNumMat& df     = dfMat.LocalMap()[key];
          DblNumMat& dv     = dvMat.LocalMap()[key];

          blas::Axpy( ntot, -1.0, vOld.Data(), 1, vNew.Data(), 1 );

          if( iter > 1 ){
            blas::Copy( ntot, cdf.VecData(0), 1, df.VecData(ipos-1), 1);
            blas::Axpy( ntot, -1.0, vNew.Data(), 1, df.VecData(ipos-1), 1);
            blas::Copy( ntot, cdf.VecData(1), 1, dv.VecData(ipos-1), 1 );
            blas::Axpy( ntot, -1.0, vOld.Data(), 1, dv.VecData(ipos-1), 1);
          }

          blas::Copy( ntot, vNew.Data(), 1, cdf.VecData(0), 1 );
          blas::Copy( ntot, vOld.Data(), 1, cdf.VecData(1), 1 );
        } // own this element
      } // for (i)

  if( iterused > 0){

    Int nrow = iterused;

    DblNumMat betamixLocal( nrow, nrow ), betamix( nrow, nrow );
    SetValue( betamixLocal, D_ZERO );
    SetValue( betamix, D_ZERO );

    // Local construction of betamix
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumMat& df = dfMat.LocalMap()[key];
            for (Int p=0; p<nrow; p++) {
              for (Int q=p; q<nrow; q++) {
                betamixLocal(p,q) = blas::Dot( ntot, df.VecData(q), 1,
                    df.VecData(p), 1 );
                betamixLocal(q,p) = betamixLocal(p,q);
              }
            }
          } // own this element
        } // for (i)

    // Reduce the data
    mpi::Allreduce( betamixLocal.Data(), betamix.Data(), nrow * nrow,
        MPI_SUM, domain_.colComm );

    // Inverse betamix using the Bunch-Kaufman diagonal pivoting method
    IntNumVec iwork;
    iwork.Resize( nrow ); SetValue( iwork, I_ZERO );

    lapack::Sytrf( 'U', nrow, betamix.Data(), nrow, iwork.Data() );
    lapack::Sytri( 'U', nrow, betamix.Data(), nrow, iwork.Data() );
    for (Int p=0; p<nrow; p++) {
      for (Int q=p+1; q<nrow; q++) {
        betamix(q,p) = betamix(p,q);
      }
    }

    DblNumVec workLocal(nrow), work(nrow);
    Real gamma0 = D_ZERO;
    SetValue( workLocal, D_ZERO ); SetValue( work, D_ZERO );

    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumMat& df     = dfMat.LocalMap()[key];
            DblNumVec& vNew = distvNew.LocalMap()[key];
            for (Int p=0; p<nrow; p++) {
              workLocal(p) = blas::Dot( ntot, df.VecData(p), 1,
                vNew.Data(), 1 );
            }
          } // own this element
        } // for (i)

    mpi::Allreduce( workLocal.Data(), work.Data(), nrow,
        MPI_SUM, domain_.colComm );

    for (Int p=0; p<nrow; p++){
      gamma0 = blas::Dot( nrow, betamix.VecData(p), 1, work.Data(), 1 );

      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              blas::Axpy( ntot, -gamma0, dvMat.LocalMap()[key].VecData(p), 1,
                  distvOld.LocalMap()[key].Data(), 1);
              blas::Axpy( ntot, -gamma0, dfMat.LocalMap()[key].VecData(p), 1,
                  distvNew.LocalMap()[key].Data(), 1);
            } // own this element
          } // for (i)
    }
  } // End of if ( iterused > 0 )

  // Update vMix
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          // vMix(:) = GvMix(:) = GvOld(:) + mixStepLength * GvNew(:)
          blas::Copy( ntot, distvOld.LocalMap()[key].Data(), 1,
              distGvMix.LocalMap()[key].Data(), 1 );

          blas::Axpy( ntot, mixStepLength, distvNew.LocalMap()[key].Data(), 1,
              distGvMix.LocalMap()[key].Data(), 1 );

          blas::Copy( ntot, distGvMix.LocalMap()[key].Data(), 1,
              distvMix.LocalMap()[key].Data(), 1 );

        } // own this element
      } // for (i)

  return ;
}         // -----  end of method SCFDG::BroydenMix( distributed vector version )  -----


void
SCFDG::BroydenMix    (
       Int             iter,
       Real            mixStepLength,
       DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvMix,
       DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvOld,
       DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distvNew,
       DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>&  dfTns,
       DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>&  dvTns,
       DistVec<ElemMatKey, NumTns<Real>, ElemMatPrtn>&  cdfTns )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  distvMix.SetComm(domain_.colComm);
  distvOld.SetComm(domain_.colComm);
  distvNew.SetComm(domain_.colComm);
  dfTns.SetComm(domain_.colComm);
  dvTns.SetComm(domain_.colComm);
  cdfTns.SetComm(domain_.colComm);

  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn> distGvMix;
  // Create space to store intermediate variables
  distGvMix.SetComm(domain_.colComm);
  distGvMix.Prtn()  = distvMix.Prtn();

  Int ntot = esdfParam.numALBElem(0, 0, 0) * esdfParam.numALBElem(0, 0, 0);
  Int numBasis = esdfParam.numALBElem(0, 0, 0);

  DblNumMat  emptyMat( numBasis,  numBasis );
  SetValue( emptyMat, 0.0 );

  for(typename std::map<ElemMatKey, DblNumMat >::iterator
    My_iterator = distvMix.LocalMap().begin();
    My_iterator != distvMix.LocalMap().end();
    ++ My_iterator )
  {
    ElemMatKey matkey = (*My_iterator).first;
    distGvMix.LocalMap()[matkey]    = emptyMat;
   }

  // *********************************************************************
  // Initialize
  // *********************************************************************

  // Number of iterations used, iter should start from 1
  Int iterused = std::min( iter-1, mixMaxDim_ );
  // The current position of dfMat, dvMat
  Int ipos = iter - 1 - ((iter-2)/ mixMaxDim_ ) * mixMaxDim_;

  // *********************************************************************
  // Broyden mixing
  // *********************************************************************

  for(typename std::map<ElemMatKey, DblNumMat >::iterator
     My_iterator = distvMix.LocalMap().begin();
     My_iterator != distvMix.LocalMap().end();
     ++ My_iterator )
  {
    ElemMatKey matkey = (*My_iterator).first;
    DblNumMat& vOld  = distvOld.LocalMap()[matkey];
    DblNumMat& vNew  = distvNew.LocalMap()[matkey];
    DblNumTns& cdf    = cdfTns.LocalMap()[matkey];
    DblNumTns& df     = dfTns.LocalMap()[matkey];
    DblNumTns& dv     = dvTns.LocalMap()[matkey];

    blas::Axpy( ntot, -1.0, vOld.Data(), 1, vNew.Data(), 1 );

    if( iter > 1 ){
      blas::Copy( ntot, cdf.MatData(0), 1, df.MatData(ipos-1), 1);
      blas::Axpy( ntot, -1.0, vNew.Data(), 1, df.MatData(ipos-1), 1);
      blas::Copy( ntot, cdf.MatData(1), 1, dv.MatData(ipos-1), 1 );
      blas::Axpy( ntot, -1.0, vOld.Data(), 1, dv.MatData(ipos-1), 1);
    }

    blas::Copy( ntot, vNew.Data(), 1, cdf.MatData(0), 1 );
    blas::Copy( ntot, vOld.Data(), 1, cdf.MatData(1), 1 );
  } // for (i)

  if( iterused > 0){

    Int nrow = iterused;
    DblNumMat betamixLocal( nrow, nrow ), betamix( nrow, nrow );
    SetValue( betamixLocal, D_ZERO );
    SetValue( betamix, D_ZERO );

    for(typename std::map<ElemMatKey, DblNumMat >::iterator
       My_iterator = distvMix.LocalMap().begin();
       My_iterator != distvMix.LocalMap().end();
       ++ My_iterator )
    {
      ElemMatKey matkey = (*My_iterator).first;
      DblNumTns& df     = dfTns.LocalMap()[matkey];

      for (Int p = 0; p < nrow; p++) {
        for (Int q = p; q < nrow; q++) {
          betamixLocal(p, q) = blas::Dot( ntot, df.MatData(q), 1, df.MatData(p), 1 );
          betamixLocal(q, p) = betamixLocal(p, q);
        }
      }
    }
    // Reduce the data
    mpi::Allreduce( betamixLocal.Data(), betamix.Data(), nrow * nrow,
        MPI_SUM, domain_.colComm );

    // Inverse betamix using the Bunch-Kaufman diagonal pivoting method
    IntNumVec iwork;
    iwork.Resize( nrow ); SetValue( iwork, I_ZERO );

    lapack::Sytrf( 'U', nrow, betamix.Data(), nrow, iwork.Data() );
    lapack::Sytri( 'U', nrow, betamix.Data(), nrow, iwork.Data() );
    for (Int p=0; p<nrow; p++) {
      for (Int q=p+1; q<nrow; q++) {
        betamix(q,p) = betamix(p,q);
      }
    }

    DblNumVec workLocal(nrow), work(nrow);
    Real gamma0 = D_ZERO;
    SetValue( workLocal, D_ZERO ); SetValue( work, D_ZERO );

    for(typename std::map<ElemMatKey, DblNumMat >::iterator
       My_iterator = distvMix.LocalMap().begin();
       My_iterator != distvMix.LocalMap().end();
       ++ My_iterator )
    {
      ElemMatKey matkey = (*My_iterator).first;
      DblNumMat& vNew  = distvNew.LocalMap()[matkey];
      DblNumTns& cdf    = cdfTns.LocalMap()[matkey];
      DblNumTns& df     = dfTns.LocalMap()[matkey];

      for (Int p=0; p<nrow; p++) {
        workLocal(p) = blas::Dot( ntot, df.MatData(p), 1,
          vNew.Data(), 1 );
      }
    }

    mpi::Allreduce( workLocal.Data(), work.Data(), nrow,
        MPI_SUM, domain_.colComm );

    for (Int p=0; p<nrow; p++){
      gamma0 = blas::Dot( nrow, betamix.VecData(p), 1, work.Data(), 1 );

      for(typename std::map<ElemMatKey, DblNumMat >::iterator
         My_iterator = distvMix.LocalMap().begin();
         My_iterator != distvMix.LocalMap().end();
         ++ My_iterator )
      {
        ElemMatKey matkey = (*My_iterator).first;
        DblNumMat& vNew  = distvNew.LocalMap()[matkey];
        DblNumMat& vOld  = distvOld.LocalMap()[matkey];
        DblNumTns& df     = dfTns.LocalMap()[matkey];
        DblNumTns& dv     = dvTns.LocalMap()[matkey];

        blas::Axpy( ntot, -gamma0, dv.MatData(p), 1, vOld.Data(), 1);
        blas::Axpy( ntot, -gamma0, df.MatData(p), 1, vNew.Data(), 1);
      } // for (i)
    }
  } // End of if ( iterused > 0 )

  // Update vMix
  for(typename std::map<ElemMatKey, DblNumMat >::iterator
     My_iterator = distvMix.LocalMap().begin();
     My_iterator != distvMix.LocalMap().end();
     ++ My_iterator )
  {
    ElemMatKey matkey = (*My_iterator).first;
    DblNumMat& vNew   = distvNew.LocalMap()[matkey];
    DblNumMat& vOld   = distvOld.LocalMap()[matkey];
    DblNumMat& vMix   = distvMix.LocalMap()[matkey];
    DblNumMat& GvMix  = distGvMix.LocalMap()[matkey];

    blas::Copy( ntot, vOld.Data(), 1, GvMix.Data(), 1 );
    blas::Axpy( ntot, mixStepLength, vNew.Data(), 1, GvMix.Data(), 1 );
    blas::Copy( ntot, GvMix.Data(), 1, vMix.Data(), 1 );

  } // for (i)

  return ;
}         // -----  end of method SCFDG::BroydenMix( distributed vector version )  -----

void
SCFDG::PrintState    ( ) 
{
  HamiltonianDG&  hamDG = *hamDGPtr_;

  Real HOMO, LUMO, EG;
  HOMO = hamDG.EigVal()( hamDG.NumOccupiedState()-1 );
  if( hamDG.NumExtraState() > 0 ){
    LUMO = hamDG.EigVal()( hamDG.NumOccupiedState());
    EG = LUMO -HOMO;
  }

#if 0
  if(SCFDG_comp_subspace_engaged_ == false)
  {  
    statusOFS << std::endl << "Eigenvalues in the global domain." << std::endl;
    for(Int i = 0; i < hamDG.EigVal().m(); i++){
      Print(statusOFS, 
          "band#    = ", i, 
          "eigval   = ", hamDG.EigVal()(i),
          "occrate  = ", hamDG.OccupationRate()(i));
    }
  }
#endif

  statusOFS << std::endl;
  Print(statusOFS, "EfreeHarris       = ",  EfreeHarris_, "[au]");
  //    Print(statusOFS, "EfreeSecondOrder  = ",  EfreeSecondOrder_, "[au]");
  Print(statusOFS, "Etot              = ",  Etot_, "[au]");
  Print(statusOFS, "Efree             = ",  Efree_, "[au]");
  Print(statusOFS, "Ekin              = ",  Ekin_, "[au]");
  Print(statusOFS, "Ehart             = ",  Ehart_, "[au]");
  Print(statusOFS, "EVxc              = ",  EVxc_, "[au]");
  Print(statusOFS, "Exc               = ",  Exc_, "[au]"); 
//  Print(statusOFS, "Exx               = ",  Ehfx_, "[au]");
  Print(statusOFS, "EVdw              = ",  Evdw_, "[au]"); 
  Print(statusOFS, "Eself             = ",  Eself_, "[au]");
  Print(statusOFS, "EIonSR            = ",  hamDG.EIonSR(), "[au]");
  Print(statusOFS, "Ecor              = ",  Ecor_, "[au]");
  Print(statusOFS, "Fermi             = ",  fermi_, "[au]");
  Print(statusOFS, "HOMO#            = ",  HOMO*au2ev, "[eV]");
  if( hamDG.NumExtraState() > 0 ){
    Print(statusOFS, "LUMO#            = ",  LUMO*au2ev, "[eV]");
    Print(statusOFS, "Bandgap#         = ",  EG*au2ev, "[eV]");
  }

  return ;
}         // -----  end of method SCFDG::PrintState  ----- 

void
SCFDG::UpdateMDParameters    ( )
{
  scfOuterMaxIter_ = esdfParam.MDscfOuterMaxIter;
  useEnergySCFconvergence_ = 1;

  return ;
}         // -----  end of method SCFDG::UpdateMDParameters  ----- 

void
SCFDG::SetupDMMix( )
{
  statusOFS << "Init hybrid mixing parameters " <<std::endl;

  HamiltonianDG& hamDG = *hamDGPtr_;

  distmixInnerSave_.SetComm(domain_.colComm);
  //distdfOuterMat_.SetComm(domain_.colComm);
  distdfInnerMat_.SetComm(domain_.colComm);
  //distdvOuterMat_.SetComm(domain_.colComm);
  distdvInnerMat_.SetComm(domain_.colComm);
  distcdfInnerMat_.SetComm(domain_.colComm);
  //distmixOuterSave_.Prtn()  = hamDG.HMat().Prtn();
  distmixInnerSave_.Prtn()  = hamDG.HMat().Prtn();
  //distdfOuterMat_.Prtn()  = hamDG.HMat().Prtn();
  distdfInnerMat_.Prtn()  = hamDG.HMat().Prtn();
  //distdvOuterMat_.Prtn()  = hamDG.HMat().Prtn();
  distdvInnerMat_.Prtn()  = hamDG.HMat().Prtn();
  distcdfInnerMat_.Prtn()  = hamDG.HMat().Prtn();

  //distmixOuterSave_.LocalMap().clear();
  distmixInnerSave_.LocalMap().clear();
  //distdfOuterMat_.LocalMap().clear();
  distdfInnerMat_.LocalMap().clear();
        //distdvOuterMat_.LocalMap().clear();
  distdvInnerMat_.LocalMap().clear();
  distcdfInnerMat_.LocalMap().clear();

  Int numBasis = esdfParam.numALBElem(0,0,0);
  DblNumMat emptyMat( numBasis, numBasis );
  SetValue( emptyMat, 0.0 );
  DblNumTns emptyTns( numBasis, numBasis, mixMaxDim_ );
  DblNumTns emptyTns2( numBasis, numBasis, 2 );
  SetValue( emptyTns, 0.0 );
  SetValue( emptyTns2, 0.0 );

  for(typename std::map<ElemMatKey, DblNumMat >::iterator
    My_iterator = hamDG.HMat().LocalMap().begin();
    My_iterator != hamDG.HMat().LocalMap().end();
    ++ My_iterator )
  {
    ElemMatKey matkey = (*My_iterator).first;

    std::map<ElemMatKey, DblNumMat>::iterator mi =
        distmixInnerSave_.LocalMap().find( matkey );

    if( mi == distmixInnerSave_.LocalMap().end() ){
      distmixInnerSave_.LocalMap()[matkey] = emptyMat;
    }
    else{
      DblNumMat&  mat = (*mi).second;
      blas::Copy( emptyMat.Size(), emptyMat.Data(), 1,
        mat.Data(), 1);
    }
  // Initialize distributed tensor in distdfInnerMat_
  // and distdvInnerMat_ 
    std::map<ElemMatKey, DblNumTns>::iterator ni =
        distdfInnerMat_.LocalMap().find( matkey );
    if( ni == distdfInnerMat_.LocalMap().end() ){
      distdfInnerMat_.LocalMap()[matkey] = emptyTns;
    }
    else{
      DblNumTns&  mixmat = (*ni).second;
      blas::Copy( emptyTns.Size(), emptyTns.Data(), 1,
        mixmat.Data(), 1);
    }
  
    std::map<ElemMatKey, DblNumTns>::iterator ki =
      distdvInnerMat_.LocalMap().find( matkey );
    if( ki == distdvInnerMat_.LocalMap().end() ){
      distdvInnerMat_.LocalMap()[matkey] = emptyTns;
    }
    else{
      DblNumTns&  mixmat = (*ki).second;
      blas::Copy( emptyTns.Size(), emptyTns.Data(), 1,
        mixmat.Data(), 1);
    }

    // Additional term for Broyden mixing
    if( hamDG.IsHybrid()  && HybridmixType_ == "broyden" )//    { 
    { 
      std::map<ElemMatKey, DblNumTns>::iterator bi =
         distcdfInnerMat_.LocalMap().find( matkey );
      if( bi == distcdfInnerMat_.LocalMap().end() ){
        distcdfInnerMat_.LocalMap()[matkey] = emptyTns2;
      }
      else{
        DblNumTns&  mixmat = (*bi).second;
        blas::Copy( emptyTns2.Size(), emptyTns2.Data(), 1,
          mixmat.Data(), 1);
      }
    }

  }

  return;
} 

// xmqin
void
SCFDG:: ProjectDM ( DistDblNumMat&  Oldbasis,
                    DistDblNumMat&  Newbasis,
                    DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>&  distDMMat )
{
  MPI_Barrier(domain_.comm);
  MPI_Barrier(domain_.colComm);
  MPI_Barrier(domain_.rowComm);
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);

  HamiltonianDG& hamDG = *hamDGPtr_;

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          DblNumMat& basisNew = Newbasis.LocalMap()[key];
          DblNumMat& basisOld = Oldbasis.LocalMap()[key];
          DblNumMat& SMat = hamDG.distSMat().LocalMap()[key];

          Int height = basisNew.m();
          Int numBasis = basisNew.n();
          Int numBasisTotal = 0;

          MPI_Allreduce( &numBasis, &numBasisTotal, 1, MPI_INT,
            MPI_SUM, domain_.rowComm );

          for( Int g = 0; g < numBasis; g++ ){
            Real *ptr1 = hamDG.LGLWeight3D().Data();
            Real *ptr2 = basisOld.VecData(g);
            for( Int l = 0; l < height; l++ ){
              *(ptr2++) *= *(ptr1++) ;
            }
          }

          Int width = numBasisTotal;

          Int widthBlocksize = width / mpisizeRow;
          Int heightBlocksize = height / mpisizeRow;

          Int widthLocal = widthBlocksize;
          Int heightLocal = heightBlocksize;

          if(mpirankRow < (width % mpisizeRow)){
            widthLocal = widthBlocksize + 1;
          }

          if(mpirankRow < (height % mpisizeRow)){
            heightLocal = heightBlocksize + 1;
          }

          DblNumMat basisOldRow( heightLocal, width );
          DblNumMat basisNewRow( heightLocal, width );

          AlltoallForward (basisOld, basisOldRow, domain_.rowComm);
          AlltoallForward (basisNew, basisNewRow, domain_.rowComm);

          DblNumMat localMatSTemp( width, width );
          SetValue( localMatSTemp, 0.0 );
          blas::Gemm( 'T', 'N', width, width, heightLocal,
              1.0, basisNewRow.Data(), heightLocal,
              basisOldRow.Data(), heightLocal, 0.0,
              localMatSTemp.Data(), width );

          SMat.Resize(width, width);
          SetValue( SMat, 0.0 );
          MPI_Allreduce( localMatSTemp.Data(),
              SMat.Data(), width*width, 
              MPI_DOUBLE, MPI_SUM, domain_.rowComm );

       if(0){
          DblNumMat Mat2( width, width );
          SetValue( Mat2, 0.0 );

          blas::Gemm( 'T', 'N', width, width, width,
              1.0, SMat.Data(), width,
              SMat.Data(), width, 0.0,
              Mat2.Data(), width );

          for(Int  p = 0; p < Mat2.m(); p++){
              statusOFS <<" Mat2 " << Mat2(p, p)  << std::endl;
          }
        }
        } //Owner element
  }

  std::vector<Index3>  getKeys_list;
  for(typename std::map<ElemMatKey, DblNumMat >::iterator
    get_neighbors_from_DM_iterator = distDMMat.LocalMap().begin();
    get_neighbors_from_DM_iterator != distDMMat.LocalMap().end();
    ++get_neighbors_from_DM_iterator)
  {
    Index3 key =  (get_neighbors_from_DM_iterator->first).first;
    Index3 neighbor_key = (get_neighbors_from_DM_iterator->first).second;

    if(neighbor_key == key)
      continue;
    else
    getKeys_list.push_back(neighbor_key);
  }

  hamDG.distSMat().GetBegin( getKeys_list, NO_MASK );
  hamDG.distSMat().GetEnd( NO_MASK );

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          Int numBasisElem =  esdfParam.numALBElem(i,j,k);

          for(typename  std::map<Index3, DblNumMat>::iterator 
            get_neighbors_iterator = hamDG.distSMat().LocalMap().begin();
            get_neighbors_iterator != hamDG.distSMat().LocalMap().end();
            ++get_neighbors_iterator)
          {
            Index3 key2 =  (*get_neighbors_iterator).first;
            
            DblNumMat& localDM = distDMMat.LocalMap()[ElemMatKey(key, key2)];
            DblNumMat& MatS = hamDG.distSMat().LocalMap()[key2];
            DblNumMat localMatTemp( numBasisElem, numBasisElem );
            SetValue( localMatTemp, 0.0 );
                    
            blas::Gemm( 'N', 'N', numBasisElem, numBasisElem, numBasisElem,
              1.0, MatS.Data(), numBasisElem,
              localDM.Data(), numBasisElem, 0.0,
              localMatTemp.Data(), numBasisElem );

            DblNumMat& MatST = hamDG.distSMat().LocalMap()[key2];
            SetValue ( localDM, 0.0 ); 

            blas::Gemm( 'N', 'T', numBasisElem, numBasisElem, numBasisElem,
              1.0, localMatTemp.Data(), numBasisElem,
              MatST.Data(), numBasisElem, 0.0,
              localDM.Data(), numBasisElem );
           } // iterator

         } //Owner element
  }  //for i

  return;
}

void 
SCFDG:: SVDLocalizeBasis ( Int iter, 
                   Index3 numGridExtElem,
//                   Index3 numGridExtElemFine,
                   Index3 numGridElemFine,
                   Index3 numLGLGrid,
                   Spinor& psi,
                   DblNumMat& basis ) 
{
  MPI_Barrier(domain_.rowComm);
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);

  HamiltonianDG& hamDG = *hamDGPtr_;
  Real timeSta, timeEnd;

  GetTime( timeSta );
  // LL: 11/11/2019.
  // For debugging purposes, let all eigenfunctions have non-negative averages. 
  // This should fix the sign flips due to LAPACK (but this would not fix the problem
  // due to degenerate eigenvectors
  {
    for( Int l = 0; l < psi.NumState(); l++ ){
      Real sum_psi = 0.0;
      for( Int i = 0; i < psi.NumGridTotal(); i++ ){
        sum_psi += psi.Wavefun(i,0,l);
//      for( Int i = 0; i < psi.NumGridTotalFine(); i++ ){
//        sum_psi += psi.WavefunFine(i,0,l);
      }
      Real sgn = (sum_psi >= 0.0) ? 1.0 : -1.0;
      blas::Scal( psi.NumGridTotal(), sgn, psi.Wavefun().VecData(0,l), 1 );
//      blas::Scal( psi.NumGridTotalFine(), sgn, psi.WavefunFine().VecData(0,l), 1 );
    }
  }

  DblNumTns&   LGLWeight3D = hamDG.LGLWeight3D();
  DblNumTns    sqrtLGLWeight3D( numLGLGrid[0], numLGLGrid[1], numLGLGrid[2] );

  Real *ptr1 = LGLWeight3D.Data(), *ptr2 = sqrtLGLWeight3D.Data();
  for( Int i = 0; i < numLGLGrid.prod(); i++ ){
    *(ptr2++) = std::sqrt( *(ptr1++) );
  }

  // Int numBasis = psi.NumState() + 1;
  // Compute numBasis in the presence of numUnusedState
  Int numBasisTotal = psi.NumStateTotal() - numUnusedState_;

  Int numBasis; // local number of basis functions
  numBasis = numBasisTotal / mpisizeRow;
  if( mpirankRow < (numBasisTotal % mpisizeRow) )
    numBasis++;

  Int numBasisTotalTest = 0;
  mpi::Allreduce( &numBasis, &numBasisTotalTest, 1, MPI_SUM, domain_.rowComm );
  if( numBasisTotalTest != numBasisTotal ){
    statusOFS << "numBasisTotal = " << numBasisTotal << std::endl;
    statusOFS << "numBasisTotalTest = " << numBasisTotalTest << std::endl;
    ErrorHandling("Sum{numBasis} = numBasisTotal does not match on local element.");
  }

  // FIXME The constant mode is now not used.
  DblNumMat localBasis( numLGLGrid.prod(), numBasis );
  SetValue( localBasis, 0.0 );

  //FIXME  xmqin This transform remove the normalization
  for( Int l = 0; l < numBasis; l++ ){
    InterpPeriodicUniformToLGL( 
//    InterpPeriodicUniformFineToLGL( 
        numGridExtElem,
//        numGridExtElemFine,
        numLGLGrid,
	psi.Wavefun().VecData(0, l), 
//        psi.WavefunFine().VecData(0, l), 
        localBasis.VecData(l) );
  }

  GetTime( timeEnd );
  statusOFS << "Time for interpolating basis = "     << timeEnd - timeSta
      << " [s]" << std::endl;
  // Post processing for the basis functions on the LGL grid.
  // Perform GEMM and threshold the basis functions for the
  // small matrix.
  //
  // This method might have lower numerical accuracy, but is
  // much more scalable than other known options.

  GetTime( timeSta );

  // Scale the basis functions by sqrt(weight).  This
  // allows the consequent SVD decomposition of the form
  //
  // X' * W * X
  for( Int g = 0; g < localBasis.n(); g++ ){
    Real *ptr1 = localBasis.VecData(g);
    Real *ptr2 = sqrtLGLWeight3D.Data();
    for( Int l = 0; l < localBasis.m(); l++ ){
      *(ptr1++)  *= *(ptr2++);
    }
  }

  // Convert the column partition to row partition
//  Int height = psi.NumGridTotal() * psi.NumComponent();
  Int heightLGL = numLGLGrid.prod();
//  Int heightElem = numGridElemFine.prod();
  Int width = numBasisTotal;

  Int widthBlocksize = width / mpisizeRow;
//  Int heightBlocksize = height / mpisizeRow;
  Int heightLGLBlocksize = heightLGL / mpisizeRow;
//  Int heightElemBlocksize = heightElem / mpisizeRow;

  Int widthLocal = widthBlocksize;
//  Int heightLocal = heightBlocksize;
  Int heightLGLLocal = heightLGLBlocksize;
//  Int heightElemLocal = heightElemBlocksize;

  if(mpirankRow < (width % mpisizeRow)){
    widthLocal = widthBlocksize + 1;
  }

//  if(mpirankRow < (height % mpisizeRow)){
//    heightLocal = heightBlocksize + 1;
//  }

  if(mpirankRow < (heightLGL % mpisizeRow)){
    heightLGLLocal = heightLGLBlocksize + 1;
  }

//  if(mpirankRow == (heightElem % mpisizeRow)){
//    heightElemLocal = heightElemBlocksize + 1;
//  }

  // FIXME Use AlltoallForward and AlltoallBackward
  // functions to replace below

  DblNumMat MMat( numBasisTotal, numBasisTotal );
  DblNumMat MMatTemp( numBasisTotal, numBasisTotal );
  SetValue( MMat, 0.0 );
  SetValue( MMatTemp, 0.0 );
  Int numLGLGridTotal = numLGLGrid.prod();
  Int numLGLGridLocal = heightLGLLocal;

  DblNumMat localBasisRow(heightLGLLocal, numBasisTotal );
  SetValue( localBasisRow, 0.0 );

  AlltoallForward (localBasis, localBasisRow, domain_.rowComm);

  SetValue( MMatTemp, 0.0 );
  blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal, numLGLGridLocal,
      1.0, localBasisRow.Data(), numLGLGridLocal, 
      localBasisRow.Data(), numLGLGridLocal, 0.0,
      MMatTemp.Data(), numBasisTotal );

  SetValue( MMat, 0.0 );
  MPI_Allreduce( MMatTemp.Data(), MMat.Data(), numBasisTotal * numBasisTotal, MPI_DOUBLE, MPI_SUM, domain_.rowComm );

  // The following operation is only performed on the
  // master processor in the row communicator

  DblNumMat    U( numBasisTotal, numBasisTotal );
  DblNumMat   VT( numBasisTotal, numBasisTotal );
  DblNumVec    S( numBasisTotal );
  SetValue(U, 0.0);
  SetValue(VT, 0.0);
  SetValue(S, 0.0);

  MPI_Barrier( domain_.rowComm );

  if ( mpirankRow == 0) {
    lapack::QRSVD( numBasisTotal, numBasisTotal, 
        MMat.Data(), numBasisTotal,
        S.Data(), U.Data(), U.m(), VT.Data(), VT.m() );
  } 
  // Broadcast U and S
  MPI_Bcast(S.Data(), numBasisTotal, MPI_DOUBLE, 0, domain_.rowComm);
  MPI_Bcast(U.Data(), numBasisTotal * numBasisTotal, MPI_DOUBLE, 0, domain_.rowComm);
  MPI_Bcast(VT.Data(), numBasisTotal * numBasisTotal, MPI_DOUBLE, 0, domain_.rowComm);

  MPI_Barrier( domain_.rowComm );

  for( Int g = 0; g < numBasisTotal; g++ ){
    S[g] = std::sqrt( S[g] );
  }
  // Total number of SVD basis functions. NOTE: Determined at the first
  // outer SCF and is not changed later. This facilitates the reuse of
  // symbolic factorization
  if( iter == 1 ){
    numSVDBasisTotal_ = 0;    
    for( Int g = 0; g < numBasisTotal; g++ ){
      if( S[g] / S[0] > SVDBasisTolerance_ )
        numSVDBasisTotal_++;
    }
  }
//    else{
//     // Reuse the value saved in numSVDBasisTotal
//      statusOFS 
//        << "NOTE: The number of basis functions (after SVD) " 
//        << "is the same as the number in the first SCF iteration." << std::endl
//        << "This facilitates the reuse of symbolic factorization in PEXSI." 
//        << std::endl;
//    }

  Int numSVDBasisBlocksize = numSVDBasisTotal_ / mpisizeRow;

  Int numSVDBasisLocal = numSVDBasisBlocksize;    

  if(mpirankRow < (numSVDBasisTotal_ % mpisizeRow)){
    numSVDBasisLocal = numSVDBasisBlocksize + 1;
  }

  Int numSVDBasisTotalTest = 0;

  mpi::Allreduce( &numSVDBasisLocal, &numSVDBasisTotalTest, 1, MPI_SUM, domain_.rowComm );

  if( numSVDBasisTotal_ != numSVDBasisTotalTest ){
    statusOFS << "numSVDBasisLocal = " << numSVDBasisLocal << std::endl;
    statusOFS << "numSVDBasisTotal = " << numSVDBasisTotal_ << std::endl;
    statusOFS << "numSVDBasisTotalTest = " << numSVDBasisTotalTest << std::endl;
    ErrorHandling("numSVDBasisTotal != numSVDBasisTotalTest");
  }

  // Multiply X <- X*U in the row-partitioned format
  // Get the first numSVDBasis which are significant.

  basis.Resize( numLGLGridTotal, numSVDBasisLocal );
  DblNumMat basisRow( numLGLGridLocal, numSVDBasisTotal_ );

  SetValue( basis, 0.0 );
  SetValue( basisRow, 0.0 );

  for( Int g = 0; g < numSVDBasisTotal_; g++ ){
    blas::Scal( numBasisTotal, 1.0 / S[g], U.VecData(g), 1 );
  }

  // FIXME
  blas::Gemm( 'N', 'N', numLGLGridLocal, numSVDBasisTotal_,
      numBasisTotal, 1.0, localBasisRow.Data(), numLGLGridLocal,
      U.Data(), numBasisTotal, 0.0, basisRow.Data(), numLGLGridLocal );

  AlltoallBackward (basisRow, basis, domain_.rowComm);

  // FIXME
  // row-partition to column partition via MPI_Alltoallv

  // Unscale the orthogonal basis functions by sqrt of
  // integration weight
  // FIXME

  for( Int g = 0; g < basis.n(); g++ ){
    Real *ptr1 = basis.VecData(g);
    Real *ptr2 = sqrtLGLWeight3D.Data();
    for( Int l = 0; l < basis.m(); l++ ){
      *(ptr1++)  /= *(ptr2++);
    }
  }

// xmqin add for fix the phase of ALBs
/*                    for( Int g = 0; g < basis.n(); g++ ){
                      Real *ptr = basis.VecData(g);
                      Real sum = 0.0;
                      for( Int l = 0; l < basis.m(); l++ ){
                       sum += *ptr++ ;
                      }
                    if (sum <= 0.0) {
                      ptr = basis.VecData(g);
                      for( Int l = 0; l < basis.m(); l++ ){
                          *ptr = - (*ptr);
                           ptr++;
                    }
                   }
                   }
*/

#if ( _DEBUGlevel_ >= 1 )
  statusOFS << " Singular values of the basis = " 
      << S << std::endl;
#endif

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Number of significant SVD basis = " 
      << numSVDBasisTotal_ << std::endl;
#endif

  MPI_Barrier( domain_.rowComm );

  GetTime( timeEnd );
  statusOFS << "Time for SVD of basis = "     << timeEnd - timeSta
      << " [s]" << std::endl;

  return;
}



////fengjw for RTTDDFT


void SCFDG::RTTDDFT_RK4(double time,double deltaT)
{

Int mpirank, mpisize;
MPI_Comm_rank( domain_.comm, &mpirank );
MPI_Comm_size( domain_.comm, &mpisize );
Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

std::complex<double> complexOne (1.0,0.0);
std::complex<double> complexZero (0.0,0.0);

HamiltonianDG&  hamDG = *hamDGPtr_;
Int ntot  = hamDGPtr_->NumUniformGridElemFine().prod();
DistCpxNumMat psi;

hamDG.EigvecCoefCpx().SetComm(hamDG.EigvecCoef().Comm());
hamDG.EigvecCoefCpx().Prtn()=hamDG.EigvecCoef().Prtn();

psi.SetComm(hamDG.EigvecCoefCpx().Comm());
psi.Prtn()=hamDG.EigvecCoefCpx().Prtn();


double timeConstructHam=0.0;
double timeUpdate=0.0;
double timeTimes=0.0;
double timeDensity=0.0;
double timeVext=0.0;
double timeDipole=0.0;
double timeEnergy=0.0;


double timeRKSta=0.0;
double timeRKEnd=0.0;

int Ehrenfest=0;

double timeSta=0.0;
double timeEnd=0.0;


int noccLocal;



CalculateKSEnergy();

Begin_DG_TDDFT_=1;


DistCpxNumMat k;
DistCpxNumMat ans;
//DistCpxNumMat k3;
////DistCpxNumMat k4;
//
//
k.SetComm(hamDG.EigvecCoef().Comm());
ans.SetComm(hamDG.EigvecCoef().Comm());
////k3.SetComm(hamDG.EigvecCoef().Comm());
////k4.SetComm(hamDG.EigvecCoef().Comm());
//
k.Prtn()=hamDG.EigvecCoef().Prtn();
ans.Prtn()=hamDG.EigvecCoef().Prtn();
////k3.Prtn()=hamDG.EigvecCoef().Prtn();
////k4.Prtn()=hamDG.EigvecCoef().Prtn();
//



//改变占据为1
//hamDG.OccupationRate()[1]=1.0;
//hamDG.OccupationRate()[0]=0.0;

//构造complex类型的本征函数以及dg哈密顿量
//

  for( std::map<ElemMatKey, DblNumMat>::iterator
      mi  = hamDG.HMat().LocalMap().begin();
      mi != hamDG.HMat().LocalMap().end(); ++mi ){
    ElemMatKey key = (*mi).first;
	int m=(hamDG.HMat().LocalMap().find(key))->second.m();
	int n=(hamDG.HMat().LocalMap().find(key))->second.n();



       CpxNumMat HTemp(m,n);
        for(int ntotidx=0;ntotidx<m*n;ntotidx++)
        {
//	if(mpirank==0)
//std::cout<<"H:  "<<((hamDG.HMat().LocalMap().find(key))->second.Data())[ntotidx]<<std::endl;

	HTemp.Data()[ntotidx]=std::complex<double>(((hamDG.HMat().LocalMap().find(key))->second.Data())[ntotidx],0.0);

//        if(mpirank==0)
//std::cout<<"HCpx:  "<<HTemp.Data()[ntotidx]<<std::endl;
       // ((hamDG.HMatCpx().LocalMap().find(key))->second.Data())[ntotidx]=std::complex<double>(((hamDG.HMat().LocalMap().find(key))->second.Data())[ntotidx],0.0);
        }

	hamDG.HMatCpx().LocalMap()[key]=HTemp;
//    statusOFS << key.first << " -- " << key.second << std::endl;
  }


for( Int kk = 0; kk < numElem_[2]; kk++ )
  for( Int j = 0; j < numElem_[1]; j++ )
    for( Int i = 0; i < numElem_[0]; i++ ){
      Index3 key( i, j, kk );
      if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
        const std::vector<Int>&  idx = hamDG.ElemBasisIdx()(i, j, kk);
//        DblNumMat& localCoef  = hamDG.EigvecCoef().LocalMap()[key];
        //(hamDG.EigvecCoefCpx().LocalMap()[key]).Resize(idx.size(),hamDG.NumStateTotal());
	//(psi.LocalMap()[key]).Resize(idx.size(),hamDG.NumStateTotal());
	int m=(hamDG.EigvecCoef().LocalMap().find(key))->second.m();
	int n=(hamDG.EigvecCoef().LocalMap().find(key))->second.n();


//Distributor wavefunction
noccLocal=n/mpisizeRow;
int MyBegin=0;
if(mpirankRow<n%mpisizeRow)
{
noccLocal++;
MyBegin=mpirankRow*noccLocal;
}
else
{
MyBegin=mpirankRow*noccLocal+(n%mpisizeRow);
}


////
	CpxNumMat psiTemp(m,noccLocal);	
        for(int ntotidx=MyBegin;ntotidx<m*noccLocal;ntotidx++)
                {
                //(hamDG.EigvecCoefCpx().LocalMap()[key])[ntotidx]=std::complex<double>((hamDG.EigvecCoef().LocalMap()[key])[ntotidx],0.0);//转换成complex类型
		psiTemp.Data()[ntotidx]=std::complex<double>(((hamDG.EigvecCoef().LocalMap().find(key))->second.Data())[ntotidx],0.0);
                //((hamDG.EigvecCoefCpx().LocalMap().find(key))->second.Data())[ntotidx]=std::complex<double>(((hamDG.EigvecCoef().LocalMap().find(key))->second.Data())[ntotidx],0.0);
		//((psi.LocalMap().find(key))->second.Data())[ntotidx]=std::complex<double>(((hamDG.EigvecCoef().LocalMap().find(key))->second.Data())[ntotidx],0.0);
//              (hamDG.EigvecCoefCpx().LocalMap()[key]).Resize(idx.size(),hamDG.NumStateTotal());
//              localCoef.Resize( idx.size(), hamDG.NumStateTotal());           
                }
		//((hamDG.EigvecCoefCpx().LocalMap().find(key))->second.Data())[ntotidx]=std::complex<double>(((hamDG.EigvecCoef().LocalMap().find(key))->second.Data())[ntotidx],0.0);
		psi.LocalMap()[key]=psiTemp;
//		k.LocalMap()[key]=psiTemp;
//		ans.LocalMap()[key]=psiTemp;
		hamDG.EigvecCoefCpx().LocalMap()[key]=psiTemp;
//        localCoef.Resize( idx.size(), hamDG.NumStateTotal() );
      }
    }


MPI_Barrier( domain_.comm );
MPI_Barrier( domain_.rowComm );
MPI_Barrier( domain_.colComm );




for( Int kk = 0; kk < numElem_[2]; kk++ )
  for( Int j = 0; j < numElem_[1]; j++ )
    for( Int i = 0; i < numElem_[0]; i++ ){
      Index3 key( i, j, kk );
      if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                psi.LocalMap()[key]=hamDG.EigvecCoefCpx().LocalMap()[key];
		ans.LocalMap()[key]=hamDG.EigvecCoefCpx().LocalMap()[key];
  		k.LocalMap()[key]=hamDG.EigvecCoefCpx().LocalMap()[key];

      }
    }


MPI_Barrier( domain_.comm );
MPI_Barrier( domain_.rowComm );
MPI_Barrier( domain_.colComm );


//DistCpxNumMat k;
//DistCpxNumMat ans;
//DistCpxNumMat k3;
//DistCpxNumMat k4;


//k.SetComm(hamDG.EigvecCoef().Comm());
//ans.SetComm(hamDG.EigvecCoef().Comm());
//k3.SetComm(hamDG.EigvecCoef().Comm());
//k4.SetComm(hamDG.EigvecCoef().Comm());

//k.Prtn()=hamDG.EigvecCoef().Prtn();
//ans.Prtn()=hamDG.EigvecCoef().Prtn();
//k3.Prtn()=hamDG.EigvecCoef().Prtn();
//k4.Prtn()=hamDG.EigvecCoef().Prtn();


/*
  Hmat_times_my_dist_mat=-i*H*my_dist_mat
void scfdg_hamiltonian_times_distmat_Cpx(DistVec<Index3, CpxNumMat, ElemPrtn>  &my_dist_mat,
    DistVec<Index3, CpxNumMat, ElemPrtn>  &Hmat_times_my_dist_mat);

void
  SCFDG::scfdg_distmat_update_Cpx(DistVec<Index3, CpxNumMat, ElemPrtn>  &dist_mat_a,
      complex<double> scal_a,
      DistVec<Index3, CpxNumMat, ElemPrtn>  &dist_mat_b,
      complex<double> scal_b)

*/
std::vector<double>    tlist;
int size = time/ deltaT + 1;
tlist.resize(size);
for(int i = 0; i < size; i++)
  tlist[i] = i * deltaT;



DblNumVec atomMass;
PeriodTable &ptable=*ptablePtr_;
std::vector<Atom>& atomList=hamDG.AtomList();


int numAtom=hamDG.AtomList().size();
  std::vector<Point3>  atompos_mid(numAtom);
  std::vector<Point3>  atompos_fin(numAtom);
// might need to setup others, will add here.

if(Ehrenfest)
{
  atomMass.Resize( numAtom );
  for(Int a=0; a < numAtom; a++) {
    Int atype = atomList[a].type;
    if (ptable.ptemap().find(atype)==ptable.ptemap().end() ){
      ErrorHandling( "Cannot find the atom type." );
    }
    atomMass[a]=amu2au*ptable.Mass(atype);
  }

hamDG.CalculateForce_Cpx(*distfftPtr_);
}



int iterMax=(time/deltaT);

if(mpirank==0)
{
std::cout<<"clean WaveFunction Groud" <<std::endl;
}




  // Clean the eigenvecs begin RTTDDFT loop to OPT Mem
  {
    std::vector<Index3>  eraseKey;
    for( std::map<Index3, DblNumMat>::iterator
        mi  = hamDG.EigvecCoef().LocalMap().begin();
        mi != hamDG.EigvecCoef().LocalMap().end(); ++mi ){
      Index3 key = (*mi).first;
      if( hamDG.EigvecCoef().Prtn().Owner(key) != (mpirank / dmRow_) ){
        eraseKey.push_back( key );
      }
    }
    for( std::vector<Index3>::iterator vi = eraseKey.begin();
        vi != eraseKey.end(); ++vi ){
      hamDG.EigvecCoef().LocalMap().erase( *vi );
    }
  }



 if(mpirank==0)
 {
 std::cout<<"Reduce LocalBasis" <<std::endl;
 }



hamDG.BasisLGL_Cpx().SetComm(hamDG.EigvecCoefCpx().Comm());

hamDG.BasisLGL_Cpx().Prtn()=hamDG.EigvecCoefCpx().Prtn();
 
MPI_Barrier( domain_.comm );
statusOFS << std::endl;
Print( statusOFS, "density-2 " );
statusOFS << std::endl;


 
  // Compute the local density in each element
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          DblNumMat& localBasis = hamDG.BasisLGL().LocalMap()[key];
          Int numGrid  = localBasis.m();
          Int numBasis = localBasis.n();
 Int numBasisTotal = 0;
 MPI_Allreduce( &numBasis, &numBasisTotal, 1, MPI_INT, MPI_SUM, domain_.rowComm );


if( numBasisTotal == 0 )
  continue;
 
 
DblNumMat localBasisLGL(numGrid,numBasisTotal);
 

DblNumMat localBasisAll(numGrid,numBasisTotal);

 SetValue(localBasisAll,0.0);
 SetValue(localBasisLGL,0.0);
 int count=0;
 for(int idx=mpirankRow;idx<numBasisTotal;idx=idx+mpisizeRow)
 {
 lapack::Lacpy( 'A', numGrid, 1, localBasis.VecData(count), numGrid,
     localBasis.VecData(idx), numGrid );
 count++;
 }
 
 MPI_Allreduce(localBasisAll.Data(),
     localBasisLGL.Data(),
     numGrid*numBasisTotal,
     MPI_DOUBLE,
     MPI_SUM,
     domain_.rowComm);


CpxNumMat localBasisLGL_Cpx(numGrid,numBasisTotal);

for(int a=0;a<numGrid*numBasisTotal;a++)
{
(localBasisLGL_Cpx.Data())[a]=std::complex<double>((localBasisLGL.Data())[a],0.0);
}

hamDG.BasisLGL_Cpx().LocalMap()[key]=localBasisLGL_Cpx;
 
 }}


MPI_Barrier( domain_.comm );
statusOFS << std::endl;
Print( statusOFS, "density-1 " );
statusOFS << std::endl;



/*
{
   const Index3 key = (hamDG.EigvecCoefCpx().LocalMap().begin())->first; // Will use same key as eigenvectors
   CpxNumMat &eigvecs_local = (hamDG.EigvecCoefCpx().LocalMap().begin())->second;
 

 ans.LocalMap()[key].Resize(eigvecs_local.m(), eigvecs_local.n());
   k.LocalMap()[key].Resize(eigvecs_local.m(), eigvecs_local.n());
}
*/
MPI_Barrier( domain_.comm );
statusOFS << std::endl;
Print( statusOFS, "density0 " );
statusOFS << std::endl;




if(mpirank==0)
{
std::cout<<"iterMax:" <<iterMax<<std::endl;
}


bool isCalculateEnergy=false;

for(int iterRK=0;iterRK<iterMax;iterRK++)
{


timeConstructHam=0.0;
timeUpdate=0.0;
timeTimes=0.0;
timeDensity=0.0;
timeEnergy=0.0;
timeVext=0.0;
timeDipole=0.0;

 GetTime( timeRKSta );





double ti = tlist[iterRK];
double tf = tlist[iterRK+1];
double dT = tf - ti;
double tmid =  (ti + tf)/2.0;


if(Ehrenfest)
{



    // Update velocity and position when doing ehrenfest dynamics

      for(Int a=0; a<numAtom; a++) {
        atompos_mid[a]  = atomList[a].pos + (atomList[a].vel/2.0 + atomList[a].force*deltaT/atomMass[a]/8.0) * deltaT;
        atompos_fin[a]  = atomList[a].pos + (atomList[a].vel + atomList[a].force*deltaT/atomMass[a]/2.0) * deltaT;

      }

//Adjust Atom Pos
	for( int a = 0; a < numAtom; a++)
  {
    atompos_mid[a][0] -= IRound(atompos_mid[a][0] / domain_.length[0]) * domain_.length[0];
    atompos_mid[a][1] -= IRound(atompos_mid[a][1] / domain_.length[1]) * domain_.length[1];
    atompos_mid[a][2] -= IRound(atompos_mid[a][2] / domain_.length[2]) * domain_.length[2];
  
   atompos_fin[a][0] -= IRound(atompos_fin[a][0] / domain_.length[0]) * domain_.length[0];
   atompos_fin[a][1] -= IRound(atompos_fin[a][1] / domain_.length[1]) * domain_.length[1];
   atompos_fin[a][2] -= IRound(atompos_fin[a][2] / domain_.length[2]) * domain_.length[2];

  }//Adjust atom pos


}

MPI_Barrier( domain_.comm );
statusOFS << std::endl;
Print( statusOFS, "density1 " );
statusOFS << std::endl;



/*
{
//debug
int c=1;
int allc = 0;
MPI_Allreduce( &c, &allc, 1, MPI_INT, MPI_SUM, domain_.comm );


if(mpirank==0)
{
std::cout<<"ProcessNum:" <<allc<<std::endl;
}
}
*/


MPI_Barrier( domain_.comm );
statusOFS << std::endl;
Print( statusOFS, "density " );
statusOFS << std::endl;




MPI_Barrier( domain_.comm );
GetTime(timeSta);
hamDG.CalculateDensity_Cpx( hamDG.Density(), hamDG.DensityLGL() );
GetTime(timeEnd);

 timeDensity+=(timeEnd-timeSta);


GetTime(timeSta);
AddVext(0,0.1,ti);
GetTime(timeEnd);

timeVext+=(timeEnd-timeSta);


GetTime(timeSta);
UpdateHam();
GetTime(timeEnd);

timeConstructHam+=(timeEnd-timeSta);



//scfdg_hamiltonian_times_distmat_Cpx(hamDG.EigvecCoefCpx(),k1);

GetTime(timeSta);
//scfdg_Hamiltonian_times_eigenvectors_Cpx(k);

scfdg_hamiltonian_times_distmat_Cpx(hamDG.EigvecCoefCpx(),k); // Y = H * X

GetTime(timeEnd);

timeTimes+=(timeEnd-timeSta);

/////////Compute Etot 


//1.Ekin_

//if(iterRK==0)
//{
//std::cout<<"Ekin_0: "<<Ekin_<<std::endl;
//}

if(isCalculateEnergy)
{
GetTime(timeSta);
Ekin_=0.0;
UpdateEkin(psi,k);
CalculateSecondOrderEnergy();
CalculateKSEnergy();

GetTime(timeEnd);
timeEnergy+=(timeEnd-timeSta);
///////////
}








//debugH正确性
/*
if(mpirank==2)
{

  for( std::map<ElemMatKey, DblNumMat>::iterator
      mi  = hamDG.HMat().LocalMap().begin();
      mi != hamDG.HMat().LocalMap().end(); ++mi ){
    ElemMatKey key = (*mi).first;
        int m=(hamDG.HMat().LocalMap().find(key))->second.m();
        int n=(hamDG.HMat().LocalMap().find(key))->second.n();

for(int debugidx=0;debugidx<50;debugidx++)
std::cout<<hamDG.HMatCpx().LocalMap()[key].Data()[debugidx]<<"----"<<hamDG.HMat().LocalMap()[key].Data()[debugidx]<<std::endl;

std::cout<<"--------------------Cpx  real------------"<<iterRK<<std::endl;
}

  }
*/





//debugH正确性



/*
//debugH*X正确性
DistDblNumMat k1real;
scfdg_Hamiltonian_times_eigenvectors(k1real);

if(mpirank==0)
{
Index3 key( 0, 0, 0 );
for(int debugidx=0;debugidx<50;debugidx++)
std::cout<<k1.LocalMap()[key].Data()[debugidx]<<"----"<<k1real.LocalMap()[key].Data()[debugidx]<<std::endl;

std::cout<<"--------------------k1  k1real------------"<<iterRK<<std::endl;
}
MPI_Barrier( domain_.comm );
MPI_Barrier( domain_.rowComm );
MPI_Barrier( domain_.colComm );
//debug 加法正确性


//debugH*X正确性
*/


/*
//debug 加法正确性
if(mpirank==0)
{
Index3 key( 0, 0, 0 );
for(int debugidx=0;debugidx<50;debugidx++)
std::cout<<k1.LocalMap()[key].Data()[debugidx]<<"+++"<<hamDG.EigvecCoefCpx().LocalMap()[key].Data()[debugidx]<<std::endl;

std::cout<<"-----------------加前----k1  eigvec------------"<<iterRK<<std::endl;
}
MPI_Barrier( domain_.comm );
MPI_Barrier( domain_.rowComm );
MPI_Barrier( domain_.colComm );
//debug 加法正确性

scfdg_distmat_update_Cpx(k1,
    std::complex<double> (0.5*deltaT),
    hamDG.EigvecCoefCpx(),
    complexOne);

//debug 加法正确性
if(mpirank==0)
{
Index3 key( 0, 0, 0 );
for(int debugidx=0;debugidx<50;debugidx++)
std::cout<<hamDG.EigvecCoefCpx().LocalMap()[key].Data()[debugidx]<<std::endl;

std::cout<<"-----------------加后----eigvec------------"<<iterRK<<std::endl;
}
MPI_Barrier( domain_.comm );
MPI_Barrier( domain_.rowComm );
MPI_Barrier( domain_.colComm );
//debug 加法正确性
*/


GetTime(timeSta);
scfdg_distmat_update_Cpx(k,
    std::complex<double> (0.5*deltaT),
    hamDG.EigvecCoefCpx(),
    complexOne);
GetTime(timeEnd);

timeUpdate+=(timeEnd-timeSta);




MPI_Barrier( domain_.comm );
MPI_Barrier( domain_.rowComm );
MPI_Barrier( domain_.colComm );


GetTime(timeSta);
hamDG.CalculateDensity_Cpx( hamDG.Density(), hamDG.DensityLGL() );
GetTime(timeEnd);

timeDensity+=(timeEnd-timeSta);


GetTime(timeSta);
AddVext(0,0.1,tmid);
GetTime(timeEnd);

timeVext+=(timeEnd-timeSta);


//change atom pos to mid
if(Ehrenfest)
{
    for( Int a = 0; a < numAtom; a++ ){
      atomList[a].pos  = atompos_mid[a];
    }

hamDG.UpdateHamiltonianDG( hamDG.AtomList() );
hamDG.CalculatePseudoPotential( ptable );

}

GetTime(timeSta);
UpdateHam();
GetTime(timeEnd);
timeConstructHam+=(timeEnd-timeSta);


GetTime(timeSta);
 scfdg_distmat_update_Cpx(k,
     std::complex<double> (deltaT/6.0),
     ans,
     complexZero);
GetTime(timeEnd);
timeUpdate+=(timeEnd-timeSta);



//scfdg_hamiltonian_times_distmat_Cpx(hamDG.EigvecCoefCpx(),k2);
//scfdg_Hamiltonian_times_eigenvectors_Cpx(k);
scfdg_hamiltonian_times_distmat_Cpx(hamDG.EigvecCoefCpx(),k); // Y = H * X


GetTime(timeSta);
scfdg_distmat_update_Cpx(k,
    std::complex<double> (0.5*deltaT),
    hamDG.EigvecCoefCpx(),
    complexZero);


scfdg_distmat_update_Cpx(psi,
    complexOne,
    hamDG.EigvecCoefCpx(),
    complexOne);
GetTime(timeEnd);
timeUpdate+=(timeEnd-timeSta);


GetTime(timeSta);
hamDG.CalculateDensity_Cpx( hamDG.Density(), hamDG.DensityLGL() );
GetTime(timeEnd);
timeDensity+=(timeEnd-timeSta);



GetTime(timeSta);
UpdateHam();
GetTime(timeEnd);
timeConstructHam+=(timeEnd-timeSta);

GetTime(timeSta);
scfdg_distmat_update_Cpx(k,
    std::complex<double> (deltaT/3.0),
    ans,
    complexOne);
GetTime(timeEnd);

timeUpdate+=(timeEnd-timeSta);

//scfdg_hamiltonian_times_distmat_Cpx(hamDG.EigvecCoefCpx(),k3);

GetTime(timeSta);
//scfdg_Hamiltonian_times_eigenvectors_Cpx(k);
scfdg_hamiltonian_times_distmat_Cpx(hamDG.EigvecCoefCpx(),k); // Y = H * X

GetTime(timeEnd);
timeTimes+=(timeEnd-timeSta);

GetTime(timeSta);
scfdg_distmat_update_Cpx(k,
    std::complex<double> (deltaT),
    hamDG.EigvecCoefCpx(),
    complexZero);


scfdg_distmat_update_Cpx(psi,
    complexOne,
    hamDG.EigvecCoefCpx(),
    complexOne);
GetTime(timeEnd);
timeUpdate+=(timeEnd-timeSta);

GetTime(timeSta);
hamDG.CalculateDensity_Cpx( hamDG.Density(), hamDG.DensityLGL() );
GetTime(timeEnd);
timeDensity+=(timeEnd-timeSta);



GetTime(timeSta);
AddVext(0,0.1,tf);
GetTime(timeEnd);

timeVext+=(timeEnd-timeSta);


//change atom pos to mid
if(Ehrenfest)
{
    for( Int a = 0; a < numAtom; a++ ){
      atomList[a].pos  = atompos_fin[a];
    }

hamDG.UpdateHamiltonianDG( hamDG.AtomList() );
hamDG.CalculatePseudoPotential( ptable );



}


GetTime(timeSta);
UpdateHam();
GetTime(timeEnd);
timeConstructHam+=(timeEnd-timeSta);


GetTime(timeSta);
scfdg_distmat_update_Cpx(k,
    std::complex<double> (deltaT/3.0),
    ans,
    complexOne);
GetTime(timeEnd);
timeUpdate+=(timeEnd-timeSta);

//scfdg_hamiltonian_times_distmat_Cpx(hamDG.EigvecCoefCpx(),k4);

GetTime(timeSta);
//scfdg_Hamiltonian_times_eigenvectors_Cpx(k);
scfdg_hamiltonian_times_distmat_Cpx(hamDG.EigvecCoefCpx(),k); // Y = H * X
GetTime(timeEnd);
timeTimes+=(timeEnd-timeSta);

GetTime(timeSta);
scfdg_distmat_update_Cpx(k,
    std::complex<double> (deltaT/6.0),
    ans,
    complexOne);
GetTime(timeEnd);
timeUpdate+=(timeEnd-timeSta);

///合并龙格结果
/*
for( Int k = 0; k < numElem_[2]; k++ )
  for( Int j = 0; j < numElem_[1]; j++ )
    for( Int i = 0; i < numElem_[0]; i++ ){
      Index3 key( i, j, k );
      if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
        const std::vector<Int>&  idx = hamDG.ElemBasisIdx()(i, j, k);
//        DblNumMat& localCoef  = hamDG.EigvecCoef().LocalMap()[key];
        //(hamDG.EigvecCoefCpx().LocalMap()[key]).Resize(idx.size(),hamDG.NumStateTotal());
        //(psi.LocalMap()[key]).Resize(idx.size(),hamDG.NumStateTotal());
//        int m=(hamDG.EigvecCoefCpx().LocalMap().find(key))->second.m();
//        int n=(hamDG.EigvecCoefCpx().LocalMap().find(key))->second.n();
int m=((hamDG.EigvecCoef().LocalMap().begin())->second).m();
int n=((hamDG.EigvecCoef().LocalMap().begin())->second).n();

for(Int iter = 0; iter < m*n ; iter ++)
  (((hamDG.EigvecCoefCpx().LocalMap().find(key))->second).Data())[iter] = (((psi.LocalMap().find(key))->second).Data())[iter]+(deltaT/6.0)*((((k1.LocalMap().find(key))->second).Data())[iter]+2.0*(((k2.LocalMap().find(key))->second).Data())[iter]+2.0*(((k3.LocalMap().find(key))->second).Data())[iter]+(((k4.LocalMap().find(key))->second).Data())[iter]);//scal_a * ptr_a[iter] + scal_b * ptr_b[iter];

}
}
*/

GetTime(timeSta);

scfdg_distmat_update_Cpx(psi,
    complexOne,
    hamDG.EigvecCoefCpx(),
    complexZero);

scfdg_distmat_update_Cpx(ans,
    complexOne,
    hamDG.EigvecCoefCpx(),
    complexOne);
GetTime(timeEnd);
timeUpdate+=(timeEnd-timeSta);

/*
scfdg_distmat_update_Cpx(k1,
    std::complex<double> (deltaT/6.0),
    hamDG.EigvecCoefCpx(),
    complexOne);


scfdg_distmat_update_Cpx(k2,
    std::complex<double> (deltaT/3.0),
    hamDG.EigvecCoefCpx(),
    complexOne);

scfdg_distmat_update_Cpx(k3,
    std::complex<double> (deltaT/3.0),
    hamDG.EigvecCoefCpx(),
    complexOne);

scfdg_distmat_update_Cpx(k4,
    std::complex<double> (deltaT/6.0),
    hamDG.EigvecCoefCpx(),
    complexOne);
*/
///合并龙格

//if(iterRK%10==0)
//{
//}
//CalculateDipole();

GetTime(timeSta);
hamDG.CalculateDensity_Cpx( hamDG.Density(), hamDG.DensityLGL() );
GetTime(timeEnd);

timeDensity+=(timeEnd-timeSta);

GetTime(timeSta);
UpdateHam();
GetTime(timeEnd);
timeConstructHam+=(timeEnd-timeSta);

GetTime(timeSta);
CalculateDipole();
GetTime(timeEnd);

timeDipole+=(timeEnd-timeSta);

PrintBlock( statusOFS, "Energy" );
statusOFS
  <<"Time:  "<<ti*au2fs<<"  fs"<<std::endl
  <<"DensityTime:    "<<timeDensity<<"  s"<<std::endl
  <<"Times Ham vector:  "<<timeTimes<<"  s"<<std::endl
  <<"Times Update:   "<<timeUpdate<<"   s"<<std::endl
  <<"Times Dipole:   "<<timeDipole<<"   s"<<std::endl
  <<"Times Vext:     "<<timeVext<<"   s"<<std::endl
  <<"Times Energy:   "<<timeEnergy<<"  s"<<std::endl
  <<"Times Construct Ham:  "<<timeConstructHam<<"   s"<<std::endl

  << "NOTE:  Ecor  = Exc - EVxc - Ehart - Eself + Evdw" << std::endl
  << "       Etot  = Ekin + Ecor" << std::endl
  << "       Efree = Etot    + Entropy" << std::endl << std::endl;
Print(statusOFS, "! EfreeHarris     = ",  EfreeHarris_, "[au]");
Print(statusOFS, "! Etot            = ",  Etot_*au2ev, "[ev]");
Print(statusOFS, "! Efree           = ",  Efree_, "[au]");
Print(statusOFS, "! Evdw            = ",  Evdw_, "[au]");
Print(statusOFS, "! Fermi           = ",  fermi_, "[au]");


for( Int k = 0; k < numElem_[2]; k++ )
  for( Int j = 0; j < numElem_[1]; j++ )
    for( Int i = 0; i < numElem_[0]; i++ ){
      Index3 key( i, j, k );
      if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                psi.LocalMap()[key]=hamDG.EigvecCoefCpx().LocalMap()[key];
      }
    }

//CalculateDipole();


  //update Velocity
  if(Ehrenfest){
//save force
std::vector<Point3>  atomforce(numAtom);
    for( Int a = 0; a < numAtom; a++ ){
      atomforce[a]   = atomList[a].force;
    }


    hamDG.CalculateForce_Cpx(*distfftPtr_);
    for( Int a = 0; a < numAtom; a++ ){
      atomList[a].vel = atomList[a].vel + (atomforce[a]/atomMass[a] + atomList[a].force/atomMass[a])*deltaT/2.0;
    } 
  }



 GetTime( timeRKEnd );

 statusOFS << std::endl << " Time for RK4 a cycle = "<< (timeRKEnd - timeRKSta ) << " s.";




}


return;
}////End RTTDDFT_RK4 

void 
SCFDG::scfdg_Hamiltonian_times_eigenvectors_Cpx(DistVec<Index3, CpxNumMat, ElemPrtn>  &result_mat)
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );
  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  Real timeSta, timeEnd;
  Real extra_timeSta, extra_timeEnd;
  Real filter_total_time = 0.0;

  HamiltonianDG&  hamDG = *hamDGPtr_;

  // We need to distribute bands according to rowComm since colComm is
  // used for element-wise partition.
  if(mpisizeRow > hamDG.NumStateTotal())
  {
    statusOFS << std::endl << " Number of processors per element exceeds number of bands !!"
      << std::endl << " Cannot continue with band-parallelization. "
      << std::endl << " Aborting ... " << std::endl;
    exit(1);

  }
  simple_distributor band_distributor(hamDG.NumStateTotal(), mpisizeRow, mpirankRow);

  // Create distributed matrices pluck_X, pluck_Y, pluck_Yt for filtering 
  const Index3 key = (hamDG.EigvecCoefCpx().LocalMap().begin())->first; // Will use same key as eigenvectors
  CpxNumMat &eigvecs_local = (hamDG.EigvecCoefCpx().LocalMap().begin())->second;

  Int local_width = band_distributor.current_proc_size;
  Int local_height = eigvecs_local.m();
  Int local_pluck_sz = local_height * local_width;

  DistVec<Index3, CpxNumMat, ElemPrtn>  pluck_X; 
  pluck_X.Prtn() = elemPrtn_;
  pluck_X.SetComm(domain_.colComm);
  pluck_X.LocalMap()[key].Resize(local_height, local_width);

  DistVec<Index3, CpxNumMat, ElemPrtn>  pluck_Y; 
  pluck_Y.Prtn() = elemPrtn_;
  pluck_Y.SetComm(domain_.colComm);
  pluck_Y.LocalMap()[key].Resize(local_height, local_width);

  // Initialize the distributed matrices
  blas::Copy(local_pluck_sz, 
      eigvecs_local.Data() + local_height * band_distributor.current_proc_start, 
      1,
      pluck_X.LocalMap()[key].Data(),
      1);

  SetValue(pluck_Y.LocalMap()[key], std::complex<double>(0.0)); // pluck_Y is initialized to 0 

  GetTime( extra_timeSta );

  scfdg_hamiltonian_times_distmat_Cpx(pluck_X, pluck_Y); // Y = H * X

  GetTime( extra_timeEnd );

  statusOFS << std::endl << " Hamiltonian times eigenvectors calculation time = " 
    << (extra_timeEnd - extra_timeSta ) << " s.";

  // Copy pluck_Y to result_mat after preparing it
  GetTime( extra_timeSta );

  CpxNumMat temp_buffer;
  temp_buffer.Resize(eigvecs_local.m(), eigvecs_local.n());
  SetValue(temp_buffer, std::complex<double>(0.0));

  blas::Copy(local_pluck_sz, 
      pluck_Y.LocalMap()[key].Data(),
      1,
      temp_buffer.Data() + local_height * band_distributor.current_proc_start,
      1);

  // Empty out the results and prepare the distributed matrix
  result_mat.LocalMap().clear();
  result_mat.Prtn() = elemPrtn_;
  result_mat.SetComm(domain_.colComm);
  result_mat.LocalMap()[key].Resize(eigvecs_local.m(), eigvecs_local.n());

  MPI_Allreduce(temp_buffer.Data(),
      result_mat.LocalMap()[key].Data(),
      (eigvecs_local.m() * eigvecs_local.n()),
      MPI_DOUBLE_COMPLEX,
      MPI_SUM,
      domain_.rowComm);

  GetTime( extra_timeEnd );
  statusOFS << std::endl << " Eigenvector block rebuild time = " 
    << (extra_timeEnd - extra_timeSta ) << " s.";
} // End of scfdg_Hamiltonian_times_eigenvectors_Cpx


void
       SCFDG::scfdg_distmat_update_Cpx(DistVec<Index3, CpxNumMat, ElemPrtn>  &dist_mat_a,
           std::complex<double> scal_a,
           DistVec<Index3, CpxNumMat, ElemPrtn>  &dist_mat_b,
           std::complex<double> scal_b)
       {

         CpxNumMat& mat_a= (dist_mat_a.LocalMap().begin())->second;
         CpxNumMat& mat_b= (dist_mat_b.LocalMap().begin())->second;

         std::complex<double> *ptr_a = mat_a.Data();
         std::complex<double> *ptr_b = mat_b.Data();

         // Conformity check
         if( (dist_mat_a.LocalMap().size() != 1) ||
             (dist_mat_b.LocalMap().size() != 1) ||
             (mat_a.m() != mat_b.m()) ||
             (mat_a.n() != mat_b.n()) )
         {
           statusOFS << std::endl << " Non-conforming distributed vectors / matrices in update routine !!"
             << std::endl << " Aborting ... " << std::endl;
           exit(1);
         }

         for(Int iter = 0; iter < (mat_a.m() * mat_a.n()) ; iter ++)
           ptr_b[iter] = scal_a * ptr_a[iter] + scal_b * ptr_b[iter];



       } // End of routine scfdg_distvec_update_Cpx


void SCFDG::UpdateEkin(DistVec<Index3, CpxNumMat, ElemPrtn>  &X,DistVec<Index3, CpxNumMat, ElemPrtn>  &HX)
{
CpxNumMat& mat_X= (X.LocalMap().begin())->second;
CpxNumMat& mat_HX= (HX.LocalMap().begin())->second;

std::complex<double> *ptr_X = mat_X.Data();
std::complex<double> *ptr_HX = mat_HX.Data();
 
std::complex<double> complexOne (1.0,0.0);
std::complex<double> complexZero (0.0,0.0);
std::complex<double> complexI(0.0,1.0);
// Conformity check
if( (X.LocalMap().size() != 1) ||
    (HX.LocalMap().size() != 1) ||
    (mat_X.m() != mat_HX.m()) ||
    (mat_X.n() != mat_HX.n()) )
{
  statusOFS << std::endl << " Non-conforming distributed vectors / matrices in update routine !!"
    << std::endl << " Aborting ... " << std::endl;
  exit(1);
}


int m=mat_HX.m();
int n=mat_HX.n();
std::complex<double>  EkinLocal(0.0,0.0);

//std::cout<<"n: "<<n<<std::endl;

CpxNumVec eigvalCpxLocal(n);

CpxNumVec eigvalCpx(n);

CpxNumVec normCoef(n);

CpxNumVec normCoefLocal(n);


SetValue(eigvalCpx,std::complex<double>(0.0));
SetValue(normCoef,std::complex<double>(0.0));

for(int i=0;i<mat_HX.n();i++)
{
//因为乘法里k1是乘了-i的
//blas::Gemm( 'C', 'N', 1, 1, m, complexI, &(mat_X.Data()[i*m]),
//    m, &(mat_HX.Data()[i*m]), m, complexOne, &EkinLocal, 1 );


blas::Gemm( 'C', 'N', 1, 1, m, complexI, &(mat_X.Data()[i*m]),
    m, &(mat_HX.Data()[i*m]), m, complexZero, &eigvalCpxLocal(i), 1 );



 blas::Gemm( 'C', 'N', 1, 1, m, complexOne, &(mat_X.Data()[i*m]),
     m, &(mat_X.Data()[i*m]), m, complexZero, &normCoefLocal(i), 1 );





}


Int mpirank, mpisize;
MPI_Comm_rank( domain_.comm, &mpirank );
MPI_Comm_size( domain_.comm, &mpisize );



MPI_Allreduce(normCoefLocal.Data(),
    normCoef.Data(),
    n,
    MPI_DOUBLE_COMPLEX,
    MPI_SUM,
    domain_.colComm);



MPI_Allreduce(eigvalCpxLocal.Data(),
    eigvalCpx.Data(),
    n,
    MPI_DOUBLE_COMPLEX,
    MPI_SUM,
    domain_.colComm);


//if(mpirank==0)
//{
//for(int i=0;i<n;i++)
//{
//std::cout<<"NormCoef: "<<normCoef(i)<<std::endl;
//}
//}


for(int i=0;i<n;i++)
{
((*hamDGPtr_).EigVal())(i)=(eigvalCpx(i).real()/normCoef(i).real());
}


MPI_Barrier( domain_.comm );

//double EkinReal=EkinLocal.real();
/*
 MPI_Allreduce(&EkinReal,
     &Ekin_,
     1,
     MPI_DOUBLE,
     MPI_SUM,
     domain_.colComm);
*/
return;

}





       void SCFDG::UpdateHam()
{
HamiltonianDG&  hamDG = *hamDGPtr_;
double timeSta=0.0;
double timeEnd =0.0;

Int mpirank, mpisize;
MPI_Comm_rank( domain_.comm, &mpirank );
MPI_Comm_size( domain_.comm, &mpisize );


//updata ham
                if( XCType_ == "XC_GGA_XC_PBE" ){
                  GetTime( timeSta );
                  hamDG.CalculateGradDensity(  *distfftPtr_ );
                  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                  statusOFS << " Time for calculating gradient of density is " <<
                    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
                }

                GetTime( timeSta );
                hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), *distfftPtr_ );
                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << " Time for computing Exc in the global domain is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

                GetTime( timeSta );

                hamDG.CalculateHartree( hamDG.Vhart(), *distfftPtr_ );

                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << " Time for computing Vhart in the global domain is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
               // Compute the second order accurate energy functional.

               // Compute the second order accurate energy functional.
               // NOTE: In computing the second order energy, the density and the
               // potential must be the OUTPUT density and potential without ANY
               // MIXING.
//               CalculateSecondOrderEnergy();

               // Compute the KS energy 

  //             GetTime( timeSta );

//               CalculateKSEnergy();

//               GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//               statusOFS << " Time for computing KSEnergy in the global domain is " <<
//                 timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif

               // Update the total potential AFTER updating the energy

               // No external potential

               // Compute the new total potential

               GetTime( timeSta );
               hamDG.CalculateVtot( hamDG.Vtot() );

//fengjw undata New LGL 
    // Save the old potential on the LGL grid
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            Index3 numLGLGrid     = hamDG.NumLGLGridElem();
            blas::Copy( numLGLGrid.prod(),
                hamDG.VtotLGL().LocalMap()[key].Data(), 1,
                vtotLGLSave_.LocalMap()[key].Data(), 1 );
          } // if (own this element)
        } // for (i)

    // Update the local potential on the extended element and on the
    // element.
    UpdateElemLocalPotential();

    // Save the difference of the potential on the LGL grid into vtotLGLSave_
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            Index3 numLGLGrid     = hamDG.NumLGLGridElem();
            Real *ptrNew = hamDG.VtotLGL().LocalMap()[key].Data();
            Real *ptrDif = vtotLGLSave_.LocalMap()[key].Data();
            for( Int p = 0; p < numLGLGrid.prod(); p++ ){
              (*ptrDif) = (*ptrNew) - (*ptrDif);
              ptrNew++;
              ptrDif++;
            }
          } // if (own this element)
        } // for (i)                    


              MPI_Barrier( domain_.comm );
              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << " Time for updating the local potential in the extended element and the element is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif





              // Update the DG Matrix
              GetTime(timeSta);
              hamDG.UpdateDGMatrix_Cpx( vtotLGLSave_ );
MPI_Barrier( domain_.comm );
	      GetTime(timeEnd);
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << " Time for UpdateDGMatrix:   " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


/*
	GetTime(timeSta);
  for( std::map<ElemMatKey, DblNumMat>::iterator
      mi  = hamDG.HMat().LocalMap().begin();
      mi != hamDG.HMat().LocalMap().end(); ++mi ){
    ElemMatKey key = (*mi).first;
        int m=(hamDG.HMat().LocalMap().find(key))->second.m();
        int n=(hamDG.HMat().LocalMap().find(key))->second.n();

        for(int ntotidx=0;ntotidx<m*n;ntotidx++)
        {

        hamDG.HMatCpx().LocalMap()[key].Data()[ntotidx]=std::complex<double>(((hamDG.HMat().LocalMap().find(key))->second.Data())[ntotidx],0.0);

        }
  }



              MPI_Barrier( domain_.comm );
              GetTime( timeEnd );

#if ( _DEBUGlevel_ >= 0 )
              statusOFS << " Time for CopyDGMatrix for Complex:   " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
*/


return;
}




void SCFDG::CalculateDipole()
{

Int mpirank, mpisize,mpicolrank;
MPI_Comm_rank( domain_.comm, &mpirank );
MPI_Comm_rank( domain_.colComm, &mpicolrank );
MPI_Comm_size( domain_.comm, &mpisize );


HamiltonianDG&  hamDG = *hamDGPtr_;

double Dx=0.0;
double Dy=0.0;
double Dz=0.0;

double DxLocal=0.0;
double DyLocal=0.0;
double DzLocal=0.0;

////////
      for( Int k = 0; k < numElem_[2]; k++ ) 
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	            
		    DblNumVec& densityLocal=hamDG.Density().LocalMap()[key];
		    DblNumVec& pseCharge= hamDG.PseudoCharge().LocalMap()[key];
                    std::vector<DblNumVec>& grid = hamDG.UniformGridElemFine()(i, j, k);
		    int zmax=hamDG.NumUniformGridElemFine()[2];
		    int ymax=hamDG.NumUniformGridElemFine()[1];
		    int xmax=hamDG.NumUniformGridElemFine()[0];
		for(int z=0;z<zmax;z++)
			for(int y=0;y<ymax;y++)
			  for(int x=0;x<xmax;x++)
				{
					DxLocal=DxLocal+(pseCharge.Data()[x+y*xmax+z*xmax*ymax]-densityLocal.Data()[x+y*xmax+z*xmax*ymax])*(grid[0]).Data()[x];
					DyLocal=DyLocal+(pseCharge.Data()[x+y*xmax+z*xmax*ymax]-densityLocal.Data()[x+y*xmax+z*xmax*ymax])*(grid[1]).Data()[y];
					DzLocal=DzLocal+(pseCharge.Data()[x+y*xmax+z*xmax*ymax]-densityLocal.Data()[x+y*xmax+z*xmax*ymax])*(grid[2]).Data()[z];
		
				}				

//                    for( Int d = 0; d < DIM; d++ ){
//                    numUniformGridElemFine() 

//}

                    
}
}
// mpi::Allreduce( &sumDensityLocal, &sumDensity, 1, MPI_SUM,
//      domain_.colComm );
//



MPI_Allreduce(&DxLocal,&Dx,1,MPI_DOUBLE,MPI_SUM, domain_.colComm);
MPI_Allreduce(&DyLocal,&Dy,1,MPI_DOUBLE,MPI_SUM, domain_.colComm);
MPI_Allreduce(&DzLocal,&Dz,1,MPI_DOUBLE,MPI_SUM, domain_.colComm);

//MPI_Bcast( &Dx, 1, MPI_DOUBLE, 0, domain_.colComm );
//MPI_Bcast( &Dy, 1, MPI_DOUBLE, 0, domain_.colComm );
//MPI_Bcast( &Dz, 1, MPI_DOUBLE, 0, domain_.colComm );


//double fac=(domain_.length[0]*domain_.length[1]*domain_.length[2])/(domain_.numGridFine[0]*domain_.numGridFine[1]*domain_.numGridFine[2]);
double fac=domain_.Volume() / domain_.NumGridTotalFine();
//sumRhoLocal *= domain_.Volume() / domain_.NumGridTotalFine();
//mpi::Allreduce( &sumRhoLocal, &sumRho, 1, MPI_SUM, domain_.colComm );


Dx=Dx*fac;
Dy=Dy*fac;
Dz=Dz*fac;

if(1)
{
if(mpirank==0)
{
std::cout<<"Dx: " <<Dx<<"         "<<"Dy: "<<Dy<<"         "<<"Dz "<<Dz<<"          "<<std::endl;
//std::cout<<Dy<<std::endl;
}

}


///////

return;
}



void SCFDG::scfdg_hamiltonian_times_distmat_Cpx(DistVec<Index3, CpxNumMat, ElemPrtn>  &my_dist_mat,
    DistVec<Index3, CpxNumMat, ElemPrtn>  &Hmat_times_my_dist_mat)
     {

       Int mpirank, mpisize;
       MPI_Comm_rank( domain_.comm, &mpirank );
       MPI_Comm_size( domain_.comm, &mpisize );

       HamiltonianDG&  hamDG = *hamDGPtr_;
       std::vector<Index3>  getKeys_list;

std::complex<double> NegativecomplexOne (0.0,-1.0);
std::complex<double> complexOne (1.0,0.0);



                 SetValue(Hmat_times_my_dist_mat.LocalMap().begin()->second, std::complex<double>(0.0)); // pluck_Y is initialized to 0



       // Check that vectors provided only contain one entry in the local map
       // This is a safeguard to ensure that we are really dealing with distributed matrices
       if((my_dist_mat.LocalMap().size() != 1) ||
           (Hmat_times_my_dist_mat.LocalMap().size() != 1) ||
           ((my_dist_mat.LocalMap().begin())->first != (Hmat_times_my_dist_mat.LocalMap().begin())->first))
       {
         statusOFS << std::endl << " Vectors in Hmat * vector_block product not formatted correctly !!"
           << std::endl << " Aborting ... " << std::endl;
         exit(1);
       }


       // Obtain key based on my_dist_mat : This assumes that my_dist_mat is formatted correctly
       // based on processor number, etc.
       Index3 key = (my_dist_mat.LocalMap().begin())->first;

       // Obtain keys of neighbors using the Hamiltonian matrix
       // We only use these keys to minimize communication in GetBegin since other parts of the vector
       // block, even if they are non-zero will get multiplied by the zero block of the Hamiltonian
       // anyway.
       for(typename std::map<ElemMatKey, CpxNumMat >::iterator
           get_neighbors_from_Ham_iterator = hamDG.HMatCpx().LocalMap().begin();
           get_neighbors_from_Ham_iterator != hamDG.HMatCpx().LocalMap().end();
           ++get_neighbors_from_Ham_iterator)
       {
         Index3 neighbor_key = (get_neighbors_from_Ham_iterator->first).second;

         if(neighbor_key == key)
           continue;
         else
           getKeys_list.push_back(neighbor_key);
       }


       // Do the communication necessary to get the information from
       // procs holding the neighbors
       // Supposedly, independent row communicators (i.e. colComm)
       //  are being used for this
       my_dist_mat.GetBegin( getKeys_list, NO_MASK );
       my_dist_mat.GetEnd( NO_MASK );


       // Obtain a reference to the chunk where we want to store
       CpxNumMat& mat_Y_local = Hmat_times_my_dist_mat.LocalMap()[key];


       // Now pluck out relevant chunks of the Hamiltonian and the vector and multiply
       for(typename std::map<Index3, CpxNumMat >::iterator
           mat_X_iterator = my_dist_mat.LocalMap().begin();
           mat_X_iterator != my_dist_mat.LocalMap().end(); ++mat_X_iterator ){

         Index3 iter_key = mat_X_iterator->first;
         CpxNumMat& mat_X_local = mat_X_iterator->second; // Chunk from input block of vectors

         // Create key for looking up Hamiltonian chunk 
         ElemMatKey myelemmatkey = std::make_pair(key, iter_key);

         std::map<ElemMatKey, CpxNumMat >::iterator ham_iterator = hamDG.HMatCpx().LocalMap().find(myelemmatkey);

         //statusOFS << std::endl << " Working on key " << key << "   " << iter_key << std::endl;

         // Now do the actual multiplication
         CpxNumMat& mat_H_local = ham_iterator->second; // Chunk from Hamiltonian

         Int m = mat_H_local.m(), n = mat_X_local.n(), k = mat_H_local.n();

         blas::Gemm( 'N', 'N', m, n, k,
             NegativecomplexOne, mat_H_local.Data(), m,
             mat_X_local.Data(), k,
             complexOne, mat_Y_local.Data(), m);


       } // End of loop using mat_X_iterator

       // Matrix * vector_block product is ready now ... 
       // Need to clean up extra entries in my_dist_mat
       typename std::map<Index3, CpxNumMat >::iterator it;
       for(Int delete_iter = 0; delete_iter <  getKeys_list.size(); delete_iter ++)
       {
         it = my_dist_mat.LocalMap().find(getKeys_list[delete_iter]);
         (my_dist_mat.LocalMap()).erase(it);
       }


     }

void SCFDG::AddVext(int direction,double E,double t)
{
Int mpirank, mpisize;
MPI_Comm_rank( domain_.comm, &mpirank );
MPI_Comm_size( domain_.comm, &mpisize );
HamiltonianDG&  hamDG = *hamDGPtr_;

//std::complex<double> expfac;


//double amp=E;
double amp=0.0194;
//double amp=0.0;


double freq=18.0/27.211385;
//double freq =0.182;
double phase=0.0;
double tau=13.6056925;
double t0=13.6056925;
//double t0=2.6056925;
//double t0 =0.011;
double temp=(t-t0)/tau;
double et=amp*std::exp(-temp*temp/2.0)*std::sin(freq*t+phase);


/*
 esdfParam.isTDDFT            = yaml_integer( "TDDFT",   0);
 esdfParam.restartTDDFTStep   = yaml_integer( "Restart_TDDFT_Step", 0 );
 esdfParam.TDDFTautoSaveSteps = yaml_integer( "TDDFT_AUTO_SAVE_STEP", 20);
 esdfParam.isTDDFTEhrenfest   = yaml_integer( "TDDFT_EHRENFEST", 1);
 esdfParam.isTDDFTVext        = yaml_integer( "TDDFT_VEXT",   1);
 esdfParam.isTDDFTDipole      = yaml_integer( "TDDFT_DIPOLE",   1);
 esdfParam.TDDFTVextPolx      = yaml_double( "TDDFT_VEXT_POLX", 1.0);
 esdfParam.TDDFTVextPoly      = yaml_double( "TDDFT_VEXT_POLY", 0.0);
 esdfParam.TDDFTVextPolz      = yaml_double( "TDDFT_VEXT_POLZ", 0.0);
 esdfParam.TDDFTVextFreq      = yaml_double( "TDDFT_VEXT_FREQ", 18.0/27.211385);
 esdfParam.TDDFTVextPhase     = yaml_double( "TDDFT_VEXT_PHASE",0.0);
 esdfParam.TDDFTVextAmp       = yaml_double( "TDDFT_VEXT_AMP",  0.0194);
 esdfParam.TDDFTVextT0        = yaml_double( "TDDFT_VEXT_T0",   13.6056925);
 esdfParam.TDDFTVextTau       = yaml_double( "TDDFT_VEXT_TAU",  13.6056925);
*/










   for( Int k = 0; k < numElem_[2]; k++ )
     for( Int j = 0; j < numElem_[1]; j++ )
       for( Int i = 0; i < numElem_[0]; i++ ){
         Index3 key = Index3( i, j, k );
         if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
           hamDG.Vext().LocalMap()[key];
	   std::vector<DblNumVec>& grid = hamDG.UniformGridElemFine()(i, j, k);


if(direction==-1)
{
SetValue(hamDG.Vext().LocalMap()[key],10.0);

}
else
{
     int zmax=hamDG.NumUniformGridElemFine()[2];
     int ymax=hamDG.NumUniformGridElemFine()[1];
     int xmax=hamDG.NumUniformGridElemFine()[0];
 for(int z=0;z<zmax;z++)
         for(int y=0;y<ymax;y++)
           for(int x=0;x<xmax;x++)
                 {
				
			 if(direction==0)
			{
                         hamDG.Vext().LocalMap()[key].Data()[x+y*xmax+z*xmax*ymax]=et*(grid[0]).Data()[x];
			 //expfac=std::complex<double>(0.0,E*(grid[0]).Data()[x]);
			 //hamDG.EigvecCoefCpx().LocalMap()[key].Data()[x+y*xmax+z*xmax*ymax]*=std::exp(expfac);
			 }
			if(direction==1)
			{
			//expfac=std::complex<double>(0.0,E*(grid[1]).Data()[y]);
			hamDG.Vext().LocalMap()[key].Data()[x+y*xmax+z*xmax*ymax]=et*(grid[1]).Data()[y];	
			//hamDG.EigvecCoefCpx().LocalMap()[key].Data()[x+y*xmax+z*xmax*ymax]*=std::exp(expfac);
			}

			if(direction==2)
			{
			hamDG.Vext().LocalMap()[key].Data()[x+y*xmax+z*xmax*ymax]=et*(grid[2]).Data()[z];
			//expfac=std::complex<double>(0.0,E*(grid[2]).Data()[z]);
			//hamDG.EigvecCoefCpx().LocalMap()[key].Data()[x+y*xmax+z*xmax*ymax]*=std::exp(expfac);
			}                 
}


}
/*
//dachu
if(mpirank==0)
{
int ind=hamDG.Vext().LocalMap()[key].m();
for(int i=0;i<m;i++)
std::cout<<hamDG.Vext().LocalMap()[key].Data()[i]<<std::endl;
//
}
MPI_Barrier( domain_.comm );
if(mpirank==1)
{
int ind=hamDG.Vext().LocalMap()[key].m();
for(int i=0;i<m;i++)
std::cout<<hamDG.Vext().LocalMap()[key].Data()[i]<<std::endl;
//
}
MPI_Barrier( domain_.comm );
//

//
*/


}
}







return;
}




} // namespace dgdft
