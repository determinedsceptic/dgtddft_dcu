
==============================
  T0 -> 0.0-35.63
  T1 -> 0-35
------------------------------
 -0- /home/export/online1/mdt00/shisuan/swustcfd/fengjw/DG-TDDF/DGTDDFT/examples/dghf 
   -1- slave__Waiting_For_Task ([2304/2304] T0 18.63/v15 18.62/v15 18.61/v15...)
   -1- main at dgdft.cpp:414 ([0/36])
     -2- dgdft::SCFDG::Iterate at scf_dg.cpp:1322 ([0/36])
       -3- dgdft::SCFDG::RTTDDFT_RK4 at dgtddft.cpp:938 ([0/36])
         -4- dgdft::SCFDG::scfdg_hamiltonian_times_distmat_Cpx at dgtddft.cpp:2703 ([0/36])
           -5- PMPI_Waitall ([36/36] T1 18/v15 19/v15 20/v15...)
==============================

==========================================================
node(taskid):svrstart,wait,sout
vn000012(0       ):	0.60	13.63	13.63
vn000013(6       ):	0.62	13.65	13.65
vn000014(12      ):	0.65	13.67	13.68
vn000015(18      ):	0.54	13.57	13.57
vn000016(24      ):	0.58	13.61	13.65
vn000017(30      ):	0.57	14.59	14.60
before:0.016949,scan:31.009517,first_data:30.009360,process:1.000238,show:0.063720,total:31.090267

