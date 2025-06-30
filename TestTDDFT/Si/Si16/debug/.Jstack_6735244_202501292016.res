
==============================
  T0 -> 0.0-35.63
  T1 -> 0-35
------------------------------
 -0- /home/export/online1/mdt00/shisuan/swustcfd/fengjw/DG-TDDF/DGTDDFT/examples/dghf 
   -1- slave__Waiting_For_Task ([2304/2304] T0 24.63/v35 24.62/v35 24.61/v35...)
   -1- main at dgdft.cpp:414 ([0/36])
     -2- dgdft::SCFDG::Iterate at scf_dg.cpp:1322 ([0/36])
       -3- dgdft::SCFDG::RTTDDFT_RK4 at dgtddft.cpp:938 ([0/36])
         -4- dgdft::SCFDG::scfdg_hamiltonian_times_distmat_Cpx at dgtddft.cpp:2682 ([0/36])
           -5- PMPI_Wait ([36/36] T1 24/v35 25/v35 26/v35...)
==============================

==========================================================
node(taskid):svrstart,wait,sout
vn000032(0       ):	0.57	13.60	13.62
vn000033(6       ):	0.59	14.63	14.63
vn000036(12      ):	0.56	14.58	14.59
vn000037(18      ):	0.58	13.61	13.61
vn000035(24      ):	0.54	13.57	13.57
vn000039(30      ):	0.57	13.62	13.62
before:0.014097,scan:31.009172,first_data:30.008992,process:1.000263,show:0.057141,total:31.080493

