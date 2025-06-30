
==============================
  T0 -> 0.0-35.63
  T1 -> 0-35
------------------------------
 -0- /home/export/online1/mdt00/shisuan/swustcfd/fengjw/DG-TDDF/DGTDDFT/examples/dghf 
   -1- slave__Waiting_For_Task ([2304/2304] T0 18.63/v11 18.62/v11 18.61/v11...)
   -1- main at dgdft.cpp:414 ([0/36])
     -2- dgdft::SCFDG::Iterate at scf_dg.cpp:1322 ([0/36])
       -3- dgdft::SCFDG::RTTDDFT_RK4 at dgtddft.cpp:938 ([0/36])
         -4- dgdft::SCFDG::scfdg_hamiltonian_times_distmat_Cpx at dgtddft.cpp:2698 ([0/36])
           -5- PMPI_Waitall ([36/36] T1 18/v11 19/v11 20/v11...)
==============================

==========================================================
node(taskid):svrstart,wait,sout
vn000008(0       ):	0.63	14.66	14.67
vn000009(6       ):	0.76	13.79	13.79
vn000010(12      ):	0.68	13.70	13.70
vn000011(18      ):	0.61	13.63	13.64
vn000002(24      ):	0.64	13.69	13.69
vn000007(30      ):	0.67	13.70	13.70
before:0.019683,scan:31.008790,first_data:30.008603,process:1.000314,show:0.018681,total:31.047281

