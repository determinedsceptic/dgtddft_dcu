#! /bin/bash
bsub -b -J fengjw_LAOSTO -o runlog -exclu  -share_size 15000 -host_stack 2048 -n 20480 -cgsp 64 -q  q_share  /home/export/online1/mdt00/shisuan/swustcfd/fengjw/DG-TDDF/DGTDDFT-MFFT/DGTDDFT/examples/dghf
