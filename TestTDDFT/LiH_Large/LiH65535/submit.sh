#! /bin/bash
bsub -b -J fengjw_LiH -o runlog -exclu  -share_size 15000 -host_stack 2048 -n 2048 -cgsp 64 -q  q_ustc  /home/export/online1/mdt00/shisuan/swustcfd/fengjw/DG-TDDF/DGTDDFT-MFFT/DGTDDFT/examples/dghf
