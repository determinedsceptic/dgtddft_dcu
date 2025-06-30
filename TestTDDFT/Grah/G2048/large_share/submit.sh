#! /bin/bash
bsub -b -J fengjw_LiH -o runlog -exclu  -xmalloc -cross_size 86000 -share_size 400  -N 256 -cgsp 64 -mpecg 6 -q  q_ustc  /home/export/online1/mdt00/shisuan/swustcfd/fengjw/DG-TDDF/DGTDDFT/examples/dghf
