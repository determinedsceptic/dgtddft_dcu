#! /bin/bash
#bsub -b -J fengjw_LiH -o runlog -exclu  -share_size 15000 -xmalloc -host_stack 2048 -n 256 -cgsp 64 -mpecg 6  -q  q_ustc  /home/export/online1/mdt00/shisuan/swustcfd/fengjw/DG-TDDF/DGTDDFT/examples/dghf


bsub -I -b -o runlog -exclu -share_size 400 -cache_size 32 -cross_size 86000 -q q_ustc -N 256 -cgsp 64 -mpecg 6 "$@" /home/export/online1/mdt00/shisuan/swustcfd/fengjw/DG-TDDF/DGTDDFT/examples/dghf
