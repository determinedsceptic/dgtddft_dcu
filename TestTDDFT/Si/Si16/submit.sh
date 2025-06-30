#! /bin/bash
bsub -b -J fengjw_LiH -o runlog -exclu  -share_size 15000 -host_stack 2048 -n 36 -cgsp 64 -q  q_share  /home/export/online1/mdt00/shisuan/swustcfd/fengjw/DG-TDDF/DGTDDFT/examples/dghf
