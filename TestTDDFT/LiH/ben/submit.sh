#! /bin/bash
bsub -b -J Ben  -o runlog  -timelimit 05:00:00 -exclu -share_size 15000 -host_stack 1024 -n 81 -cgsp 64 -q q_share /home/export/online1/mdt00/shisuan/swustcfd/fengjw/DG-TDDF/DGTDDFT/examples/dghf
