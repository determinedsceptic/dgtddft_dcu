#! /bin/bash
bsub -b -J LiNa16w  -o runlog  -timelimit 05:00:00 -exclu -share_size 15000 -host_stack 1024 -n 4096 -cgsp 64 -q q_share /home/export/online1/mdt00/shisuan/swustcfd/fengjw/DG-TDDF/TDDFT-DG-MFFT-HF/examples/dghf
