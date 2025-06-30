#!/bin/bash
bsub -b -J hyj_LAOSTO -o runlog -exclu  -share_size 15000 -host_stack 2048 -n 512 -cgsp 64 -q  q_ustc  /public/home/acsa/hyj/DGTDDFT-GPU/examples/dghf