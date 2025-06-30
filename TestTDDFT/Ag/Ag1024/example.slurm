#!/bin/bash

#SBATCH -J hyj_Ag1024             # 指定作业名称
#SBATCH -p normal                 # 指定队列/分区名称
#SBATCH -n 512                    # 指定任务数量（总 CPU 核心数）
#SBATCH -N 32                     # 指定节点数量
#SBATCH --ntasks-per-node=8       # 每节点的任务数量（对应原命令中的 -cgsp 8）
#SBATCH --mem=15G                 # 每节点申请的内存大小（根据原命令中的 -share_size 15000 转换为 15GB）
#SBATCH --time=00:30:00           # 指定作业运行时间（假设默认运行时间为 30 分钟）
#SBATCH --gres=dcu:4              # 指定每节点申请的加速卡数量
#SBATCH --exclusive               # 独占节点（对应原命令中的 -exclu）
#SBATCH -o runlog                 # 指定标准输出文件名称
#SBATCH -e runlog                 # 指定错误输出文件名称

# 显示作业信息
echo "Start time: `date`"                           # 显示开始时间
echo "SLURM_JOB_ID: $SLURM_JOB_ID"                  # 显示作业号
echo "SLURM_NNODES: $SLURM_NNODES"                  # 显示节点数
echo "SLURM_NTASKS: $SLURM_NTASKS"                  # 显示总任务数
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"  # 显示每节点的任务数
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"    # 显示每个任务使用的 CPU 数量
echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION"    # 显示队列/分区名称
echo "SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR"           # 显示提交作业目录的路径
echo "SLURM_NODELIST:$SLURM_NODELIST"               # 显示执行节点列表名称

# 设置环境变量
export MPITYPE=pmix_v3
export LD_LIBRARY_PATH=/public/home/acsa/weihu/yaoyf/software/pwdft-lib/lib/fftw-3.3.10/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/hpc/software/compiler/intel/intel-compiler-2017.5.239/mkl/lib/intel64:$LD_LIBRARY_PATH

# 执行程序
srun --mpi=pmix_v3 /public/home/acsa/hyj/DGTDDFT-GPU/examples/dghf

# 显示结束时间
echo "End time: `date`"
