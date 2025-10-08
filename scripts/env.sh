module purge
module load miniconda3
conda activate hw2

module load gcc/13
module load openmpi
export UCX_NET_DEVICES=mlx5_0:1
