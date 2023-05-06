#!/bin/bash
#SBATCH -J Final_proj
#SBATCH -A ACF-UTK0011
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --ntasks=48
#SBATCH --partition=campus
#SBATCH --qos=campus
#SBATCH --time=001:00:00
#SBATCH -e Final_proj%j
#SBATCH -o Final_proj%j
echo $SLURM_JOB_NODELIST
echo $SLURM_NTASKS
#module load quantum-espresso/6.6-intel
conda activate /lustre/isaac/scratch/ahoust17/gpaw
python3 for_isaac_run

#./THIS_WILL_RUN_AN_EXECUTABLE_IN_CURRENT_DIR
