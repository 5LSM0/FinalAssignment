#!/bin/bash
#SBATCH --job-name="vis_data_job"
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --time 00:01:00
#SBATCH --partition=normal

# Load necessary modules
module load 2020
module load foss/2018b
module load Python/3.9.5-GCCcore-10.3.0
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0


# Copy necessary files to the temporary directory on the compute node
cp -r /home/scur0766 $TMPDIR

# Change to the temporary directory
cd $TMPDIR

# Run the executable
srun python train.py --data_path /home/scur0766/FinalAssignment

mkdir -p $HOME/FinalAssignment/results

cp dataset_info.txt Cityspace-test-vis.png $HOME/FinalAssignment/results