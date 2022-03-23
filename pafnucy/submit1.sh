#!/bin/bash
#SBATCH -A research
#SBATCH -n 20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=op_file_fold1-protein-only.txt

PYTHONUNBUFFERED=1

# python3 training.py --input_dir HDF/ --output_prefix ./PUBLISHED_RESULTS/output --batch_size=5
python3 training.py --input_dir SPLITS/PROTEIN_ONLY_DATA1/ --output_prefix ./PROTEIN_ONLY_FOLD1/output --batch_size=5 


# cd ~/pafnucy/LIGAND_ONLY_FOLD1
cd ~/pafnucy/PROTEIN_ONLY_FOLD1

rm *00001