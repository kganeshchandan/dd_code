#!/bin/bash
#SBATCH -A research
#SBATCH -n 20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=op_file_fold0-protein-only.txt

PYTHONUNBUFFERED=1

# python3 training.py --input_dir HDF/ --output_prefix ./PUBLISHED_RESULTS/output --batch_size=5

# python3 training.py --input_dir SPLITS/LIGAND_ONLY_DATA0/ --output_prefix ./LIGAND_ONLY_FOLD0/output --batch_size=5 
python3 training.py --input_dir SPLITS/PROTEIN_ONLY_DATA0/ --output_prefix ./PROTEIN_ONLY_FOLD0/output --batch_size=5

# cd ~/pafnucy/LIGAND_ONLY_FOLD0
cd ~/pafnucy/PROTEIN_ONLY_FOLD0


rm *00001