#!/bin/bash
#SBATCH -J skeptical-editor
#SBATCH -A c02046
#SBATCH -p gpu-interactive
#SBATCH -o logs/job_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# 1. Start clean
module purge

# 2. Attempt to load the deeplearning or stack modules
# These are the most reliable 'parent' modules on Quartz
module load deeplearning || module load anaconda/python3.11 || module load gcc-stack

# 3. Load SQLite (if it's still missing, we'll use the system version)
module load sqlite

# 4. Activate your environment using the ABSOLUTE path
# This bypasses the 'module load python' error entirely
source /geode2/home/u050/kupokh/Quartz/envs/dl_env/bin/activate

# 5. Fix the Import Error (The relative import issue)
# We move to the project root and run as a module
cd /geode2/home/u050/kupokh/Quartz/Skeptical-Editor
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 6. Run the code 
python -m src.train --data-dir data/coco_correction --batch-size 32 --num-epochs 20 --pretrained-resnet --freeze-image-encoder
# python -m src.test --checkpoint runs/coco_correction/checkpoints/best.pt
# python -m src.saliency_aggregate --checkpoint runs/coco_correction/checkpoints/best.pt
# python -m src.gradcam --checkpoint runs/coco_correction/checkpoints/best.pt --data-dir data/coco_correction
# python -m src.vit-gradcam --checkpoint runs/coco_correction/checkpoints/best.pt --data-dir data/coco_correction
# python -m src.tune