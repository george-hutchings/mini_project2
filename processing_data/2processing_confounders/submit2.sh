#!/bin/bash
#SBATCH --job-name=p_2_2        # job name, you can change this to any name to recongnise in slurm
#SBATCH --account=ms          # account name
##SBATCH --partition=long
#SBATCH --mem=20G            # RAM
#SBATCH --output=z-%x.%j.out
#SBATCH --error=z-%x.%j.err     # file to save the errors of the code
#SBATCH --time=01:00:00      # Time limit for your job, hrs:min:sec

module purge   # libraries used
module load TensorFlow/2.0.0-fosscuda-2019b-Python-3.7.4
module load Python/3.7.4-GCCcore-8.3.0
module load SimpleITK/1.2.4-foss-2019b-Python-3.7.4
module load scikit-learn/0.21.3-fosscuda-2019b-Python-3.7.4
module load scikit-image/0.16.2-fosscuda-2019b-Python-3.7.4
module load torchvision/0.7.0-fosscuda-2019b-Python-3.7.4-PyTorch-1.6.0
module load cleanlab/1.0-foss-2019b-Python-3.7.4
module load tqdm/4.41.1-GCCcore-8.3.0
module load NiBabel/3.2.0-foss-2019b-Python-3.7.4
module load OpenCV/4.2.0-foss-2019b-Python-3.7.4
module load timm/0.4.12-fosscuda-2019b-Python-3.7.4-PyTorch-1.6.0
module load h5py/2.10.0-fosscuda-2019b-Python-3.7.4
module load SciPy-bundle/2019.10-fosscuda-2019b-Python-3.7.4
module load Pillow/7.0.0-GCCcore-8.3.0-Python-3.7.4
module load pandas-plink/2.0.4-foss-2019b-Python-3.7.4
module load matplotlib/3.4.2-foss-2021a

python confounder_processing_std2.py
